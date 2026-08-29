#![cfg(feature = "rocm")]

//! Standalone GPU integration tests for the native HIP RMSNorm op.
//!
//! Expected values come from `mote_reference::rms_norm_f32`: inputs and weight
//! are quantized to F16 first, converted back to F32 for the oracle, and the
//! oracle output is re-quantized to F16 before comparison with the GPU
//! readback. `hidden = 513` deliberately exercises a row length that is not a
//! multiple of the kernel's 256-thread block.

use half::f16;
use mote_backend_hip::{HipContext, HipError, rms_norm_f16};
use mote_reference::rms_norm_f32;
use mote_types::{DType, Encoding, Layout, Shape, TensorDesc};

const ROWS: usize = 3;
const HIDDEN: usize = 513;
const EPSILON: f32 = 1e-6;

#[test]
fn hip_rms_norm_f16_matches_reference_for_hidden_513() {
    let context = HipContext::new(0).unwrap();

    // Quantize to F16 first, then widen back to F32: the oracle must see the
    // exact bit patterns the GPU reads.
    let input: Vec<f16> = (0..ROWS * HIDDEN)
        .map(|index| f16::from_f32(input_element(index)))
        .collect();
    let weight: Vec<f16> = (0..HIDDEN)
        .map(|index| f16::from_f32(weight_element(index)))
        .collect();
    let input_f32: Vec<f32> = input.iter().map(|value| value.to_f32()).collect();
    let weight_f32: Vec<f32> = weight.iter().map(|value| value.to_f32()).collect();
    let expected: Vec<f16> = rms_norm_f32(&input_f32, &weight_f32, HIDDEN, EPSILON)
        .unwrap()
        .iter()
        .map(|&value| f16::from_f32(value))
        .collect();

    let input_tensor = context
        .from_bytes(desc(&[ROWS, HIDDEN]), &encode_f16(&input))
        .unwrap();
    let weight_tensor = context
        .from_bytes(desc(&[HIDDEN]), &encode_f16(&weight))
        .unwrap();
    let output = context.empty(desc(&[ROWS, HIDDEN])).unwrap();

    rms_norm_f16(&context, &input_tensor, &weight_tensor, &output, EPSILON).unwrap();

    let actual = decode_f16(&context.read_bytes(&output).unwrap());
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        // Both sides accumulate in F32 and round once to F16. The tolerance
        // absorbs a few F16 ulps plus reduction-order noise.
        let actual = actual.to_f32();
        let expected = expected.to_f32();
        let tolerance = 3e-3_f32.max(expected.abs() * 3e-3);
        assert!(
            (actual - expected).abs() <= tolerance,
            "mismatch at {index}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn empty_tensors_return_without_launching() {
    let context = HipContext::new(0).unwrap();
    let weight = context
        .from_bytes(desc(&[HIDDEN]), &encode_f16(&vec![f16::ONE; HIDDEN]))
        .unwrap();

    // Zero rows.
    let input = context.empty(desc(&[0, HIDDEN])).unwrap();
    let output = context.empty(desc(&[0, HIDDEN])).unwrap();
    rms_norm_f16(&context, &input, &weight, &output, EPSILON).unwrap();
    assert!(context.read_bytes(&output).unwrap().is_empty());

    // Zero hidden size: the weight is empty too, and no kernel may launch.
    let input = context.empty(desc(&[2, 0])).unwrap();
    let weight = context.empty(desc(&[0])).unwrap();
    let output = context.empty(desc(&[2, 0])).unwrap();
    rms_norm_f16(&context, &input, &weight, &output, EPSILON).unwrap();
    assert!(context.read_bytes(&output).unwrap().is_empty());
}

#[test]
fn invalid_epsilon_is_rejected() {
    let context = HipContext::new(0).unwrap();
    let (input, weight, output) = fixtures(&context);

    for epsilon in [-EPSILON, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let error = rms_norm_f16(&context, &input, &weight, &output, epsilon).unwrap_err();
        assert!(
            matches!(error, HipError::InvalidEpsilon { .. }),
            "epsilon {epsilon} should be rejected, got {error:?}"
        );
    }
}

#[test]
fn shape_mismatches_are_rejected() {
    let context = HipContext::new(0).unwrap();
    let (input, weight, _) = fixtures(&context);

    let wrong_output = context.empty(desc(&[ROWS, HIDDEN + 1])).unwrap();
    let error = rms_norm_f16(&context, &input, &weight, &wrong_output, EPSILON).unwrap_err();
    assert!(
        matches!(
            error,
            HipError::ShapeMismatch {
                tensor: "output",
                ..
            }
        ),
        "{error:?}"
    );

    let wrong_weight = context.empty(desc(&[HIDDEN - 1])).unwrap();
    let output = context.empty(desc(&[ROWS, HIDDEN])).unwrap();
    let error = rms_norm_f16(&context, &input, &wrong_weight, &output, EPSILON).unwrap_err();
    assert!(
        matches!(
            error,
            HipError::ShapeMismatch {
                tensor: "weight",
                ..
            }
        ),
        "{error:?}"
    );
}

#[test]
fn tensors_from_another_context_are_rejected() {
    let context = HipContext::new(0).unwrap();
    let (input, weight, _) = fixtures(&context);

    let stranger = HipContext::new(0).unwrap();
    let output = stranger.empty(desc(&[ROWS, HIDDEN])).unwrap();
    let error = rms_norm_f16(&context, &input, &weight, &output, EPSILON).unwrap_err();
    assert!(matches!(error, HipError::WrongStorage), "{error:?}");
}

fn fixtures(context: &HipContext) -> (mote_core::Tensor, mote_core::Tensor, mote_core::Tensor) {
    let input: Vec<f16> = (0..ROWS * HIDDEN)
        .map(|index| f16::from_f32(input_element(index)))
        .collect();
    let weight: Vec<f16> = (0..HIDDEN)
        .map(|index| f16::from_f32(weight_element(index)))
        .collect();
    let input = context
        .from_bytes(desc(&[ROWS, HIDDEN]), &encode_f16(&input))
        .unwrap();
    let weight = context
        .from_bytes(desc(&[HIDDEN]), &encode_f16(&weight))
        .unwrap();
    let output = context.empty(desc(&[ROWS, HIDDEN])).unwrap();
    (input, weight, output)
}

fn input_element(index: usize) -> f32 {
    ((index * 37 + 5) % 19) as f32 / 6.0 - 1.5
}

fn weight_element(index: usize) -> f32 {
    0.5 + ((index * 7 + 3) % 13) as f32 / 12.0
}

fn encode_f16(values: &[f16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_ne_bytes())
        .collect()
}

fn decode_f16(bytes: &[u8]) -> Vec<f16> {
    let (words, remainder) = bytes.as_chunks::<2>();
    assert!(remainder.is_empty());
    words
        .iter()
        .copied()
        .map(|word| f16::from_bits(u16::from_ne_bytes(word)))
        .collect()
}

fn desc(dims: &[usize]) -> TensorDesc {
    TensorDesc::new(
        Shape::new(dims),
        Encoding::Plain(DType::F16),
        Layout::Contiguous,
    )
    .unwrap()
}

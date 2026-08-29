#![cfg(feature = "rocm")]

//! Standalone GPU integration tests for the native HIP fused residual-add +
//! RMSNorm op.
//!
//! Expected values come from `mote_reference::fused_add_rms_norm_f32`:
//! input, residual, and weight are quantized to F16 first, converted back to
//! F32 for the oracle, and both oracle results (the normalized rows and the
//! updated residual) are re-quantized to F16 before comparison with the GPU
//! readback. `hidden = 513` deliberately exercises a row length that is not a
//! multiple of the kernel's 256-thread block.

use half::f16;
use mote_backend_hip::{HipContext, HipError, fused_add_rms_norm_f16};
use mote_core::Tensor;
use mote_reference::fused_add_rms_norm_f32;
use mote_types::{DType, Encoding, Layout, Shape, TensorDesc};

const ROWS: usize = 3;
const HIDDEN: usize = 513;
const EPSILON: f32 = 1e-6;

#[test]
fn hip_fused_add_rms_norm_f16_matches_reference_for_hidden_513() {
    let context = HipContext::new(0).unwrap();

    // Quantize to F16 first, then widen back to F32: the oracle must see the
    // exact bit patterns the GPU reads.
    let input: Vec<f16> = (0..ROWS * HIDDEN)
        .map(|index| f16::from_f32(input_element(index)))
        .collect();
    let residual: Vec<f16> = (0..ROWS * HIDDEN)
        .map(|index| f16::from_f32(residual_element(index)))
        .collect();
    let weight: Vec<f16> = (0..HIDDEN)
        .map(|index| f16::from_f32(weight_element(index)))
        .collect();
    let input_f32: Vec<f32> = input.iter().map(|value| value.to_f32()).collect();
    let residual_f32: Vec<f32> = residual.iter().map(|value| value.to_f32()).collect();
    let weight_f32: Vec<f32> = weight.iter().map(|value| value.to_f32()).collect();
    let reference =
        fused_add_rms_norm_f32(&input_f32, &residual_f32, &weight_f32, HIDDEN, EPSILON).unwrap();
    let expected_normalized: Vec<f16> = reference
        .normalized
        .iter()
        .map(|&value| f16::from_f32(value))
        .collect();
    let expected_residual: Vec<f16> = reference
        .residual
        .iter()
        .map(|&value| f16::from_f32(value))
        .collect();

    let input_tensor = context
        .from_bytes(desc(&[ROWS, HIDDEN]), &encode_f16(&input))
        .unwrap();
    let residual_tensor = context
        .from_bytes(desc(&[ROWS, HIDDEN]), &encode_f16(&residual))
        .unwrap();
    let weight_tensor = context
        .from_bytes(desc(&[HIDDEN]), &encode_f16(&weight))
        .unwrap();
    let output = context.empty(desc(&[ROWS, HIDDEN])).unwrap();

    fused_add_rms_norm_f16(
        &context,
        &input_tensor,
        &residual_tensor,
        &weight_tensor,
        &output,
        EPSILON,
    )
    .unwrap();

    let actual_output = decode_f16(&context.read_bytes(&output).unwrap());
    assert_close(&actual_output, &expected_normalized, "normalized output");
    let actual_residual = decode_f16(&context.read_bytes(&residual_tensor).unwrap());
    assert_close(&actual_residual, &expected_residual, "updated residual");
}

#[test]
fn in_place_norm_with_output_aliasing_input_matches_reference() {
    let context = HipContext::new(0).unwrap();
    let (input_tensor, residual_tensor, weight_tensor, _, reference) = fixtures(&context);
    let expected_normalized: Vec<f16> = reference
        .normalized
        .iter()
        .map(|&value| f16::from_f32(value))
        .collect();
    let expected_residual: Vec<f16> = reference
        .residual
        .iter()
        .map(|&value| f16::from_f32(value))
        .collect();

    // `output` shares storage with `input`: the kernel computes the row
    // statistics before publishing, so the normalized rows may overwrite the
    // input in place.
    let output = input_tensor.clone();
    fused_add_rms_norm_f16(
        &context,
        &input_tensor,
        &residual_tensor,
        &weight_tensor,
        &output,
        EPSILON,
    )
    .unwrap();

    let actual_output = decode_f16(&context.read_bytes(&input_tensor).unwrap());
    assert_close(&actual_output, &expected_normalized, "in-place output");
    let actual_residual = decode_f16(&context.read_bytes(&residual_tensor).unwrap());
    assert_close(&actual_residual, &expected_residual, "updated residual");
}

#[test]
fn input_aliasing_residual_matches_reference() {
    let context = HipContext::new(0).unwrap();
    let (input_tensor, _, weight_tensor, input_f32, _, weight_f32) = small_fixtures(&context);

    // `residual` shares storage with `input`: every sum becomes
    // `input + input`, and the shared buffer ends up holding the F16 sums.
    let reference =
        fused_add_rms_norm_f32(&input_f32, &input_f32, &weight_f32, HIDDEN, EPSILON).unwrap();
    let expected_normalized: Vec<f16> = reference
        .normalized
        .iter()
        .map(|&value| f16::from_f32(value))
        .collect();
    let expected_residual: Vec<f16> = reference
        .residual
        .iter()
        .map(|&value| f16::from_f32(value))
        .collect();

    let residual = input_tensor.clone();
    let output = context.empty(desc(&[ROWS, HIDDEN])).unwrap();
    fused_add_rms_norm_f16(
        &context,
        &input_tensor,
        &residual,
        &weight_tensor,
        &output,
        EPSILON,
    )
    .unwrap();

    let actual_output = decode_f16(&context.read_bytes(&output).unwrap());
    assert_close(&actual_output, &expected_normalized, "normalized output");
    let actual_residual = decode_f16(&context.read_bytes(&input_tensor).unwrap());
    assert_close(&actual_residual, &expected_residual, "updated residual");
}

#[test]
fn output_aliasing_residual_is_rejected() {
    let context = HipContext::new(0).unwrap();
    let (input_tensor, residual_tensor, weight_tensor, _, _) = fixtures(&context);

    // One allocation cannot hold both the normalized row and the updated
    // residual, so the pairing must be rejected up front.
    let output = residual_tensor.clone();
    let error = fused_add_rms_norm_f16(
        &context,
        &input_tensor,
        &residual_tensor,
        &weight_tensor,
        &output,
        EPSILON,
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            HipError::AliasedStorage {
                tensor: "output",
                other: "residual"
            }
        ),
        "{error:?}"
    );
}

#[test]
fn weight_aliasing_writable_outputs_is_rejected() {
    let context = HipContext::new(0).unwrap();
    let (input_tensor, residual_tensor, _, output, _) = fixtures(&context);

    // A rank-1 weight view over the output storage: the kernel reads the
    // weight while writing the normalized rows, so sharing is rejected.
    let aliased_weight = Tensor::new(desc(&[HIDDEN]), output.storage().clone(), 0).unwrap();
    assert!(aliased_weight.shares_storage_with(&output));
    let error = fused_add_rms_norm_f16(
        &context,
        &input_tensor,
        &residual_tensor,
        &aliased_weight,
        &output,
        EPSILON,
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            HipError::AliasedStorage {
                tensor: "weight",
                other: "output"
            }
        ),
        "{error:?}"
    );

    // The same conflict exists between the weight and the in/out residual.
    let aliased_weight =
        Tensor::new(desc(&[HIDDEN]), residual_tensor.storage().clone(), 0).unwrap();
    assert!(aliased_weight.shares_storage_with(&residual_tensor));
    let error = fused_add_rms_norm_f16(
        &context,
        &input_tensor,
        &residual_tensor,
        &aliased_weight,
        &output,
        EPSILON,
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            HipError::AliasedStorage {
                tensor: "weight",
                other: "residual"
            }
        ),
        "{error:?}"
    );
}

#[test]
fn invalid_epsilon_is_rejected() {
    let context = HipContext::new(0).unwrap();
    let (input_tensor, residual_tensor, weight_tensor, output, _) = fixtures(&context);

    for epsilon in [-EPSILON, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let error = fused_add_rms_norm_f16(
            &context,
            &input_tensor,
            &residual_tensor,
            &weight_tensor,
            &output,
            epsilon,
        )
        .unwrap_err();
        assert!(
            matches!(error, HipError::InvalidEpsilon { .. }),
            "epsilon {epsilon} should be rejected, got {error:?}"
        );
    }
}

#[test]
fn shape_mismatches_are_rejected() {
    let context = HipContext::new(0).unwrap();
    let (input_tensor, residual_tensor, weight_tensor, _, _) = fixtures(&context);

    let output = context.empty(desc(&[ROWS, HIDDEN])).unwrap();
    let wrong_output = context.empty(desc(&[ROWS, HIDDEN + 1])).unwrap();
    let error = fused_add_rms_norm_f16(
        &context,
        &input_tensor,
        &residual_tensor,
        &weight_tensor,
        &wrong_output,
        EPSILON,
    )
    .unwrap_err();
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

    let wrong_residual = context.empty(desc(&[ROWS + 1, HIDDEN])).unwrap();
    let error = fused_add_rms_norm_f16(
        &context,
        &input_tensor,
        &wrong_residual,
        &weight_tensor,
        &output,
        EPSILON,
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            HipError::ShapeMismatch {
                tensor: "residual",
                ..
            }
        ),
        "{error:?}"
    );

    let wrong_weight = context.empty(desc(&[HIDDEN - 1])).unwrap();
    let error = fused_add_rms_norm_f16(
        &context,
        &input_tensor,
        &residual_tensor,
        &wrong_weight,
        &output,
        EPSILON,
    )
    .unwrap_err();
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
fn wrong_dtype_is_rejected() {
    let context = HipContext::new(0).unwrap();
    let (input_tensor, residual_tensor, _, output, _) = fixtures(&context);

    let weight_tensor = context
        .from_bytes(
            TensorDesc::new(
                Shape::new(&[HIDDEN]),
                Encoding::Plain(DType::F32),
                Layout::Contiguous,
            )
            .unwrap(),
            &vec![0_u8; HIDDEN * 4],
        )
        .unwrap();
    let error = fused_add_rms_norm_f16(
        &context,
        &input_tensor,
        &residual_tensor,
        &weight_tensor,
        &output,
        EPSILON,
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            HipError::DTypeMismatch {
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
    let (input_tensor, residual_tensor, weight_tensor, _, _) = fixtures(&context);

    let stranger = HipContext::new(0).unwrap();
    let output = stranger.empty(desc(&[ROWS, HIDDEN])).unwrap();
    let error = fused_add_rms_norm_f16(
        &context,
        &input_tensor,
        &residual_tensor,
        &weight_tensor,
        &output,
        EPSILON,
    )
    .unwrap_err();
    assert!(matches!(error, HipError::WrongStorage), "{error:?}");
}

#[test]
fn empty_tensors_return_without_launching() {
    let context = HipContext::new(0).unwrap();
    let weight = context
        .from_bytes(desc(&[HIDDEN]), &encode_f16(&vec![f16::ONE; HIDDEN]))
        .unwrap();

    // Zero rows.
    let input = context.empty(desc(&[0, HIDDEN])).unwrap();
    let residual = context.empty(desc(&[0, HIDDEN])).unwrap();
    let output = context.empty(desc(&[0, HIDDEN])).unwrap();
    fused_add_rms_norm_f16(&context, &input, &residual, &weight, &output, EPSILON).unwrap();
    assert!(context.read_bytes(&output).unwrap().is_empty());
    assert!(context.read_bytes(&residual).unwrap().is_empty());

    // Zero hidden size: the weight is empty too, and no kernel may launch.
    let input = context.empty(desc(&[2, 0])).unwrap();
    let residual = context.empty(desc(&[2, 0])).unwrap();
    let weight = context.empty(desc(&[0])).unwrap();
    let output = context.empty(desc(&[2, 0])).unwrap();
    fused_add_rms_norm_f16(&context, &input, &residual, &weight, &output, EPSILON).unwrap();
    assert!(context.read_bytes(&output).unwrap().is_empty());
    assert!(context.read_bytes(&residual).unwrap().is_empty());
}

/// Uploads F16-quantized fixtures and computes the F32 oracle in one go.
type Fixtures = (
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    mote_reference::FusedAddRmsNormOutput,
);

fn fixtures(context: &HipContext) -> Fixtures {
    let (input_tensor, residual_tensor, weight_tensor, input_f32, residual_f32, weight_f32) =
        small_fixtures(context);
    let reference =
        fused_add_rms_norm_f32(&input_f32, &residual_f32, &weight_f32, HIDDEN, EPSILON).unwrap();
    let output = context.empty(desc(&[ROWS, HIDDEN])).unwrap();
    (
        input_tensor,
        residual_tensor,
        weight_tensor,
        output,
        reference,
    )
}

fn small_fixtures(context: &HipContext) -> (Tensor, Tensor, Tensor, Vec<f32>, Vec<f32>, Vec<f32>) {
    let input: Vec<f16> = (0..ROWS * HIDDEN)
        .map(|index| f16::from_f32(input_element(index)))
        .collect();
    let residual: Vec<f16> = (0..ROWS * HIDDEN)
        .map(|index| f16::from_f32(residual_element(index)))
        .collect();
    let weight: Vec<f16> = (0..HIDDEN)
        .map(|index| f16::from_f32(weight_element(index)))
        .collect();
    let input_f32: Vec<f32> = input.iter().map(|value| value.to_f32()).collect();
    let residual_f32: Vec<f32> = residual.iter().map(|value| value.to_f32()).collect();
    let weight_f32: Vec<f32> = weight.iter().map(|value| value.to_f32()).collect();
    let input_tensor = context
        .from_bytes(desc(&[ROWS, HIDDEN]), &encode_f16(&input))
        .unwrap();
    let residual_tensor = context
        .from_bytes(desc(&[ROWS, HIDDEN]), &encode_f16(&residual))
        .unwrap();
    let weight_tensor = context
        .from_bytes(desc(&[HIDDEN]), &encode_f16(&weight))
        .unwrap();
    (
        input_tensor,
        residual_tensor,
        weight_tensor,
        input_f32,
        residual_f32,
        weight_f32,
    )
}

fn assert_close(actual: &[f16], expected: &[f16], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what} length");
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        // Both sides accumulate in F32 and round once to F16. The tolerance
        // absorbs a few F16 ulps plus reduction-order noise.
        let actual = actual.to_f32();
        let expected = expected.to_f32();
        let tolerance = 3e-3_f32.max(expected.abs() * 3e-3);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{what} mismatch at {index}: expected {expected}, got {actual}"
        );
    }
}

fn input_element(index: usize) -> f32 {
    ((index * 37 + 5) % 19) as f32 / 6.0 - 1.5
}

fn residual_element(index: usize) -> f32 {
    ((index * 11 + 7) % 23) as f32 / 8.0 - 1.4
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

#![cfg(feature = "rocm")]

use half::f16;
use mote_backend_hip::{HipContext, HipError, quantized_linear_q4_0_f16};
use mote_core::Tensor;
use mote_reference::quantized_linear_f32;
use mote_types::{DType, Encoding, Layout, QuantFormat, Shape, TensorDesc};

const ROWS: usize = 3;
const OUTPUT_FEATURES: usize = 5;
const INPUT_FEATURES: usize = 64;

#[test]
fn q4_0_weight_only_linear_matches_the_cpu_oracle() {
    let context = HipContext::new(0).unwrap();
    let input: Vec<f16> = (0..ROWS * INPUT_FEATURES)
        .map(|index| f16::from_f32((index % 23) as f32 / 7.0 - 1.5))
        .collect();
    let weights = q4_0_weights(OUTPUT_FEATURES, INPUT_FEATURES);

    let input_tensor = context
        .from_bytes(
            plain_desc(ROWS, INPUT_FEATURES, DType::F16),
            &encode_f16(&input),
        )
        .unwrap();
    let weight_tensor = context
        .from_bytes(
            quantized_desc(OUTPUT_FEATURES, INPUT_FEATURES, QuantFormat::Q4_0),
            &weights,
        )
        .unwrap();
    let output = context
        .empty(plain_desc(ROWS, OUTPUT_FEATURES, DType::F16))
        .unwrap();

    // The bytes stay in native HIP storage and can be read back unchanged.
    assert_eq!(context.read_bytes(&weight_tensor).unwrap(), weights);
    quantized_linear_q4_0_f16(&context, &input_tensor, &weight_tensor, &output).unwrap();

    let actual = decode_f16(&context.read_bytes(&output).unwrap());
    let input_f32: Vec<f32> = input.iter().map(|value| value.to_f32()).collect();
    let expected = quantized_linear_f32(
        &input_f32,
        &weights,
        QuantFormat::Q4_0,
        ROWS,
        OUTPUT_FEATURES,
        INPUT_FEATURES,
    )
    .unwrap();

    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        let expected = f16::from_f32(expected).to_f32();
        let actual = actual.to_f32();
        let tolerance = 0.01f32.max(expected.abs() * 0.002);
        assert!(
            (actual - expected).abs() <= tolerance,
            "mismatch at output {index}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn validates_q4_0_linear_geometry_encoding_and_aliasing() {
    let context = HipContext::new(0).unwrap();
    let input = context
        .from_bytes(plain_desc(1, 32, DType::F16), &encode_f16(&[f16::ZERO; 32]))
        .unwrap();
    let weights = context
        .from_bytes(
            quantized_desc(32, 32, QuantFormat::Q4_0),
            &q4_0_weights(32, 32),
        )
        .unwrap();
    let output = context.empty(plain_desc(1, 32, DType::F16)).unwrap();

    assert!(matches!(
        quantized_linear_q4_0_f16(&context, &input, &weights, &input),
        Err(HipError::AliasedStorage {
            tensor: "output",
            other: "input"
        })
    ));

    let output_on_weights =
        Tensor::new(plain_desc(1, 32, DType::F16), weights.storage().clone(), 0).unwrap();
    assert!(matches!(
        quantized_linear_q4_0_f16(&context, &input, &weights, &output_on_weights),
        Err(HipError::AliasedStorage {
            tensor: "output",
            other: "weights"
        })
    ));

    let q8_weights = context
        .from_bytes(
            quantized_desc(32, 32, QuantFormat::Q8_0),
            &vec![0; 32 * QuantFormat::Q8_0.block_bytes()],
        )
        .unwrap();
    assert!(matches!(
        quantized_linear_q4_0_f16(&context, &input, &q8_weights, &output),
        Err(HipError::UnsupportedEncoding {
            actual: Encoding::Quantized(QuantFormat::Q8_0)
        })
    ));

    let short_weights = context
        .from_bytes(
            quantized_desc(32, 64, QuantFormat::Q4_0),
            &q4_0_weights(32, 64),
        )
        .unwrap();
    assert!(matches!(
        quantized_linear_q4_0_f16(&context, &input, &short_weights, &output),
        Err(HipError::ShapeMismatch {
            tensor: "weights",
            ..
        })
    ));
}

#[test]
fn fully_validated_empty_outputs_return_without_a_launch() {
    let context = HipContext::new(0).unwrap();
    let input = context
        .from_bytes(plain_desc(0, 32, DType::F16), &[])
        .unwrap();
    let weights = context
        .from_bytes(
            quantized_desc(2, 32, QuantFormat::Q4_0),
            &q4_0_weights(2, 32),
        )
        .unwrap();
    let output = context.empty(plain_desc(0, 2, DType::F16)).unwrap();

    quantized_linear_q4_0_f16(&context, &input, &weights, &output).unwrap();
    assert!(context.read_bytes(&output).unwrap().is_empty());
}

#[test]
fn zero_input_features_launches_and_writes_a_nonempty_zero_output() {
    let context = HipContext::new(0).unwrap();
    let input = context
        .from_bytes(plain_desc(3, 0, DType::F16), &[])
        .unwrap();
    let weights = context
        .from_bytes(quantized_desc(2, 0, QuantFormat::Q4_0), &[])
        .unwrap();
    let output = context.empty(plain_desc(3, 2, DType::F16)).unwrap();

    quantized_linear_q4_0_f16(&context, &input, &weights, &output).unwrap();
    assert_eq!(
        decode_f16(&context.read_bytes(&output).unwrap()),
        [f16::ZERO; 6]
    );
}

fn q4_0_weights(output_features: usize, input_features: usize) -> Vec<u8> {
    assert!(input_features.is_multiple_of(32));
    let mut bytes = Vec::with_capacity(
        output_features * (input_features / 32) * QuantFormat::Q4_0.block_bytes(),
    );
    for output_feature in 0..output_features {
        for block in 0..input_features / 32 {
            let scale = f16::from_f32(0.125 * ((output_feature + block) % 4 + 1) as f32);
            bytes.extend_from_slice(&scale.to_bits().to_le_bytes());
            for j in 0..16 {
                let low = (output_feature * 3 + block * 5 + j) % 16;
                let high = (output_feature * 7 + block * 2 + j * 3 + 1) % 16;
                bytes.push((low | (high << 4)) as u8);
            }
        }
    }
    bytes
}

fn plain_desc(rows: usize, cols: usize, dtype: DType) -> TensorDesc {
    TensorDesc::new(
        Shape::new(&[rows, cols]),
        Encoding::Plain(dtype),
        Layout::Contiguous,
    )
    .unwrap()
}

fn quantized_desc(rows: usize, cols: usize, quant_format: QuantFormat) -> TensorDesc {
    TensorDesc::new(
        Shape::new(&[rows, cols]),
        Encoding::Quantized(quant_format),
        Layout::Contiguous,
    )
    .unwrap()
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
        .map(u16::from_ne_bytes)
        .map(f16::from_bits)
        .collect()
}

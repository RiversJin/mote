use mote_reference::{QuantizedLinearError, dequantize_quantized_row, quantized_linear_f32};
use mote_types::QuantFormat;

fn q4_0_block(scale_bits: u16, quants: [u8; 16]) -> Vec<u8> {
    let mut block = Vec::from(scale_bits.to_le_bytes());
    block.extend_from_slice(&quants);
    block
}

fn q8_0_block(scale_bits: u16, quants: [i8; 32]) -> Vec<u8> {
    let mut block = Vec::from(scale_bits.to_le_bytes());
    block.extend(quants.map(|value| value as u8));
    block
}

#[test]
fn decodes_q4_0_low_and_high_nibbles_into_separate_half_blocks() {
    // f16 scale 2.0. Low nibble 0 => -16; high nibble 15 => 14.
    let mut quants = [0x88; 16];
    quants[0] = 0xf0;
    quants[1] = 0x78;
    let decoded =
        dequantize_quantized_row(&q4_0_block(0x4000, quants), QuantFormat::Q4_0, 32).unwrap();

    let mut expected = vec![0.0; 32];
    expected[0] = -16.0;
    expected[16] = 14.0;
    expected[17] = -2.0;
    assert_eq!(decoded, expected);
}

#[test]
fn q4_0_keeps_consecutive_blocks_separate() {
    let first = q4_0_block(0x3c00, [0x99; 16]); // scale 1, all values 1
    let second = q4_0_block(0x3800, [0x66; 16]); // scale 0.5, all values -1
    let bytes = [first, second].concat();

    let decoded = dequantize_quantized_row(&bytes, QuantFormat::Q4_0, 64).unwrap();
    assert_eq!(&decoded[..32], &[1.0; 32]);
    assert_eq!(&decoded[32..], &[-1.0; 32]);
}

#[test]
fn decodes_q8_0_signed_values_and_f16_scale() {
    let mut quants = [0i8; 32];
    quants[..5].copy_from_slice(&[-128, -1, 0, 1, 127]);
    let decoded =
        dequantize_quantized_row(&q8_0_block(0x3800, quants), QuantFormat::Q8_0, 32).unwrap();

    assert_eq!(&decoded[..5], &[-64.0, -0.5, 0.0, 0.5, 63.5]);
    assert!(decoded[5..].iter().all(|&value| value == 0.0));
}

#[test]
fn applies_q4_0_weight_rows_as_input_times_weight_transpose() {
    let first_weight = q4_0_block(0x3c00, [0x89; 16]);
    let second_weight = q4_0_block(0x3800, [0xa8; 16]);
    let weights = [first_weight, second_weight].concat();

    let mut input = vec![1.0; 32];
    input.extend([2.0; 16]);
    input.extend([-1.0; 16]);

    let output = quantized_linear_f32(&input, &weights, QuantFormat::Q4_0, 2, 2, 32).unwrap();
    assert_eq!(output, vec![16.0, 16.0, 32.0, -16.0]);
}

#[test]
fn applies_q8_0_rows_without_crossing_weight_row_boundaries() {
    let mut first = [0i8; 32];
    first[0] = 2;
    first[31] = -4;
    let mut second = [0i8; 32];
    second[0] = -3;
    second[31] = 5;
    let weights = [q8_0_block(0x3800, first), q8_0_block(0x3c00, second)].concat();

    let mut input = vec![0.0; 32];
    input[0] = 3.0;
    input[31] = 2.0;
    let output = quantized_linear_f32(&input, &weights, QuantFormat::Q8_0, 1, 2, 32).unwrap();
    assert_eq!(output, vec![-1.0, 1.0]);
}

#[test]
fn accepts_empty_but_block_aligned_linear_geometries() {
    assert_eq!(
        quantized_linear_f32(&[], &[], QuantFormat::Q4_0, 3, 2, 0).unwrap(),
        vec![0.0; 6]
    );
    let empty_batch_weights = [q8_0_block(0x3c00, [0; 32]), q8_0_block(0x3c00, [0; 32])].concat();
    assert!(
        quantized_linear_f32(&[], &empty_batch_weights, QuantFormat::Q8_0, 0, 2, 32,)
            .unwrap()
            .is_empty()
    );
    assert!(
        dequantize_quantized_row(&[], QuantFormat::Q8_0, 0)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn rejects_unsupported_formats_before_interpreting_their_bytes() {
    for format in [QuantFormat::Q4_K, QuantFormat::Q6_K] {
        assert_eq!(
            dequantize_quantized_row(&[], format, 0),
            Err(QuantizedLinearError::UnsupportedFormat { format })
        );
        assert_eq!(
            quantized_linear_f32(&[], &[], format, 0, 0, 0),
            Err(QuantizedLinearError::UnsupportedFormat { format })
        );
    }
}

#[test]
fn rejects_misaligned_rows_and_inexact_slice_lengths() {
    assert_eq!(
        dequantize_quantized_row(&[], QuantFormat::Q4_0, 31),
        Err(QuantizedLinearError::RowMisaligned {
            format: QuantFormat::Q4_0,
            row_elements: 31,
            block_elements: 32,
        })
    );
    assert_eq!(
        dequantize_quantized_row(&[0; 17], QuantFormat::Q4_0, 32),
        Err(QuantizedLinearError::RowByteLengthMismatch {
            expected: 18,
            actual: 17,
        })
    );
    assert_eq!(
        quantized_linear_f32(&[0.0; 31], &[0; 18], QuantFormat::Q4_0, 1, 1, 32),
        Err(QuantizedLinearError::InputLengthMismatch {
            expected: 32,
            actual: 31,
        })
    );
    assert_eq!(
        quantized_linear_f32(&[0.0; 32], &[0; 17], QuantFormat::Q4_0, 1, 1, 32),
        Err(QuantizedLinearError::WeightLengthMismatch {
            expected: 18,
            actual: 17,
        })
    );
}

#[test]
fn rejects_dimension_products_and_encoded_row_sizes_that_overflow() {
    assert_eq!(
        quantized_linear_f32(&[], &[], QuantFormat::Q4_0, usize::MAX, 1, 32),
        Err(QuantizedLinearError::DimensionsOverflow {
            rows: usize::MAX,
            output_features: 1,
            input_features: 32,
        })
    );

    let huge_q8_row = (usize::MAX / 32) * 32;
    assert_eq!(
        dequantize_quantized_row(&[], QuantFormat::Q8_0, huge_q8_row),
        Err(QuantizedLinearError::DimensionsOverflow {
            rows: 1,
            output_features: 1,
            input_features: huge_q8_row,
        })
    );
}

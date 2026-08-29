use std::{error::Error, time::Instant};

use half::f16;
use mote_backend_hip::{HipContext, quantized_linear_q4_0_f16};
use mote_reference::dequantize_quantized_row;
use mote_types::{DType, Encoding, Layout, QuantFormat, Shape, TensorDesc};

const CASES: &[(usize, usize, usize)] = &[
    (1, 4096, 4096),
    (1, 11008, 4096),
    (1, 4096, 11008),
    (16, 4096, 4096),
];
const WARMUPS: usize = 5;
const SAMPLES: usize = 9;
const BATCH: usize = 20;
const MIB: usize = 1024 * 1024;

fn main() {
    run().unwrap();
}

fn run() -> Result<(), Box<dyn Error>> {
    let context = HipContext::new(0)?;

    println!("Mote native HIP Q4_0 linear: F16 input/output, F32 accumulation");
    println!("samples per case: {SAMPLES}, {BATCH} launches per sample\n");
    println!(
        "{:>5}{:>8}{:>8}{:>14}{:>11}{:>11}{:>15}{:>11}{:>13}",
        "rows", "N", "K", "median us", "spread %", "TOP/s", "weight GB/s", "Q4 MiB", "storage MiB"
    );

    for &(rows, output_features, input_features) in CASES {
        let input_values = input_values(rows, input_features);
        let weight_bytes = q4_0_weights(output_features, input_features);
        let input = context.from_bytes(
            plain_desc(rows, input_features, DType::F16),
            &encode_f16(&input_values),
        )?;
        let weights = context.from_bytes(
            quantized_desc(output_features, input_features),
            &weight_bytes,
        )?;
        let output = context.empty(plain_desc(rows, output_features, DType::F16))?;

        for _ in 0..WARMUPS {
            quantized_linear_q4_0_f16(&context, &input, &weights, &output)?;
        }
        context.sync()?;

        let mut samples = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let start = Instant::now();
            for _ in 0..BATCH {
                quantized_linear_q4_0_f16(&context, &input, &weights, &output)?;
            }
            context.sync()?;
            samples.push(start.elapsed().as_secs_f64() * 1.0e6 / BATCH as f64);
        }
        samples.sort_by(f64::total_cmp);
        let microseconds = samples[SAMPLES / 2];
        let spread = (samples[SAMPLES - 1] - samples[0]) / microseconds * 100.0;
        let seconds = microseconds / 1.0e6;
        let operations = 2.0 * rows as f64 * output_features as f64 * input_features as f64;
        let tops = operations / seconds / 1.0e12;
        let encoded_weight_bandwidth = rows as f64 * weight_bytes.len() as f64 / seconds / 1.0e9;
        let storage_bytes =
            input_values.len() * 2 + weight_bytes.len() + rows * output_features * 2;

        let actual = decode_f16(&context.read_bytes(&output)?);
        validate(
            rows,
            output_features,
            input_features,
            &input_values,
            &weight_bytes,
            &actual,
        )?;

        println!(
            "{rows:>5}{output_features:>8}{input_features:>8}{microseconds:>14.3}{spread:>11.1}{tops:>11.3}{encoded_weight_bandwidth:>15.2}{:>11.2}{:>13.2}",
            bytes_to_mib(weight_bytes.len()),
            bytes_to_mib(storage_bytes),
        );
    }
    Ok(())
}

fn validate(
    rows: usize,
    output_features: usize,
    input_features: usize,
    input: &[f16],
    weights: &[u8],
    output: &[f16],
) -> Result<(), Box<dyn Error>> {
    let row_bytes =
        (input_features / QuantFormat::Q4_0.block_elements()) * QuantFormat::Q4_0.block_bytes();
    let positions = [
        (0, 0),
        (rows / 2, output_features / 3),
        (rows - 1, output_features - 1),
    ];
    for (row, output_feature) in positions {
        let byte_start = output_feature * row_bytes;
        let decoded = dequantize_quantized_row(
            &weights[byte_start..byte_start + row_bytes],
            QuantFormat::Q4_0,
            input_features,
        )?;
        let input_start = row * input_features;
        let expected = (0..input_features)
            .map(|inner| input[input_start + inner].to_f32() * decoded[inner])
            .sum::<f32>();
        let expected = f16::from_f32(expected).to_f32();
        let actual = output[row * output_features + output_feature].to_f32();
        let tolerance = 0.25f32.max(expected.abs() * 0.005);
        if (expected - actual).abs() > tolerance {
            return Err(format!(
                "validation failed for ({rows}, {output_features}, {input_features}) at ({row}, {output_feature}): expected {expected}, got {actual}"
            )
            .into());
        }
    }
    Ok(())
}

fn input_values(rows: usize, input_features: usize) -> Vec<f16> {
    (0..rows * input_features)
        .map(|index| f16::from_f32((index % 29) as f32 / 14.0 - 1.0))
        .collect()
}

fn q4_0_weights(output_features: usize, input_features: usize) -> Vec<u8> {
    assert!(input_features.is_multiple_of(QuantFormat::Q4_0.block_elements()));
    let blocks_per_row = input_features / QuantFormat::Q4_0.block_elements();
    let mut bytes =
        Vec::with_capacity(output_features * blocks_per_row * QuantFormat::Q4_0.block_bytes());
    for output_feature in 0..output_features {
        for block in 0..blocks_per_row {
            let scale =
                f16::from_f32(0.015625 * ((output_feature.wrapping_mul(3) + block) % 4 + 1) as f32);
            bytes.extend_from_slice(&scale.to_bits().to_le_bytes());
            for j in 0..16 {
                let low = (output_feature.wrapping_mul(5) + block * 3 + j) % 16;
                let high = (output_feature.wrapping_mul(7) + block + j * 3 + 1) % 16;
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

fn quantized_desc(rows: usize, cols: usize) -> TensorDesc {
    TensorDesc::new(
        Shape::new(&[rows, cols]),
        Encoding::Quantized(QuantFormat::Q4_0),
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

fn bytes_to_mib(bytes: usize) -> f64 {
    bytes as f64 / MIB as f64
}

use std::{error::Error, time::Instant};

use half::f16;
use mote_backend_hip::{HipContext, matmul_f16_f32};
use mote_types::{DType, Encoding, Layout, Shape, TensorDesc};

const SIZES: [usize; 4] = [256, 512, 1024, 2048];
const WARMUPS: usize = 5;
const SAMPLES: usize = 7;
const BATCH: usize = 20;
const MIB: usize = 1024 * 1024;

fn main() {
    run().unwrap();
}

fn run() -> Result<(), Box<dyn Error>> {
    let context = HipContext::new(0)?;
    let baseline_free = context.memory_info()?.free_bytes;

    println!("Mote native HIP/hipBLASLt: F16 inputs, F32 accumulation/output");
    println!("workspace: {} MiB", context.workspace_size_bytes() / MIB);
    println!("samples per case: {SAMPLES}, {BATCH} launches per sample\n");
    println!(
        "{:>8}{:>15}{:>11}{:>15}{:>15}",
        "M=N=K", "hipBLASLt us", "TFLOP/s", "tensor MiB", "HIP heap MiB"
    );

    for size in SIZES {
        let elements = size * size;
        let lhs_values = matrix_values(size, lhs_value);
        let rhs_values = matrix_values(size, rhs_value);
        let lhs_bytes = encode_f16(&lhs_values);
        let rhs_bytes = encode_f16(&rhs_values);
        let lhs = context.from_bytes(matrix_desc(size, DType::F16), &lhs_bytes)?;
        let rhs = context.from_bytes(matrix_desc(size, DType::F16), &rhs_bytes)?;
        let output = context.empty(matrix_desc(size, DType::F32))?;

        for _ in 0..WARMUPS {
            matmul_f16_f32(&context, &lhs, &rhs, &output)?;
        }
        context.sync()?;

        let mut samples = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let start = Instant::now();
            for _ in 0..BATCH {
                matmul_f16_f32(&context, &lhs, &rhs, &output)?;
            }
            context.sync()?;
            samples.push(start.elapsed().as_secs_f64() * 1.0e6 / BATCH as f64);
        }
        samples.sort_by(f64::total_cmp);
        let microseconds = samples[SAMPLES / 2];
        let operations = 2.0 * (size as f64).powi(3);
        let tflops = operations / (microseconds / 1.0e6) / 1.0e12;
        let heap_mib =
            bytes_to_mib(baseline_free.saturating_sub(context.memory_info()?.free_bytes));

        let actual = decode_f32(&context.read_bytes(&output)?);
        validate(size, &lhs_values, &rhs_values, &actual)?;

        println!(
            "{size:>8}{microseconds:>15.3}{tflops:>11.2}{:>15.2}{heap_mib:>15.2}",
            bytes_to_mib(elements * 8)
        );
    }
    Ok(())
}

fn matrix_values(size: usize, value: fn(usize, usize) -> f32) -> Vec<f16> {
    (0..size)
        .flat_map(|row| (0..size).map(move |col| f16::from_f32(value(row, col))))
        .collect()
}

fn lhs_value(row: usize, col: usize) -> f32 {
    ((row * 17 + col * 13) % 23) as f32 / 16.0 - 11.0 / 16.0
}

fn rhs_value(row: usize, col: usize) -> f32 {
    ((row * 7 + col * 19) % 29) as f32 / 16.0 - 14.0 / 16.0
}

fn validate(size: usize, lhs: &[f16], rhs: &[f16], output: &[f32]) -> Result<(), Box<dyn Error>> {
    let positions = [
        (0, 0),
        (size / 7, size / 5),
        (size / 2, size / 3),
        (size - 2, size - 3),
        (size - 1, size - 1),
    ];
    for (row, col) in positions {
        let expected = (0..size)
            .map(|inner| lhs[row * size + inner].to_f32() * rhs[inner * size + col].to_f32())
            .sum::<f32>();
        let actual = output[row * size + col];
        let tolerance = 0.05_f32.max(expected.abs() * 0.002);
        if (expected - actual).abs() > tolerance {
            return Err(format!(
                "validation failed for {size} at ({row}, {col}): expected {expected}, got {actual}"
            )
            .into());
        }
    }
    Ok(())
}

fn matrix_desc(size: usize, dtype: DType) -> TensorDesc {
    TensorDesc::new(
        Shape::new(&[size, size]),
        Encoding::Plain(dtype),
        Layout::Contiguous,
    )
    .unwrap()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    let (words, remainder) = bytes.as_chunks::<4>();
    assert!(remainder.is_empty());
    words.iter().copied().map(f32::from_ne_bytes).collect()
}

fn encode_f16(values: &[f16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_ne_bytes())
        .collect()
}

fn bytes_to_mib(bytes: usize) -> f64 {
    bytes as f64 / MIB as f64
}

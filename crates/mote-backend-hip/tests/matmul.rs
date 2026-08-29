#![cfg(feature = "rocm")]

use half::f16;
use mote_backend_hip::{HipContext, matmul_f16_f32};
use mote_reference::matmul_f32;
use mote_types::{DType, Encoding, Layout, Shape, TensorDesc};

const M: usize = 48;
const N: usize = 64;
const K: usize = 32;

#[test]
fn hipblaslt_multiplies_row_major_matrices_in_native_hip_storage() {
    let context = HipContext::new(0).unwrap();
    let memory = context.memory_info().unwrap();
    assert!(memory.total_bytes > 0);
    assert!(memory.free_bytes <= memory.total_bytes);
    let lhs = values(M, K, |row, col| {
        ((row * 7 + col * 3) % 17) as f32 / 8.0 - 1.0
    });
    let rhs = values(K, N, |row, col| {
        ((row * 5 + col * 11) % 19) as f32 / 8.0 - 1.0
    });
    let lhs_bytes = encode_f16(&lhs);
    let rhs_bytes = encode_f16(&rhs);
    let lhs_tensor = context
        .from_bytes(desc(M, K, DType::F16), &lhs_bytes)
        .unwrap();
    let rhs_tensor = context
        .from_bytes(desc(K, N, DType::F16), &rhs_bytes)
        .unwrap();
    let output = context.empty(desc(M, N, DType::F32)).unwrap();

    matmul_f16_f32(&context, &lhs_tensor, &rhs_tensor, &output).unwrap();
    let actual = decode_f32(&context.read_bytes(&output).unwrap());
    let lhs_f32: Vec<f32> = lhs.iter().map(|value| value.to_f32()).collect();
    let rhs_f32: Vec<f32> = rhs.iter().map(|value| value.to_f32()).collect();
    let expected = matmul_f32(&lhs_f32, &rhs_f32, M, N, K).unwrap();

    for row in 0..M {
        for col in 0..N {
            let expected = expected[row * N + col];
            let value = actual[row * N + col];
            let tolerance = 0.05_f32.max(expected.abs() * 0.002);
            assert!(
                (expected - value).abs() <= tolerance,
                "mismatch at ({row}, {col}): expected {expected}, got {value}"
            );
        }
    }
}

fn values(rows: usize, cols: usize, value: impl Fn(usize, usize) -> f32 + Copy) -> Vec<f16> {
    (0..rows)
        .flat_map(|row| (0..cols).map(move |col| f16::from_f32(value(row, col))))
        .collect()
}

fn encode_f16(values: &[f16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_ne_bytes())
        .collect()
}

fn desc(rows: usize, cols: usize, dtype: DType) -> TensorDesc {
    TensorDesc::new(
        Shape::new(&[rows, cols]),
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

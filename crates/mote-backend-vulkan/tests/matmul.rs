#![cfg(feature = "vulkan-tests")]

use mote_backend_vulkan::{
    VulkanContext,
    matmul::{matmul, matmul_cmma_f16_f32, profile_matmul, profile_matmul_cmma_f16_f32},
};
use mote_types::{DType, Encoding, Layout, Shape, TensorDesc};

#[test]
fn multiplies_non_tile_aligned_f32_matrices() {
    let context = VulkanContext::new(0).unwrap();
    let lhs_values = (0..51)
        .map(|index| (index % 7) as f32 - 3.0)
        .collect::<Vec<_>>();
    let rhs_values = (0..68)
        .map(|index| (index % 5) as f32 * 0.25 - 0.5)
        .collect::<Vec<_>>();
    let expected = reference_matmul(&lhs_values, &rhs_values, 3, 4, 17);
    let lhs = context
        .from_bytes(plain_f32_desc(&[3, 17]), &f32_bytes(&lhs_values))
        .unwrap();
    let rhs = context
        .from_bytes(plain_f32_desc(&[17, 4]), &f32_bytes(&rhs_values))
        .unwrap();
    let output = context.empty(plain_f32_desc(&[3, 4])).unwrap();

    matmul(&context, &lhs, &rhs, &output).unwrap();
    let device_time = profile_matmul(&context, &lhs, &rhs, &output).unwrap();
    assert!(!device_time.is_zero());

    let actual = bytes_as_f32(&context.read_bytes(&output).unwrap());
    assert_eq!(actual, expected);
}

#[test]
fn multiplies_f16_matrices_with_f32_cooperative_accumulation() {
    const SIZE: usize = 64;

    let context = VulkanContext::new(0).unwrap();
    assert!(context.supports_cooperative_matrix());
    let lhs_values = (0..SIZE * SIZE)
        .map(|index| half::f16::from_f32((index % 17) as f32 / 16.0 - 0.5))
        .collect::<Vec<_>>();
    let rhs_values = (0..SIZE * SIZE)
        .map(|index| half::f16::from_f32((index % 13) as f32 / 16.0 - 0.375))
        .collect::<Vec<_>>();
    let lhs_f32 = lhs_values
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let rhs_f32 = rhs_values
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let expected = reference_matmul(&lhs_f32, &rhs_f32, SIZE, SIZE, SIZE);
    let lhs = context
        .from_bytes(
            plain_desc(&[SIZE, SIZE], DType::F16),
            &f16_bytes(&lhs_values),
        )
        .unwrap();
    let rhs = context
        .from_bytes(
            plain_desc(&[SIZE, SIZE], DType::F16),
            &f16_bytes(&rhs_values),
        )
        .unwrap();
    let output = context
        .empty(plain_desc(&[SIZE, SIZE], DType::F32))
        .unwrap();

    matmul_cmma_f16_f32(&context, &lhs, &rhs, &output).unwrap();
    let device_time = profile_matmul_cmma_f16_f32(&context, &lhs, &rhs, &output).unwrap();
    assert!(!device_time.is_zero());

    let actual = bytes_as_f32(&context.read_bytes(&output).unwrap());
    assert_close(&actual, &expected, 1.0e-3);
}

fn reference_matmul(lhs: &[f32], rhs: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut output = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            output[row * n + col] = (0..k)
                .map(|inner| lhs[row * k + inner] * rhs[inner * n + col])
                .sum();
        }
    }
    output
}

fn plain_f32_desc(shape: &[usize]) -> TensorDesc {
    plain_desc(shape, DType::F32)
}

fn plain_desc(shape: &[usize], dtype: DType) -> TensorDesc {
    TensorDesc::new(
        Shape::new(shape),
        Encoding::Plain(dtype),
        Layout::Contiguous,
    )
    .unwrap()
}

fn f16_bytes(values: &[half::f16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn bytes_as_f32(bytes: &[u8]) -> Vec<f32> {
    let (values, remainder) = bytes.as_chunks::<{ size_of::<f32>() }>();
    assert!(remainder.is_empty());
    values
        .iter()
        .map(|bytes| f32::from_ne_bytes(*bytes))
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: expected {expected}, got {actual}"
        );
    }
}

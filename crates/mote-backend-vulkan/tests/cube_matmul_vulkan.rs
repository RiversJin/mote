#![cfg(all(feature = "vulkan-tests", feature = "comparison"))]

use cubecl::{
    CubeElement, Runtime,
    wgpu::{WgpuDevice, WgpuRuntime},
};
use mote_cube::{
    CubeContext,
    matmul::{matmul, matmul_cmma_f16_f32},
};
use mote_types::{BackendKind, DType, Device, Encoding, Layout, Shape, TensorDesc};

#[test]
fn runs_f32_and_cooperative_matrix_matmul_on_vulkan() {
    const SIZE: usize = 128;

    let client = WgpuRuntime::client(&WgpuDevice::DiscreteGpu(0));
    let context = CubeContext::new(Device::new(BackendKind::Vulkan, 0), client.clone()).unwrap();
    eprintln!(
        "CubeCL Vulkan reports {} cooperative-matrix configurations: {:?}",
        context.cooperative_matrix_config_count(),
        client.properties().features.matmul.cmma
    );
    assert!(context.supports_cooperative_matrix());

    let lhs_values = (0..SIZE * SIZE)
        .map(|index| (index % 17) as f32 / 16.0 - 0.5)
        .collect::<Vec<_>>();
    let rhs_values = (0..SIZE * SIZE)
        .map(|index| (index % 13) as f32 / 16.0 - 0.375)
        .collect::<Vec<_>>();
    let expected = reference_matmul(&lhs_values, &rhs_values, SIZE);
    let lhs = context
        .from_bytes(plain_f32_desc(SIZE), f32::as_bytes(&lhs_values))
        .unwrap();
    let rhs = context
        .from_bytes(plain_f32_desc(SIZE), f32::as_bytes(&rhs_values))
        .unwrap();
    let lhs_f16 = lhs_values
        .iter()
        .copied()
        .map(half::f16::from_f32)
        .collect::<Vec<_>>();
    let rhs_f16 = rhs_values
        .iter()
        .copied()
        .map(half::f16::from_f32)
        .collect::<Vec<_>>();
    let lhs_cmma = context
        .from_bytes(plain_f16_desc(SIZE), half::f16::as_bytes(&lhs_f16))
        .unwrap();
    let rhs_cmma = context
        .from_bytes(plain_f16_desc(SIZE), half::f16::as_bytes(&rhs_f16))
        .unwrap();
    let exact = context.empty(plain_f32_desc(SIZE)).unwrap();
    let cmma = context.empty(plain_f32_desc(SIZE)).unwrap();

    matmul(&context, &lhs, &rhs, &exact).unwrap();
    matmul_cmma_f16_f32(&context, &lhs_cmma, &rhs_cmma, &cmma).unwrap();
    cubecl::future::block_on(client.sync()).unwrap();

    let exact_bytes = context.read_bytes(&exact).unwrap();
    let cmma_bytes = context.read_bytes(&cmma).unwrap();
    let exact = f32::from_bytes(&exact_bytes);
    let cmma = f32::from_bytes(&cmma_bytes);
    assert_close(exact, &expected, 1.0e-4);
    assert_close(cmma, &expected, 1.0e-3);
}

fn reference_matmul(lhs: &[f32], rhs: &[f32], size: usize) -> Vec<f32> {
    let mut output = vec![0.0; size * size];
    for row in 0..size {
        for col in 0..size {
            output[row * size + col] = (0..size)
                .map(|inner| lhs[row * size + inner] * rhs[inner * size + col])
                .sum();
        }
    }
    output
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: expected {expected}, got {actual}"
        );
    }
}

fn plain_f32_desc(size: usize) -> TensorDesc {
    TensorDesc::new(
        Shape::new(&[size, size]),
        Encoding::Plain(DType::F32),
        Layout::Contiguous,
    )
    .unwrap()
}

fn plain_f16_desc(size: usize) -> TensorDesc {
    TensorDesc::new(
        Shape::new(&[size, size]),
        Encoding::Plain(DType::F16),
        Layout::Contiguous,
    )
    .unwrap()
}

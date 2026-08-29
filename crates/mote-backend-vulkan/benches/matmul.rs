use std::time::{Duration, Instant};

use cubecl::{
    Runtime,
    prelude::ComputeClient,
    profile::TimingMethod,
    wgpu::{WgpuDevice, WgpuRuntime},
};
use mote_backend_vulkan::{
    VulkanContext,
    matmul::{
        matmul as vulkan_matmul, matmul_cmma_f16_f32 as vulkan_matmul_cmma, profile_matmul,
        profile_matmul_cmma_f16_f32,
    },
};
use mote_core::Tensor;
use mote_cube::{
    CubeContext,
    matmul::{matmul as cube_matmul, matmul_cmma_f16_f32},
};
use mote_types::{BackendKind, DType, Device, Encoding, Layout, Shape, TensorDesc};

const SAMPLES: usize = 7;
const CASES: &[usize] = &[256, 512, 1024, 2048];

#[derive(Debug, Clone, Copy)]
enum CubeMode {
    F32,
    CmmaF16F32,
}

struct CaseResult {
    size: usize,
    cube_f32: Duration,
    cube_cmma: Duration,
    vulkan_f32: Duration,
    vulkan_cmma: Duration,
}

fn main() {
    let cube_client = WgpuRuntime::client(&WgpuDevice::DiscreteGpu(0));
    let cube_context = CubeContext::new(Device::new(BackendKind::Vulkan, 0), cube_client.clone())
        .expect("failed to initialize CubeCL Vulkan context");
    let vulkan_context = VulkanContext::new(0).expect("failed to initialize direct Vulkan context");

    assert_eq!(
        cube_client.properties().timing_method,
        TimingMethod::Device,
        "CubeCL device timestamp queries are required"
    );
    println!("device: {}", vulkan_context.device_name());
    println!(
        "CubeCL cooperative-matrix configs ({}): {:?}",
        cube_context.cooperative_matrix_config_count(),
        cube_client.properties().features.matmul.cmma
    );
    assert!(
        cube_context.supports_cooperative_matrix(),
        "the selected CubeCL Vulkan device reports no cooperative-matrix configurations"
    );
    assert!(
        vulkan_context.supports_cooperative_matrix(),
        "the selected direct Vulkan device reports no cooperative-matrix support"
    );
    println!("samples per case: {SAMPLES} (median reported, order rotates)");
    println!("Cube CMMA uses F16 inputs and F32 accumulation/output");

    print_cold_launch(&cube_client, &cube_context, &vulkan_context);

    let mut results = Vec::with_capacity(CASES.len());
    for &size in CASES {
        let (lhs_values, rhs_values) = inputs(size);
        let bytes_lhs = f32_bytes(&lhs_values);
        let bytes_rhs = f32_bytes(&rhs_values);
        let bytes_lhs_f16 = f16_bytes(&lhs_values);
        let bytes_rhs_f16 = f16_bytes(&rhs_values);
        let desc = plain_f32_desc(size);

        let cube_lhs = cube_context
            .from_bytes(desc.clone(), &bytes_lhs)
            .expect("CubeCL lhs upload failed");
        let cube_rhs = cube_context
            .from_bytes(desc.clone(), &bytes_rhs)
            .expect("CubeCL rhs upload failed");
        let cube_lhs_f16 = cube_context
            .from_bytes(plain_f16_desc(size), &bytes_lhs_f16)
            .expect("CubeCL F16 lhs upload failed");
        let cube_rhs_f16 = cube_context
            .from_bytes(plain_f16_desc(size), &bytes_rhs_f16)
            .expect("CubeCL F16 rhs upload failed");
        let cube_f32_output = cube_context
            .empty(desc.clone())
            .expect("CubeCL F32 output allocation failed");
        let cube_cmma_output = cube_context
            .empty(desc.clone())
            .expect("CubeCL CMMA output allocation failed");

        let vulkan_lhs = vulkan_context
            .from_bytes(desc.clone(), &bytes_lhs)
            .expect("Vulkan lhs upload failed");
        let vulkan_rhs = vulkan_context
            .from_bytes(desc.clone(), &bytes_rhs)
            .expect("Vulkan rhs upload failed");
        let vulkan_lhs_f16 = vulkan_context
            .from_bytes(plain_f16_desc(size), &bytes_lhs_f16)
            .expect("Vulkan F16 lhs upload failed");
        let vulkan_rhs_f16 = vulkan_context
            .from_bytes(plain_f16_desc(size), &bytes_rhs_f16)
            .expect("Vulkan F16 rhs upload failed");
        let vulkan_output = vulkan_context
            .empty(desc.clone())
            .expect("Vulkan output allocation failed");
        let vulkan_cmma_output = vulkan_context
            .empty(desc)
            .expect("Vulkan CMMA output allocation failed");

        launch_cube(
            &cube_context,
            &cube_lhs,
            &cube_rhs,
            &cube_f32_output,
            CubeMode::F32,
        );
        sync_cube(&cube_client);
        launch_cube(
            &cube_context,
            &cube_lhs_f16,
            &cube_rhs_f16,
            &cube_cmma_output,
            CubeMode::CmmaF16F32,
        );
        sync_cube(&cube_client);
        vulkan_matmul(&vulkan_context, &vulkan_lhs, &vulkan_rhs, &vulkan_output)
            .expect("Vulkan warmup failed");
        vulkan_matmul_cmma(
            &vulkan_context,
            &vulkan_lhs_f16,
            &vulkan_rhs_f16,
            &vulkan_cmma_output,
        )
        .expect("Vulkan CMMA warmup failed");

        let (cube_f32, cube_cmma, vulkan_f32, vulkan_cmma) = measure_four(
            || {
                profile_cube(
                    &cube_client,
                    &cube_context,
                    &cube_lhs,
                    &cube_rhs,
                    &cube_f32_output,
                    CubeMode::F32,
                )
            },
            || {
                profile_cube(
                    &cube_client,
                    &cube_context,
                    &cube_lhs_f16,
                    &cube_rhs_f16,
                    &cube_cmma_output,
                    CubeMode::CmmaF16F32,
                )
            },
            || {
                profile_matmul(&vulkan_context, &vulkan_lhs, &vulkan_rhs, &vulkan_output)
                    .expect("Vulkan timestamp profiling failed")
            },
            || {
                profile_matmul_cmma_f16_f32(
                    &vulkan_context,
                    &vulkan_lhs_f16,
                    &vulkan_rhs_f16,
                    &vulkan_cmma_output,
                )
                .expect("Vulkan CMMA timestamp profiling failed")
            },
        );

        validate_samples(
            size,
            &lhs_values,
            &rhs_values,
            &bytes_as_f32(
                &cube_context
                    .read_bytes(&cube_f32_output)
                    .expect("CubeCL F32 readback failed"),
            ),
            false,
        );
        validate_samples(
            size,
            &lhs_values,
            &rhs_values,
            &bytes_as_f32(
                &vulkan_context
                    .read_bytes(&vulkan_cmma_output)
                    .expect("Vulkan CMMA readback failed"),
            ),
            true,
        );
        validate_samples(
            size,
            &lhs_values,
            &rhs_values,
            &bytes_as_f32(
                &cube_context
                    .read_bytes(&cube_cmma_output)
                    .expect("CubeCL CMMA readback failed"),
            ),
            true,
        );
        validate_samples(
            size,
            &lhs_values,
            &rhs_values,
            &bytes_as_f32(
                &vulkan_context
                    .read_bytes(&vulkan_output)
                    .expect("Vulkan readback failed"),
            ),
            false,
        );

        results.push(CaseResult {
            size,
            cube_f32,
            cube_cmma,
            vulkan_f32,
            vulkan_cmma,
        });
    }

    println!();
    println!("GPU timestamp matrix multiplication:");
    println!(
        "{:>8} {:>13} {:>9} {:>13} {:>9} {:>13} {:>9} {:>13} {:>9}",
        "M=N=K",
        "Cube F32 us",
        "TFLOP/s",
        "Cube CMMA us",
        "TFLOP/s",
        "Slang F32 us",
        "TFLOP/s",
        "Slang CMMA us",
        "TFLOP/s"
    );
    for result in results {
        println!(
            "{:>8} {:>13.3} {:>9.2} {:>13.3} {:>9.2} {:>13.3} {:>9.2} {:>13.3} {:>9.2}",
            result.size,
            micros(result.cube_f32),
            tflops(result.size, result.cube_f32),
            micros(result.cube_cmma),
            tflops(result.size, result.cube_cmma),
            micros(result.vulkan_f32),
            tflops(result.size, result.vulkan_f32),
            micros(result.vulkan_cmma),
            tflops(result.size, result.vulkan_cmma),
        );
    }
}

fn print_cold_launch(
    client: &ComputeClient<WgpuRuntime>,
    cube: &CubeContext<WgpuRuntime>,
    vulkan: &VulkanContext,
) {
    let size = 256;
    let (lhs_values, rhs_values) = inputs(size);
    let desc = plain_f32_desc(size);
    let cube_lhs = cube
        .from_bytes(desc.clone(), &f32_bytes(&lhs_values))
        .unwrap();
    let cube_rhs = cube
        .from_bytes(desc.clone(), &f32_bytes(&rhs_values))
        .unwrap();
    let cube_lhs_f16 = cube
        .from_bytes(plain_f16_desc(size), &f16_bytes(&lhs_values))
        .unwrap();
    let cube_rhs_f16 = cube
        .from_bytes(plain_f16_desc(size), &f16_bytes(&rhs_values))
        .unwrap();
    let cube_output = cube.empty(desc.clone()).unwrap();
    let vulkan_lhs = vulkan
        .from_bytes(desc.clone(), &f32_bytes(&lhs_values))
        .unwrap();
    let vulkan_rhs = vulkan
        .from_bytes(desc.clone(), &f32_bytes(&rhs_values))
        .unwrap();
    let vulkan_lhs_f16 = vulkan
        .from_bytes(plain_f16_desc(size), &f16_bytes(&lhs_values))
        .unwrap();
    let vulkan_rhs_f16 = vulkan
        .from_bytes(plain_f16_desc(size), &f16_bytes(&rhs_values))
        .unwrap();
    let vulkan_output = vulkan.empty(desc.clone()).unwrap();
    let vulkan_cmma_output = vulkan.empty(desc).unwrap();

    let started = Instant::now();
    launch_cube(cube, &cube_lhs, &cube_rhs, &cube_output, CubeMode::F32);
    sync_cube(client);
    let cube_f32 = started.elapsed();

    let started = Instant::now();
    launch_cube(
        cube,
        &cube_lhs_f16,
        &cube_rhs_f16,
        &cube_output,
        CubeMode::CmmaF16F32,
    );
    sync_cube(client);
    let cube_cmma = started.elapsed();

    let started = Instant::now();
    vulkan_matmul(vulkan, &vulkan_lhs, &vulkan_rhs, &vulkan_output)
        .expect("Vulkan cold launch failed");
    let vulkan_f32 = started.elapsed();

    let started = Instant::now();
    vulkan_matmul_cmma(
        vulkan,
        &vulkan_lhs_f16,
        &vulkan_rhs_f16,
        &vulkan_cmma_output,
    )
    .expect("Vulkan CMMA cold launch failed");
    let vulkan_cmma = started.elapsed();

    println!("synchronous cold launch ({size}x{size}):");
    println!("  CubeCL F32       : {:>10.3} ms", millis(cube_f32));
    println!("  CubeCL F16 CMMA  : {:>10.3} ms", millis(cube_cmma));
    println!("  Slang Vulkan F32 : {:>10.3} ms", millis(vulkan_f32));
    println!("  Slang Vulkan CMMA: {:>10.3} ms", millis(vulkan_cmma));
}

fn launch_cube(
    context: &CubeContext<WgpuRuntime>,
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
    mode: CubeMode,
) {
    match mode {
        CubeMode::F32 => cube_matmul(context, lhs, rhs, output).expect("CubeCL F32 launch failed"),
        CubeMode::CmmaF16F32 => {
            matmul_cmma_f16_f32(context, lhs, rhs, output).expect("CubeCL CMMA launch failed")
        }
    }
}

fn profile_cube(
    client: &ComputeClient<WgpuRuntime>,
    context: &CubeContext<WgpuRuntime>,
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
    mode: CubeMode,
) -> Duration {
    sync_cube(client);
    let label = match mode {
        CubeMode::F32 => "matmul_f32",
        CubeMode::CmmaF16F32 => "matmul_cmma_f16_f32",
    };
    let (_, profile) = client
        .profile(|| launch_cube(context, lhs, rhs, output, mode), label)
        .expect("CubeCL timestamp profiling failed");
    assert_eq!(profile.timing_method(), TimingMethod::Device);
    cubecl::future::block_on(profile.resolve()).duration()
}

fn measure_four(
    mut first: impl FnMut() -> Duration,
    mut second: impl FnMut() -> Duration,
    mut third: impl FnMut() -> Duration,
    mut fourth: impl FnMut() -> Duration,
) -> (Duration, Duration, Duration, Duration) {
    let mut first_samples = Vec::with_capacity(SAMPLES);
    let mut second_samples = Vec::with_capacity(SAMPLES);
    let mut third_samples = Vec::with_capacity(SAMPLES);
    let mut fourth_samples = Vec::with_capacity(SAMPLES);
    for sample in 0..SAMPLES {
        match sample % 4 {
            0 => {
                first_samples.push(first());
                second_samples.push(second());
                third_samples.push(third());
                fourth_samples.push(fourth());
            }
            1 => {
                second_samples.push(second());
                third_samples.push(third());
                fourth_samples.push(fourth());
                first_samples.push(first());
            }
            2 => {
                third_samples.push(third());
                fourth_samples.push(fourth());
                first_samples.push(first());
                second_samples.push(second());
            }
            _ => {
                fourth_samples.push(fourth());
                first_samples.push(first());
                second_samples.push(second());
                third_samples.push(third());
            }
        }
    }
    (
        median(first_samples),
        median(second_samples),
        median(third_samples),
        median(fourth_samples),
    )
}

fn validate_samples(size: usize, lhs: &[f32], rhs: &[f32], output: &[f32], mixed_precision: bool) {
    for (row, col) in [
        (0, 0),
        (size / 7, size / 5),
        (size / 2, size / 3),
        (size - 2, size - 3),
        (size - 1, size - 1),
    ] {
        let expected = (0..size)
            .map(|inner| lhs[row * size + inner] * rhs[inner * size + col])
            .sum::<f32>();
        let actual = output[row * size + col];
        let tolerance = if mixed_precision {
            0.02 + expected.abs() * 0.002
        } else {
            0.002 + expected.abs() * 0.0001
        };
        assert!(
            (actual - expected).abs() <= tolerance,
            "{size}x{size} element ({row}, {col}): expected {expected}, got {actual}"
        );
    }
}

fn inputs(size: usize) -> (Vec<f32>, Vec<f32>) {
    let lhs = (0..size * size)
        .map(|index| ((index.wrapping_mul(17).wrapping_add(3)) % 31) as f32 / 32.0 - 0.46875)
        .collect();
    let rhs = (0..size * size)
        .map(|index| ((index.wrapping_mul(13).wrapping_add(5)) % 29) as f32 / 32.0 - 0.4375)
        .collect();
    (lhs, rhs)
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

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .copied()
        .map(half::f16::from_f32)
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

fn sync_cube(client: &ComputeClient<WgpuRuntime>) {
    cubecl::future::block_on(client.sync()).expect("CubeCL synchronization failed");
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000.0
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn tflops(size: usize, duration: Duration) -> f64 {
    2.0 * (size as f64).powi(3) / duration.as_secs_f64() / 1.0e12
}

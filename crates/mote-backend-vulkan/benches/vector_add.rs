use std::time::{Duration, Instant};

use cubecl::{
    Runtime,
    wgpu::{WgpuDevice, WgpuRuntime},
};
use mote_backend_vulkan::{VulkanContext, vector_add::vector_add as vulkan_vector_add};
use mote_cube::{CubeContext, vector_add::vector_add as cube_vector_add};
use mote_types::{BackendKind, DType, Device, Encoding, Layout, Shape, TensorDesc};

const COLD_NUMEL: usize = 1 << 20;
const CASES: &[(usize, usize)] = &[(1 << 10, 2_000), (1 << 20, 200), (1 << 23, 40)];

fn main() {
    let cube_client = WgpuRuntime::client(&WgpuDevice::DiscreteGpu(0));
    let cube_context =
        CubeContext::<WgpuRuntime>::new(Device::new(BackendKind::Vulkan, 0), cube_client.clone())
            .expect("failed to initialize CubeCL Vulkan context");
    let vulkan_context = VulkanContext::new(0).expect("failed to initialize direct Vulkan context");

    println!("device: {}", vulkan_context.device_name());
    println!("timing: host wall-clock, one GPU synchronization per dispatch");

    let (lhs_bytes, rhs_bytes) = input_bytes(COLD_NUMEL);
    let desc = plain_f32_desc(COLD_NUMEL);
    let cube_lhs = cube_context
        .from_bytes(desc.clone(), &lhs_bytes)
        .expect("failed to upload CubeCL lhs");
    let cube_rhs = cube_context
        .from_bytes(desc.clone(), &rhs_bytes)
        .expect("failed to upload CubeCL rhs");
    let cube_output = cube_context
        .empty(desc.clone())
        .expect("failed to allocate CubeCL output");
    let vulkan_lhs = vulkan_context
        .from_bytes(desc.clone(), &lhs_bytes)
        .expect("failed to upload Vulkan lhs");
    let vulkan_rhs = vulkan_context
        .from_bytes(desc.clone(), &rhs_bytes)
        .expect("failed to upload Vulkan rhs");
    let vulkan_output = vulkan_context
        .empty(desc)
        .expect("failed to allocate Vulkan output");

    // CubeCL uploads may be queued lazily. Complete them before timing the
    // first kernel so both cold-launch measurements exclude host uploads.
    sync_cube(&cube_client);

    let started = Instant::now();
    cube_vector_add(&cube_context, &cube_lhs, &cube_rhs, &cube_output)
        .expect("CubeCL cold launch failed");
    sync_cube(&cube_client);
    let cube_cold = started.elapsed();

    let started = Instant::now();
    vulkan_vector_add(&vulkan_context, &vulkan_lhs, &vulkan_rhs, &vulkan_output)
        .expect("Vulkan cold launch failed");
    let vulkan_cold = started.elapsed();

    assert_output(
        &cube_context
            .read_bytes(&cube_output)
            .expect("CubeCL cold readback failed"),
    );
    assert_output(
        &vulkan_context
            .read_bytes(&vulkan_output)
            .expect("Vulkan cold readback failed"),
    );

    println!("cold launch ({COLD_NUMEL} elements):");
    println!("  CubeCL Vulkan : {:>10.3} ms", millis(cube_cold));
    println!("  direct Vulkan : {:>10.3} ms", millis(vulkan_cold));
    println!();
    println!(
        "{:>12} {:>8} {:>14} {:>12} {:>14} {:>12} {:>10}",
        "elements", "iters", "CubeCL us", "GB/s", "Vulkan us", "GB/s", "Vk/Cube"
    );

    for &(numel, iterations) in CASES {
        let (lhs_bytes, rhs_bytes) = input_bytes(numel);
        let desc = plain_f32_desc(numel);
        let cube_lhs = cube_context
            .from_bytes(desc.clone(), &lhs_bytes)
            .expect("failed to upload CubeCL lhs");
        let cube_rhs = cube_context
            .from_bytes(desc.clone(), &rhs_bytes)
            .expect("failed to upload CubeCL rhs");
        let cube_output = cube_context
            .empty(desc.clone())
            .expect("failed to allocate CubeCL output");
        let vulkan_lhs = vulkan_context
            .from_bytes(desc.clone(), &lhs_bytes)
            .expect("failed to upload Vulkan lhs");
        let vulkan_rhs = vulkan_context
            .from_bytes(desc.clone(), &rhs_bytes)
            .expect("failed to upload Vulkan rhs");
        let vulkan_output = vulkan_context
            .empty(desc)
            .expect("failed to allocate Vulkan output");

        cube_vector_add(&cube_context, &cube_lhs, &cube_rhs, &cube_output)
            .expect("CubeCL warmup failed");
        sync_cube(&cube_client);
        vulkan_vector_add(&vulkan_context, &vulkan_lhs, &vulkan_rhs, &vulkan_output)
            .expect("Vulkan warmup failed");

        let cube_elapsed = measure(iterations, || {
            cube_vector_add(&cube_context, &cube_lhs, &cube_rhs, &cube_output)
                .expect("CubeCL benchmark launch failed");
            sync_cube(&cube_client);
        });
        let vulkan_elapsed = measure(iterations, || {
            vulkan_vector_add(&vulkan_context, &vulkan_lhs, &vulkan_rhs, &vulkan_output)
                .expect("Vulkan benchmark launch failed");
        });

        assert_output(
            &cube_context
                .read_bytes(&cube_output)
                .expect("CubeCL readback failed"),
        );
        assert_output(
            &vulkan_context
                .read_bytes(&vulkan_output)
                .expect("Vulkan readback failed"),
        );

        let cube_average = cube_elapsed.div_f64(iterations as f64);
        let vulkan_average = vulkan_elapsed.div_f64(iterations as f64);
        println!(
            "{:>12} {:>8} {:>14.3} {:>12.2} {:>14.3} {:>12.2} {:>10.2}",
            numel,
            iterations,
            micros(cube_average),
            bandwidth_gbps(numel, cube_average),
            micros(vulkan_average),
            bandwidth_gbps(numel, vulkan_average),
            vulkan_average.as_secs_f64() / cube_average.as_secs_f64(),
        );
    }
}

fn measure(iterations: usize, mut operation: impl FnMut()) -> Duration {
    let started = Instant::now();
    for _ in 0..iterations {
        operation();
    }
    started.elapsed()
}

fn sync_cube(client: &cubecl::prelude::ComputeClient<WgpuRuntime>) {
    cubecl::future::block_on(client.sync()).expect("CubeCL synchronization failed");
}

fn plain_f32_desc(numel: usize) -> TensorDesc {
    TensorDesc::new(
        Shape::new(&[numel]),
        Encoding::Plain(DType::F32),
        Layout::Contiguous,
    )
    .expect("invalid benchmark tensor descriptor")
}

fn input_bytes(numel: usize) -> (Vec<u8>, Vec<u8>) {
    (
        repeated_f32_bytes(1.25, numel),
        repeated_f32_bytes(2.5, numel),
    )
}

fn repeated_f32_bytes(value: f32, numel: usize) -> Vec<u8> {
    let word = value.to_ne_bytes();
    let mut bytes = Vec::with_capacity(numel * word.len());
    for _ in 0..numel {
        bytes.extend_from_slice(&word);
    }
    bytes
}

fn assert_output(bytes: &[u8]) {
    let (words, remainder) = bytes.as_chunks::<{ size_of::<f32>() }>();
    assert!(remainder.is_empty());
    assert!(words.iter().all(|word| f32::from_ne_bytes(*word) == 3.75));
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000.0
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn bandwidth_gbps(numel: usize, duration: Duration) -> f64 {
    let transferred_bytes = numel as f64 * size_of::<f32>() as f64 * 3.0;
    transferred_bytes / duration.as_secs_f64() / 1_000_000_000.0
}

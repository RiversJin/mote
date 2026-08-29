use std::time::{Duration, Instant};

use cubecl::{
    Runtime,
    prelude::ComputeClient,
    profile::TimingMethod,
    wgpu::{WgpuDevice, WgpuRuntime},
};
use mote_backend_vulkan::{
    VulkanContext,
    vector_add::{profile_vector_add, vector_add as vulkan_vector_add, vector_add_batch},
};
use mote_core::Tensor;
use mote_cube::{CubeContext, vector_add::vector_add as cube_vector_add};
use mote_types::{BackendKind, DType, Device, Encoding, Layout, Shape, TensorDesc};

const COLD_NUMEL: usize = 1 << 20;
const SAMPLES: usize = 9;
const CASES: &[(usize, usize)] = &[(1 << 10, 2_000), (1 << 20, 200), (1 << 23, 40)];

struct CaseResult {
    numel: usize,
    dispatches: usize,
    sync_cube: Duration,
    sync_vulkan: Duration,
    batch_cube: Duration,
    batch_vulkan: Duration,
    device_cube: Duration,
    device_vulkan: Duration,
}

fn main() {
    let cube_client = WgpuRuntime::client(&WgpuDevice::DiscreteGpu(0));
    let cube_context =
        CubeContext::<WgpuRuntime>::new(Device::new(BackendKind::Vulkan, 0), cube_client.clone())
            .expect("failed to initialize CubeCL Vulkan context");
    let vulkan_context = VulkanContext::new(0).expect("failed to initialize direct Vulkan context");

    println!("device: {}", vulkan_context.device_name());
    println!(
        "CubeCL timing capability: {}",
        cube_client.properties().timing_method
    );
    println!("samples per case: {SAMPLES} (median reported, order alternates)");

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

    println!("synchronous cold launch ({COLD_NUMEL} elements):");
    println!("  CubeCL Vulkan : {:>10.3} ms", millis(cube_cold));
    println!("  direct Vulkan : {:>10.3} ms", millis(vulkan_cold));

    let mut results = Vec::with_capacity(CASES.len());
    for &(numel, dispatches) in CASES {
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

        for _ in 0..5 {
            cube_vector_add(&cube_context, &cube_lhs, &cube_rhs, &cube_output)
                .expect("CubeCL warmup failed");
            sync_cube(&cube_client);
            vulkan_vector_add(&vulkan_context, &vulkan_lhs, &vulkan_rhs, &vulkan_output)
                .expect("Vulkan warmup failed");
        }

        let (sync_cube_time, sync_vulkan_time) = measure_pairs(
            || {
                measure(dispatches, || {
                    cube_vector_add(&cube_context, &cube_lhs, &cube_rhs, &cube_output)
                        .expect("CubeCL synchronous launch failed");
                    sync_cube(&cube_client);
                })
                .div_f64(dispatches as f64)
            },
            || {
                measure(dispatches, || {
                    vulkan_vector_add(&vulkan_context, &vulkan_lhs, &vulkan_rhs, &vulkan_output)
                        .expect("Vulkan synchronous launch failed");
                })
                .div_f64(dispatches as f64)
            },
        );

        let (batch_cube_time, batch_vulkan_time) = measure_pairs(
            || {
                let elapsed = measure(1, || {
                    for _ in 0..dispatches {
                        cube_vector_add(&cube_context, &cube_lhs, &cube_rhs, &cube_output)
                            .expect("CubeCL batched launch failed");
                    }
                    sync_cube(&cube_client);
                });
                elapsed.div_f64(dispatches as f64)
            },
            || {
                measure(1, || {
                    vector_add_batch(
                        &vulkan_context,
                        &vulkan_lhs,
                        &vulkan_rhs,
                        &vulkan_output,
                        dispatches,
                    )
                    .expect("Vulkan batched launch failed");
                })
                .div_f64(dispatches as f64)
            },
        );

        let (device_cube_time, device_vulkan_time) = measure_pairs(
            || {
                profile_cube_vector_add(
                    &cube_client,
                    &cube_context,
                    &cube_lhs,
                    &cube_rhs,
                    &cube_output,
                    1,
                )
            },
            || {
                profile_vector_add(&vulkan_context, &vulkan_lhs, &vulkan_rhs, &vulkan_output, 1)
                    .expect("Vulkan timestamp profiling failed")
            },
        );

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

        results.push(CaseResult {
            numel,
            dispatches,
            sync_cube: sync_cube_time,
            sync_vulkan: sync_vulkan_time,
            batch_cube: batch_cube_time,
            batch_vulkan: batch_vulkan_time,
            device_cube: device_cube_time,
            device_vulkan: device_vulkan_time,
        });
    }

    print_table(
        "synchronous end-to-end latency (one completion per dispatch)",
        &results,
        |result| result.dispatches,
        |result| (result.sync_cube, result.sync_vulkan),
    );
    print_table(
        "batched runtime throughput (one explicit final sync, per-dispatch average)",
        &results,
        |result| result.dispatches,
        |result| (result.batch_cube, result.batch_vulkan),
    );
    print_table(
        "GPU timestamp latency (one dispatch per sample)",
        &results,
        |_| 1,
        |result| (result.device_cube, result.device_vulkan),
    );
}

fn measure_pairs(
    mut cube_operation: impl FnMut() -> Duration,
    mut vulkan_operation: impl FnMut() -> Duration,
) -> (Duration, Duration) {
    let mut cube_samples = Vec::with_capacity(SAMPLES);
    let mut vulkan_samples = Vec::with_capacity(SAMPLES);

    for sample in 0..SAMPLES {
        if sample % 2 == 0 {
            cube_samples.push(cube_operation());
            vulkan_samples.push(vulkan_operation());
        } else {
            vulkan_samples.push(vulkan_operation());
            cube_samples.push(cube_operation());
        }
    }

    (median(cube_samples), median(vulkan_samples))
}

fn profile_cube_vector_add(
    client: &ComputeClient<WgpuRuntime>,
    context: &CubeContext<WgpuRuntime>,
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
    dispatches: usize,
) -> Duration {
    sync_cube(client);
    let (_, profile) = client
        .profile(
            || {
                for _ in 0..dispatches {
                    cube_vector_add(context, lhs, rhs, output)
                        .expect("CubeCL profiled launch failed");
                }
            },
            "vector_add",
        )
        .expect("CubeCL timestamp profiling failed");
    assert_eq!(
        profile.timing_method(),
        TimingMethod::Device,
        "CubeCL WGPU timestamp queries are unavailable"
    );
    cubecl::future::block_on(profile.resolve()).duration()
}

fn print_table(
    title: &str,
    results: &[CaseResult],
    dispatches: impl Fn(&CaseResult) -> usize,
    select: impl Fn(&CaseResult) -> (Duration, Duration),
) {
    println!();
    println!("{title}:");
    println!(
        "{:>12} {:>12} {:>14} {:>14} {:>12}",
        "elements", "dispatches", "CubeCL us", "Vulkan us", "Cube/Vk"
    );
    for result in results {
        let (cube, vulkan) = select(result);
        println!(
            "{:>12} {:>12} {:>14.3} {:>14.3} {:>12.2}",
            result.numel,
            dispatches(result),
            micros(cube),
            micros(vulkan),
            cube.as_secs_f64() / vulkan.as_secs_f64(),
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

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn sync_cube(client: &ComputeClient<WgpuRuntime>) {
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

use std::{env, error::Error, ffi::c_void, ptr::null_mut};

use cubecl::{
    CubeElement, MemoryConfiguration,
    hip::{AmdDevice, HipRuntime, RuntimeOptions},
};
use cubecl_hip_sys::{
    HIP_SUCCESS, hipEvent_t, hipEventCreate, hipEventDestroy, hipEventElapsedTime, hipEventRecord,
    hipEventSynchronize, hipMemGetInfo, hipSetDevice,
};
use cubecl_runtime::{
    memory_management::{MemoryPoolOptions, PoolType},
    native::NativeError,
};
use half::f16;
use mote_core::Tensor;
use mote_cube::{CubeContext, matmul::matmul_cmma_f16_f32};
use mote_types::{BackendKind, DType, Device, Encoding, Layout, Shape, TensorDesc};

const SIZES: [usize; 4] = [256, 512, 1024, 2048];
const WARMUPS: usize = 5;
const SAMPLES: usize = 7;
const MIB: usize = 1024 * 1024;

fn main() {
    run().unwrap();
}

fn run() -> Result<(), Box<dyn Error>> {
    check_hip(unsafe { hipSetDevice(0) }, "hipSetDevice")?;
    let baseline_free = free_device_bytes()?;
    let (memory_name, memory_config) = benchmark_memory_configuration()?;
    let client = HipRuntime::init_client(&AmdDevice::default(), RuntimeOptions { memory_config })?;
    let context = CubeContext::<HipRuntime>::new(Device::new(BackendKind::Hip, 0), client)?;

    if !context.supports_cooperative_matrix() {
        return Err("CubeCL HIP runtime reports no cooperative-matrix support".into());
    }

    println!("CubeK/CubeCL HIP: F16 inputs, F32 accumulation/output");
    println!("memory configuration: {memory_name}");
    println!(
        "cooperative-matrix configurations: {}",
        context.cooperative_matrix_config_count()
    );
    println!("samples per case: {SAMPLES} (median reported)\n");
    println!(
        "{:>8}{:>15}{:>11}{:>15}{:>15}",
        "M=N=K", "CubeK HIP us", "TFLOP/s", "tensor MiB", "HIP heap MiB"
    );

    let start = HipEvent::new()?;
    let stop = HipEvent::new()?;

    for size in SIZES {
        let elements = size * size;
        let lhs_values = matrix_values(size, lhs_value);
        let rhs_values = matrix_values(size, rhs_value);
        let lhs = context.from_bytes(matrix_desc(size, DType::F16), f16::as_bytes(&lhs_values))?;
        let rhs = context.from_bytes(matrix_desc(size, DType::F16), f16::as_bytes(&rhs_values))?;
        let output = context.empty(matrix_desc(size, DType::F32))?;
        debug_assert_eq!(output.desc().numel(), elements);

        for _ in 0..WARMUPS {
            matmul_cmma_f16_f32(&context, &lhs, &rhs, &output)?;
        }
        context.sync()?;
        let heap_mib = bytes_to_mib(baseline_free.saturating_sub(free_device_bytes()?));

        let mut samples = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            record_event(&context, &[&lhs, &rhs, &output], start.raw())?;
            matmul_cmma_f16_f32(&context, &lhs, &rhs, &output)?;
            record_event(&context, &[&output], stop.raw())?;
            check_hip(
                unsafe { hipEventSynchronize(stop.raw()) },
                "hipEventSynchronize(stop)",
            )?;

            let mut milliseconds = 0.0_f32;
            check_hip(
                unsafe { hipEventElapsedTime(&mut milliseconds, start.raw(), stop.raw()) },
                "hipEventElapsedTime",
            )?;
            samples.push(milliseconds);
        }

        samples.sort_by(f32::total_cmp);
        let milliseconds = f64::from(samples[SAMPLES / 2]);
        let operations = 2.0 * (size as f64).powi(3);
        let tflops = operations / (milliseconds / 1000.0) / 1.0e12;

        let actual_bytes = context.read_bytes(&output)?;
        let actual = f32::from_bytes(&actual_bytes);
        validate(size, &lhs_values, &rhs_values, actual)?;

        let tensor_mib = bytes_to_mib(elements * 8);
        println!(
            "{size:>8}{:>15.3}{tflops:>11.2}{tensor_mib:>15.2}{heap_mib:>15.2}",
            milliseconds * 1000.0
        );
    }

    Ok(())
}

fn record_event(
    context: &CubeContext<HipRuntime>,
    tensors: &[&Tensor],
    event: hipEvent_t,
) -> Result<(), Box<dyn Error>> {
    // Raw HIP handles are not Send in the bindings, but the callback runs on
    // CubeCL's server thread. Move only the pointer bits across that boundary.
    let event_address = event.cast::<c_void>() as usize;
    context.submit_hip_native(tensors, move |stream, _resources| {
        let event = event_address as hipEvent_t;
        let status = unsafe { hipEventRecord(event, *stream) };
        if status == HIP_SUCCESS {
            Ok(())
        } else {
            Err(NativeError::new(format!(
                "hipEventRecord failed with status {status}"
            )))
        }
    })?;
    Ok(())
}

struct HipEvent(hipEvent_t);

impl HipEvent {
    fn new() -> Result<Self, Box<dyn Error>> {
        let mut event = null_mut();
        check_hip(unsafe { hipEventCreate(&mut event) }, "hipEventCreate")?;
        Ok(Self(event))
    }

    fn raw(&self) -> hipEvent_t {
        self.0
    }
}

impl Drop for HipEvent {
    fn drop(&mut self) {
        let _ = unsafe { hipEventDestroy(self.0) };
    }
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

fn check_hip(status: cubecl_hip_sys::hipError_t, operation: &str) -> Result<(), Box<dyn Error>> {
    if status == HIP_SUCCESS {
        Ok(())
    } else {
        Err(format!("{operation} failed with status {status}").into())
    }
}

fn free_device_bytes() -> Result<usize, Box<dyn Error>> {
    let mut free = 0;
    let mut total = 0;
    check_hip(
        unsafe { hipMemGetInfo(&mut free, &mut total) },
        "hipMemGetInfo",
    )?;
    Ok(free)
}

fn bytes_to_mib(bytes: usize) -> f64 {
    bytes as f64 / MIB as f64
}

fn benchmark_memory_configuration() -> Result<(&'static str, MemoryConfiguration), Box<dyn Error>> {
    let mode = env::var("MOTE_CUBE_HIP_MEMORY").unwrap_or_else(|_| "default".into());
    match mode.as_str() {
        "default" => Ok((
            "CubeCL default sub-slice pools",
            MemoryConfiguration::default(),
        )),
        "exclusive" => Ok(("exclusive pages", MemoryConfiguration::ExclusivePages)),
        "bounded" => Ok(("bounded sub-slice pools", bounded_memory_configuration())),
        _ => Err(format!(
            "invalid MOTE_CUBE_HIP_MEMORY={mode:?}; expected default, bounded, or exclusive"
        )
        .into()),
    }
}

fn bounded_memory_configuration() -> MemoryConfiguration {
    const MIB_U64: u64 = 1024 * 1024;

    MemoryConfiguration::Custom {
        pool_options: vec![
            MemoryPoolOptions {
                pool_type: PoolType::SlicedPages {
                    page_size: 8 * MIB_U64,
                    max_slice_size: 768 * 1024,
                },
                dealloc_period: None,
            },
            MemoryPoolOptions {
                pool_type: PoolType::SlicedPages {
                    page_size: 16 * MIB_U64,
                    max_slice_size: 6 * MIB_U64,
                },
                dealloc_period: None,
            },
            MemoryPoolOptions {
                pool_type: PoolType::SlicedPages {
                    page_size: 64 * MIB_U64,
                    max_slice_size: 48 * MIB_U64,
                },
                dealloc_period: None,
            },
            MemoryPoolOptions {
                pool_type: PoolType::ExclusivePages {
                    max_alloc_size: 16 * 1024 * MIB_U64,
                },
                dealloc_period: Some(64),
            },
        ],
    }
}

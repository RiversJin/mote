#![cfg(feature = "hip")]

use std::ffi::c_void;

use cubecl::{
    hip::{AmdDevice, HipRuntime},
    prelude::*,
};
use cubecl_hip_sys::{
    HIP_SUCCESS, hipMemcpy, hipMemcpyKind_hipMemcpyDeviceToHost,
    hipMemcpyKind_hipMemcpyHostToDevice, hipMemsetAsync, hipSetDevice,
};
use cubecl_runtime::native::NativeError;
use mote_cube::CubeContext;
use mote_types::{BackendKind, DType, Device, Encoding, Layout, Shape, TensorDesc};

#[test]
fn native_hip_reads_and_writes_cubecl_device_memory() {
    assert_hip_success(unsafe { hipSetDevice(0) });

    let context = CubeContext::<HipRuntime>::new(
        Device::new(BackendKind::Hip, 0),
        HipRuntime::client(&AmdDevice::default()),
    )
    .unwrap();
    let original = [1.0_f32, 2.0, 3.0, 4.0];
    let tensor = context
        .from_bytes(plain_f32_desc(original.len()), f32::as_bytes(&original))
        .unwrap();

    context.sync().unwrap();
    let exported = context.export_hip_device_slice(&tensor).unwrap();
    assert_eq!(exported.len_bytes(), size_of_val(&original));

    let mut read_by_hip = [0.0_f32; 4];
    assert_hip_success(unsafe {
        hipMemcpy(
            read_by_hip.as_mut_ptr().cast::<c_void>(),
            exported.as_mut_ptr().cast_const(),
            exported.len_bytes(),
            hipMemcpyKind_hipMemcpyDeviceToHost,
        )
    });
    assert_eq!(read_by_hip, original);

    let written_by_hip = [10.0_f32, 20.0, 30.0, 40.0];
    assert_hip_success(unsafe {
        hipMemcpy(
            exported.as_mut_ptr(),
            written_by_hip.as_ptr().cast::<c_void>(),
            exported.len_bytes(),
            hipMemcpyKind_hipMemcpyHostToDevice,
        )
    });

    let read_by_cubecl = context.read_bytes(&tensor).unwrap();
    assert_eq!(f32::from_bytes(&read_by_cubecl), written_by_hip);
}

#[test]
fn native_submission_stays_ordered_on_cubecl_stream() {
    assert_hip_success(unsafe { hipSetDevice(0) });

    let context = CubeContext::<HipRuntime>::new(
        Device::new(BackendKind::Hip, 0),
        HipRuntime::client(&AmdDevice::default()),
    )
    .unwrap();
    let original = [1.0_f32, 2.0, 3.0, 4.0];
    let tensor = context
        .from_bytes(plain_f32_desc(original.len()), f32::as_bytes(&original))
        .unwrap();

    context
        .submit_hip_native(&[&tensor], |stream, resources| {
            let resource = &resources[0];
            let status =
                unsafe { hipMemsetAsync(resource.ptr, 0, resource.size as usize, *stream) };
            if status == HIP_SUCCESS {
                Ok(())
            } else {
                Err(NativeError::new(format!(
                    "hipMemsetAsync failed with status {status}"
                )))
            }
        })
        .unwrap();

    // The read is submitted after the native memset on the same physical
    // stream, so no explicit CubeCL or HIP synchronization is needed here.
    let actual = context.read_bytes(&tensor).unwrap();
    assert_eq!(&actual[..], &[0; size_of::<[f32; 4]>()]);
}

fn assert_hip_success(status: cubecl_hip_sys::hipError_t) {
    assert_eq!(status, HIP_SUCCESS, "HIP call failed with status {status}");
}

fn plain_f32_desc(len: usize) -> TensorDesc {
    TensorDesc::new(
        Shape::new(&[len]),
        Encoding::Plain(DType::F32),
        Layout::Contiguous,
    )
    .unwrap()
}

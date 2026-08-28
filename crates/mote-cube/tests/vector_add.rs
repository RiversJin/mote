#![cfg(any(feature = "cpu", feature = "hip"))]

#[cfg(feature = "cpu")]
use cubecl::cpu::{CpuDevice, CpuRuntime};
#[cfg(feature = "hip")]
use cubecl::hip::{AmdDevice, HipRuntime};
use cubecl::prelude::*;
#[cfg(feature = "cpu")]
use mote_cube::CubeError;
use mote_cube::{CubeContext, vector_add::vector_add};
#[cfg(feature = "hip")]
use mote_types::BackendKind;
use mote_types::{DType, Device, Encoding, Layout, Shape, TensorDesc};

fn assert_adds_vectors<R: Runtime>(device: Device, client: ComputeClient<R>) {
    let lhs = (0..1_000).map(|value| value as f32).collect::<Vec<_>>();
    let rhs = (0..1_000)
        .map(|value| (value as f32) * 0.5)
        .collect::<Vec<_>>();
    let expected = lhs
        .iter()
        .zip(&rhs)
        .map(|(lhs, rhs)| lhs + rhs)
        .collect::<Vec<_>>();

    let context = CubeContext::<R>::new(device, client).unwrap();
    let desc = plain_desc(lhs.len(), DType::F32);
    let lhs = context
        .from_bytes(desc.clone(), f32::as_bytes(&lhs))
        .unwrap();
    let rhs = context
        .from_bytes(desc.clone(), f32::as_bytes(&rhs))
        .unwrap();
    let output = context.empty(desc).unwrap();

    vector_add(&context, &lhs, &rhs, &output).unwrap();

    let actual = context.read_bytes(&output).unwrap();
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual, expected);
}

fn plain_desc(len: usize, dtype: DType) -> TensorDesc {
    TensorDesc::new(
        Shape::new(&[len]),
        Encoding::Plain(dtype),
        Layout::Contiguous,
    )
    .unwrap()
}

#[cfg(feature = "cpu")]
#[test]
fn adds_vectors_on_cpu() {
    assert_adds_vectors::<CpuRuntime>(Device::cpu(), CpuRuntime::client(&CpuDevice));
}

#[cfg(feature = "hip")]
#[test]
fn adds_vectors_on_hip() {
    assert_adds_vectors::<HipRuntime>(
        Device::new(BackendKind::Hip, 0),
        HipRuntime::client(&AmdDevice::default()),
    );
}

#[cfg(feature = "cpu")]
#[test]
fn rejects_mismatched_shapes() {
    let context =
        CubeContext::<CpuRuntime>::new(Device::cpu(), CpuRuntime::client(&CpuDevice)).unwrap();
    let lhs = context.empty(plain_desc(4, DType::F32)).unwrap();
    let rhs = context.empty(plain_desc(3, DType::F32)).unwrap();
    let output = context.empty(plain_desc(4, DType::F32)).unwrap();

    let error = vector_add(&context, &lhs, &rhs, &output).unwrap_err();

    assert!(matches!(
        error,
        CubeError::ShapeMismatch { tensor: "rhs", .. }
    ));
}

#[cfg(feature = "cpu")]
#[test]
fn rejects_non_f32_tensors() {
    let context =
        CubeContext::<CpuRuntime>::new(Device::cpu(), CpuRuntime::client(&CpuDevice)).unwrap();
    let desc = plain_desc(4, DType::I32);
    let lhs = context.empty(desc.clone()).unwrap();
    let rhs = context.empty(desc.clone()).unwrap();
    let output = context.empty(desc).unwrap();

    let error = vector_add(&context, &lhs, &rhs, &output).unwrap_err();

    assert!(matches!(
        error,
        CubeError::DTypeMismatch {
            tensor: "lhs",
            expected: DType::F32,
            actual: DType::I32,
        }
    ));
}

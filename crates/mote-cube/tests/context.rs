#![cfg(feature = "cpu")]

use cubecl::{
    cpu::{CpuDevice, CpuRuntime},
    prelude::Runtime,
};
use mote_core::{CpuOwnedStorage, Storage, Tensor};
use mote_cube::{CubeContext, CubeError};
use mote_types::{DType, Device, Encoding, Layout, Shape, Strides, TensorDesc};

fn context() -> CubeContext<CpuRuntime> {
    CubeContext::new(Device::cpu(), CpuRuntime::client(&CpuDevice)).unwrap()
}

fn plain_desc(shape: &[usize], layout: Layout) -> TensorDesc {
    TensorDesc::new(Shape::new(shape), Encoding::Plain(DType::F32), layout).unwrap()
}

#[test]
fn rejects_upload_with_the_wrong_byte_length() {
    let context = context();
    let error = context
        .from_bytes(plain_desc(&[2], Layout::Contiguous), &[0; 4])
        .unwrap_err();

    assert!(matches!(
        error,
        CubeError::ByteLengthMismatch {
            expected: 8,
            actual: 4,
        }
    ));
}

#[test]
fn rejects_non_contiguous_allocation() {
    let context = context();
    let desc = plain_desc(&[2], Layout::Strided(Strides::new(&[2])));

    let error = context.empty(desc).unwrap_err();

    assert!(matches!(error, CubeError::UnsupportedLayout { .. }));
}

#[test]
fn rejects_non_cube_storage() {
    let context = context();
    let desc = plain_desc(&[2], Layout::Contiguous);
    let storage = Storage::new(CpuOwnedStorage::zeroed(8).unwrap()).unwrap();
    let tensor = Tensor::new(desc, storage, 0).unwrap();

    let error = context.read_bytes(&tensor).unwrap_err();

    assert!(matches!(error, CubeError::WrongStorageRuntime { .. }));
}

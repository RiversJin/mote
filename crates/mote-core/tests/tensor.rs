use std::any::Any;

use mote_core::{CpuOwnedStorage, Device, Storage, StorageImpl, Tensor, TensorError};
use mote_types::{DType, Encoding, Layout, Shape, Strides, TensorDesc};

#[derive(Debug)]
struct SyntheticStorage {
    device: Device,
    size_bytes: usize,
    alignment: usize,
}

impl StorageImpl for SyntheticStorage {
    fn device(&self) -> &Device {
        &self.device
    }

    fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    fn alignment(&self) -> usize {
        self.alignment
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn plain_desc(dims: &[usize], dtype: DType) -> TensorDesc {
    TensorDesc::new(Shape::new(dims), Encoding::Plain(dtype), Layout::Contiguous).unwrap()
}

#[test]
fn tensor_can_exactly_fill_storage() {
    let storage = Storage::new(CpuOwnedStorage::zeroed(16).unwrap()).unwrap();
    let tensor = Tensor::new(plain_desc(&[4], DType::F32), storage, 0).unwrap();

    assert_eq!(tensor.byte_offset(), 0);
    assert_eq!(tensor.desc().required_span_bytes(), 16);
    assert_eq!(tensor.device(), &Device::cpu());
}

#[test]
fn tensor_can_begin_at_a_non_zero_byte_offset() {
    let storage = Storage::new(CpuOwnedStorage::zeroed(32).unwrap()).unwrap();
    let tensor = Tensor::new(plain_desc(&[2], DType::F32), storage, 8).unwrap();

    assert_eq!(tensor.byte_offset(), 8);
}

#[test]
fn rejects_a_tensor_that_exceeds_storage() {
    let storage = Storage::new(CpuOwnedStorage::zeroed(16).unwrap()).unwrap();

    assert_eq!(
        Tensor::new(plain_desc(&[4], DType::F32), storage, 4).unwrap_err(),
        TensorError::StorageTooSmall {
            byte_offset: 4,
            end_offset: 20,
            storage_size: 16,
        }
    );
}

#[test]
fn strided_tensor_bounds_use_the_physical_span() {
    let descriptor = TensorDesc::new(
        Shape::new(&[2, 3]),
        Encoding::Plain(DType::F16),
        Layout::Strided(Strides::new(&[5, 1])),
    )
    .unwrap();
    let storage = Storage::new(CpuOwnedStorage::zeroed(12).unwrap()).unwrap();

    assert_eq!(
        Tensor::new(descriptor, storage, 0).unwrap_err(),
        TensorError::StorageTooSmall {
            byte_offset: 0,
            end_offset: 16,
            storage_size: 12,
        }
    );
}

#[test]
fn rejects_an_offset_beyond_storage() {
    let storage = Storage::new(CpuOwnedStorage::zeroed(16).unwrap()).unwrap();

    assert_eq!(
        Tensor::new(plain_desc(&[0], DType::F32), storage, 17).unwrap_err(),
        TensorError::OffsetOutOfBounds {
            byte_offset: 17,
            storage_size: 16,
        }
    );
}

#[test]
fn rejects_a_misaligned_tensor_offset() {
    let storage = Storage::new(CpuOwnedStorage::zeroed(16).unwrap()).unwrap();
    let storage_alignment = storage.alignment();

    assert_eq!(
        Tensor::new(plain_desc(&[1], DType::F32), storage, 2).unwrap_err(),
        TensorError::Misaligned {
            byte_offset: 2,
            required_alignment: 4,
            storage_alignment,
        }
    );
}

#[test]
fn rejects_insufficient_storage_base_alignment() {
    let storage = Storage::new(SyntheticStorage {
        device: Device::cpu(),
        size_bytes: 16,
        alignment: 2,
    })
    .unwrap();

    assert_eq!(
        Tensor::new(plain_desc(&[1], DType::F32), storage, 0).unwrap_err(),
        TensorError::Misaligned {
            byte_offset: 0,
            required_alignment: 4,
            storage_alignment: 2,
        }
    );
}

#[test]
fn rejects_an_overflowing_byte_range() {
    let storage = Storage::new(SyntheticStorage {
        device: Device::cpu(),
        size_bytes: usize::MAX,
        alignment: 1,
    })
    .unwrap();

    assert_eq!(
        Tensor::new(plain_desc(&[usize::MAX], DType::U8), storage, 1).unwrap_err(),
        TensorError::ByteRangeOverflow {
            byte_offset: 1,
            span_bytes: usize::MAX,
        }
    );
}

#[test]
fn cloned_tensors_share_storage_identity() {
    let storage = Storage::new(CpuOwnedStorage::zeroed(16).unwrap()).unwrap();
    let tensor = Tensor::new(plain_desc(&[4], DType::F32), storage, 0).unwrap();
    let cloned = tensor.clone();

    assert!(tensor.shares_storage_with(&cloned));
    assert_eq!(tensor.storage_id(), cloned.storage_id());
}

#[test]
fn empty_tensor_may_start_at_the_end_of_storage() {
    let storage = Storage::new(CpuOwnedStorage::zeroed(3).unwrap()).unwrap();
    let tensor = Tensor::new(plain_desc(&[0], DType::F32), storage, 3).unwrap();

    assert_eq!(tensor.byte_offset(), 3);
}

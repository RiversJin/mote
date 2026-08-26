use std::any::Any;

use mote_core::{BackendKind, CpuOwnedStorage, Device, Storage, StorageError, StorageImpl};

#[derive(Debug)]
struct InvalidAlignmentStorage {
    device: Device,
}

impl StorageImpl for InvalidAlignmentStorage {
    fn device(&self) -> &Device {
        &self.device
    }

    fn size_bytes(&self) -> usize {
        16
    }

    fn alignment(&self) -> usize {
        3
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[test]
fn owns_zeroed_cpu_bytes() {
    let implementation = CpuOwnedStorage::zeroed(13).unwrap();

    assert_eq!(implementation.as_bytes(), &[0; 13]);
    assert_eq!(implementation.size_bytes(), 13);
    assert_eq!(implementation.device(), &Device::cpu());
    assert!(implementation.alignment() >= 4);
}

#[test]
fn copies_cpu_bytes_and_supports_type_erasure() {
    let storage = Storage::new(CpuOwnedStorage::from_bytes(&[1, 2, 3, 4]).unwrap()).unwrap();
    let implementation = storage.downcast_ref::<CpuOwnedStorage>().unwrap();

    assert_eq!(implementation.as_bytes(), &[1, 2, 3, 4]);
    assert_eq!(storage.device().backend(), BackendKind::Cpu);
}

#[test]
fn clones_share_an_identity_but_independent_storages_do_not() {
    let storage = Storage::new(CpuOwnedStorage::zeroed(8).unwrap()).unwrap();
    let cloned = storage.clone();
    let independent = Storage::new(CpuOwnedStorage::zeroed(8).unwrap()).unwrap();

    assert!(storage.shares_allocation_with(&cloned));
    assert_eq!(storage.id(), cloned.id());
    assert!(!storage.shares_allocation_with(&independent));
    assert_ne!(storage.id(), independent.id());
}

#[test]
fn rejects_invalid_storage_alignment() {
    assert_eq!(
        Storage::new(InvalidAlignmentStorage {
            device: Device::cpu(),
        })
        .unwrap_err(),
        StorageError::InvalidAlignment { alignment: 3 }
    );
}

//! Core tensor and storage types for Mote.

mod storage;
mod tensor;

pub use mote_types::{BackendKind, Device};
pub use storage::{
    CpuOwnedStorage, CpuStorageError, Storage, StorageError, StorageId, StorageImpl,
};
pub use tensor::{Tensor, TensorError};

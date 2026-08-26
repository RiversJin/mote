use std::{
    any::Any,
    fmt,
    mem::{align_of, size_of},
    slice,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::Device;

static NEXT_STORAGE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StorageId(u64);

impl StorageId {
    pub const fn get(self) -> u64 {
        self.0
    }

    fn fresh() -> Result<Self, StorageError> {
        NEXT_STORAGE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map(Self)
            .map_err(|_| StorageError::IdExhausted)
    }
}

pub trait StorageImpl: Send + Sync + 'static {
    fn device(&self) -> &Device;
    fn size_bytes(&self) -> usize;
    fn alignment(&self) -> usize;
    fn as_any(&self) -> &dyn Any;
}

#[derive(Clone)]
pub struct Storage {
    id: StorageId,
    inner: Arc<dyn StorageImpl>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StorageError {
    #[error("storage alignment must be a non-zero power of two, got {alignment}")]
    InvalidAlignment { alignment: usize },

    #[error("storage identity space is exhausted")]
    IdExhausted,
}

impl Storage {
    pub fn new(inner: impl StorageImpl) -> Result<Self, StorageError> {
        let alignment = inner.alignment();
        if !alignment.is_power_of_two() {
            return Err(StorageError::InvalidAlignment { alignment });
        }

        Ok(Self {
            id: StorageId::fresh()?,
            inner: Arc::new(inner),
        })
    }

    pub fn id(&self) -> StorageId {
        self.id
    }

    pub fn device(&self) -> &Device {
        self.inner.device()
    }

    pub fn size_bytes(&self) -> usize {
        self.inner.size_bytes()
    }

    pub fn alignment(&self) -> usize {
        self.inner.alignment()
    }

    pub fn downcast_ref<T: StorageImpl>(&self) -> Option<&T> {
        self.inner.as_any().downcast_ref()
    }

    pub fn shares_allocation_with(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl fmt::Debug for Storage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Storage")
            .field("id", &self.id)
            .field("device", self.device())
            .field("size_bytes", &self.size_bytes())
            .field("alignment", &self.alignment())
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub struct CpuOwnedStorage {
    device: Device,
    words: Box<[usize]>,
    size_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("failed to allocate {size_bytes} bytes for CPU storage")]
pub struct CpuStorageError {
    pub size_bytes: usize,
}

impl CpuOwnedStorage {
    pub fn zeroed(size_bytes: usize) -> Result<Self, CpuStorageError> {
        let word_count = size_bytes.div_ceil(size_of::<usize>());
        let mut words = Vec::new();
        words
            .try_reserve_exact(word_count)
            .map_err(|_| CpuStorageError { size_bytes })?;
        words.resize(word_count, 0);

        Ok(Self {
            device: Device::cpu(),
            words: words.into_boxed_slice(),
            size_bytes,
        })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CpuStorageError> {
        let mut storage = Self::zeroed(bytes.len())?;
        storage.as_bytes_mut().copy_from_slice(bytes);
        Ok(storage)
    }

    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: every word is initialized, the byte slice stays within the
        // allocation, and u8 has an alignment of one.
        unsafe { slice::from_raw_parts(self.words.as_ptr().cast(), self.size_bytes) }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: this is the unique mutable borrow of initialized storage,
        // bounded by the requested logical byte size.
        unsafe { slice::from_raw_parts_mut(self.words.as_mut_ptr().cast(), self.size_bytes) }
    }
}

impl StorageImpl for CpuOwnedStorage {
    fn device(&self) -> &Device {
        &self.device
    }

    fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    fn alignment(&self) -> usize {
        align_of::<usize>()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

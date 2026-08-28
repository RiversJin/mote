use mote_core::{StorageError, TensorError};
use mote_types::{DType, Device, Encoding, Layout, Shape};

#[derive(Debug, thiserror::Error)]
pub enum VulkanError {
    #[error("Vulkan {operation} failed: {message}")]
    Backend {
        operation: &'static str,
        message: String,
    },

    #[error("no Vulkan physical device with a compute queue is available")]
    NoComputeDevice,

    #[error("Vulkan device ordinal {ordinal} is unavailable; found {available} compute device(s)")]
    DeviceUnavailable { ordinal: u32, available: usize },

    #[error("Vulkan storage size {size_bytes} does not fit in VkDeviceSize")]
    StorageSizeTooLarge { size_bytes: usize },

    #[error("byte length mismatch: expected {expected}, got {actual}")]
    ByteLengthMismatch { expected: usize, actual: usize },

    #[error("unsupported tensor encoding {actual:?}")]
    UnsupportedEncoding { actual: Encoding },

    #[error("unsupported tensor layout {actual:?}; raw Vulkan v0 only accepts contiguous tensors")]
    UnsupportedLayout { actual: Layout },

    #[error("tensor device mismatch: expected {expected:?}, got {actual:?}")]
    DeviceMismatch { expected: Device, actual: Device },

    #[error("tensor has byte offset {byte_offset}; raw Vulkan v0 only accepts zero-offset tensors")]
    UnsupportedByteOffset { byte_offset: usize },

    #[error("tensor is not backed by storage created by this Vulkan context")]
    WrongStorage,

    #[error("Vulkan storage uses an unsupported external or sparse memory binding")]
    UnsupportedStorageMemory,

    #[error("{tensor} dtype mismatch: expected {expected:?}, got {actual:?}")]
    DTypeMismatch {
        tensor: &'static str,
        expected: DType,
        actual: DType,
    },

    #[error("{tensor} shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        tensor: &'static str,
        expected: Shape,
        actual: Shape,
    },

    #[error("launch geometry overflow for {numel} elements")]
    LaunchGeometryOverflow { numel: usize },

    #[error(transparent)]
    Storage(#[from] StorageError),

    #[error(transparent)]
    Tensor(#[from] TensorError),
}

impl VulkanError {
    pub(crate) fn backend(operation: &'static str, error: impl std::fmt::Display) -> Self {
        Self::Backend {
            operation,
            message: error.to_string(),
        }
    }
}

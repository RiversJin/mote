use cubecl::server::ServerError;
use mote_core::{StorageError, TensorError};
use mote_types::{DType, Device, Encoding, Layout, Shape};

#[derive(Debug, thiserror::Error)]
pub enum CubeError {
    #[error("tensor device {actual:?} does not match context device {expected:?}")]
    DeviceMismatch { expected: Device, actual: Device },

    #[error("tensor storage is not CubeStorage<{expected_runtime}>")]
    WrongStorageRuntime { expected_runtime: &'static str },

    #[error("unsupported tensor encoding: {actual:?}")]
    UnsupportedEncoding { actual: Encoding },

    #[error("{tensor} tensor has dtype {actual:?}, expected {expected:?}")]
    DTypeMismatch {
        tensor: &'static str,
        expected: DType,
        actual: DType,
    },

    #[error("unsupported tensor layout: {actual:?}; expected contiguous layout")]
    UnsupportedLayout { actual: Layout },

    #[error("{tensor} tensor has shape {actual:?}, expected {expected:?}")]
    ShapeMismatch {
        tensor: &'static str,
        expected: Shape,
        actual: Shape,
    },

    #[error("non-zero tensor byte offset {byte_offset} is unsupported")]
    UnsupportedByteOffset { byte_offset: usize },

    #[error("input byte length mismatch: expected {expected} bytes, got {actual}")]
    ByteLengthMismatch { expected: usize, actual: usize },

    #[error("tensor element count {numel} exceeds CubeCL launch geometry")]
    LaunchGeometryOverflow { numel: usize },

    #[error(transparent)]
    Storage(#[from] StorageError),

    #[error(transparent)]
    Tensor(#[from] TensorError),

    #[error("CubeCL readback failed: {source}")]
    Readback {
        #[source]
        source: ServerError,
    },

    #[error("CubeCL storage align {align} doesn't fit usize")]
    StorageAlignTooLarge { align: u64 },
}

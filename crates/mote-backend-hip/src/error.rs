use mote_core::{StorageError, TensorError};
use mote_types::{DType, Device, Encoding, Layout, Shape};

use crate::ffi::hip;

#[derive(Debug, thiserror::Error)]
pub enum HipError {
    #[error("HIP {operation} failed with status {status}")]
    Runtime {
        operation: &'static str,
        status: i32,
    },

    #[error("hipBLASLt {operation} failed with status {status}: {message}")]
    BlasLt {
        operation: &'static str,
        status: i32,
        message: String,
    },

    #[error("HIP device ordinal {ordinal} is unavailable; found {available} device(s)")]
    DeviceUnavailable { ordinal: u32, available: i32 },

    #[error("HIP device ordinal {ordinal} does not fit the HIP API")]
    DeviceOrdinalTooLarge { ordinal: u32 },

    #[error("byte length mismatch: expected {expected}, got {actual}")]
    ByteLengthMismatch { expected: usize, actual: usize },

    #[error("unsupported tensor encoding {actual:?}")]
    UnsupportedEncoding { actual: Encoding },

    #[error("unsupported tensor layout {actual:?}; native HIP v0 only accepts contiguous tensors")]
    UnsupportedLayout { actual: Layout },

    #[error("tensor device mismatch: expected {expected:?}, got {actual:?}")]
    DeviceMismatch { expected: Device, actual: Device },

    #[error("tensor has byte offset {byte_offset}; native HIP v0 only accepts zero-offset tensors")]
    UnsupportedByteOffset { byte_offset: usize },

    #[error("tensor is not backed by storage created by this HIP context")]
    WrongStorage,

    #[error("{tensor} must not share storage with {other}: concurrent writes would conflict")]
    AliasedStorage {
        tensor: &'static str,
        other: &'static str,
    },

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

    #[error("{tensor} tensor has rank {actual}, expected rank {expected}")]
    RankMismatch {
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },

    #[error("{tensor} tensor has rank {actual}, expected rank >= {minimum}")]
    RankTooSmall {
        tensor: &'static str,
        minimum: usize,
        actual: usize,
    },

    #[error("matrix dimensions M={m}, N={n}, K={k} do not fit hipBLASLt")]
    MatmulDimensionsTooLarge { m: usize, n: usize, k: usize },

    #[error(
        "Q4_0 linear dimensions rows={rows}, output_features={output_features}, input_features={input_features} do not fit the HIP launch grid or its u64 ABI"
    )]
    QuantizedLinearDimensionsTooLarge {
        rows: usize,
        output_features: usize,
        input_features: usize,
    },

    #[error("RMSNorm dimensions rows={rows}, hidden={hidden} do not fit the HIP launch grid")]
    RmsNormDimensionsTooLarge { rows: usize, hidden: usize },

    #[error(
        "fused add + RMSNorm dimensions rows={rows}, hidden={hidden} do not fit the HIP launch grid or its u64 ABI"
    )]
    FusedAddRmsNormDimensionsTooLarge { rows: usize, hidden: usize },

    #[error(
        "RoPE dimensions tokens={tokens}, heads={heads}, head_dim={head_dim}, rotary_dim={rotary_dim} do not fit the HIP launch grid"
    )]
    RopeDimensionsTooLarge {
        tokens: usize,
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
    },

    #[error("rotary dimension {rotary_dim} must be even and at most head_dim={head_dim}")]
    InvalidRotaryDim { rotary_dim: usize, head_dim: usize },

    #[error("epsilon must be finite and non-negative, got {epsilon}")]
    InvalidEpsilon { epsilon: f32 },

    #[error("HIP backend state lock was poisoned")]
    StatePoisoned,

    #[error(transparent)]
    Storage(#[from] StorageError),

    #[error(transparent)]
    Tensor(#[from] TensorError),
}

pub(crate) fn check_hip(status: hip::Status, operation: &'static str) -> Result<(), HipError> {
    if status == hip::SUCCESS {
        Ok(())
    } else {
        Err(HipError::Runtime { operation, status })
    }
}

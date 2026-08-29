//! Native ROCm/HIP specialization backend.
//!
//! The backend owns its HIP stream and device allocations. `hipBLASLt` consumes
//! those allocations directly; CubeCL is not involved in storage or execution.

#[cfg(feature = "rocm")]
mod context;
#[cfg(feature = "rocm")]
mod error;
#[cfg(feature = "rocm")]
mod ffi;
#[cfg(feature = "rocm")]
mod fused_add_rms_norm;
#[cfg(feature = "rocm")]
mod matmul;
#[cfg(feature = "rocm")]
mod quantized_linear;
#[cfg(feature = "rocm")]
mod rms_norm;
#[cfg(feature = "rocm")]
mod rope;
#[cfg(feature = "rocm")]
mod storage;

#[cfg(feature = "rocm")]
pub use context::{HipContext, HipMemoryInfo};
#[cfg(feature = "rocm")]
pub use error::HipError;
#[cfg(feature = "rocm")]
pub use fused_add_rms_norm::fused_add_rms_norm_f16;
#[cfg(feature = "rocm")]
pub use matmul::matmul_f16_f32;
#[cfg(feature = "rocm")]
pub use quantized_linear::quantized_linear_q4_0_f16;
#[cfg(feature = "rocm")]
pub use rms_norm::rms_norm_f16;
#[cfg(feature = "rocm")]
pub use rope::rope_f16;

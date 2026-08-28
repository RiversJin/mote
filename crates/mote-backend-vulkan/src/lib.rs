//! Vulkan specialization backend.
//!
//! Intended escape hatch: raw SPIR-V/Vulkan compute kernels where the portable
//! path leaves measurable performance or hardware features on the table.

mod context;
mod error;
mod storage;
pub mod vector_add;

pub use context::{VulkanContext, VulkanContextOptions, VulkanMemoryInfo, VulkanMemoryMode};
pub use error::VulkanError;

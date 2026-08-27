//! Portable CubeCL-backed kernels.
//!
//! CubeCL integration intentionally starts behind Mote's kernel interface so
//! backend-specific implementations can replace individual hotspots later.

use mote_kernel::{KernelArgs, KernelError, KernelImpl, KernelKey};
use mote_types::Device;

pub struct PortableKernel {
    name: &'static str,
}

impl PortableKernel {
    pub const fn new(name: &'static str) -> Self {
        Self { name }
    }
}

impl KernelImpl for PortableKernel {
    fn name(&self) -> &'static str {
        self.name
    }

    fn supports(&self, _device: &Device, _key: &KernelKey) -> bool {
        true
    }

    fn launch(&self, _device: &Device, _args: &KernelArgs<'_>) -> Result<(), KernelError> {
        Err(KernelError::Launch(
            "CubeCL integration has not been wired yet".into(),
        ))
    }
}

//! Mote execution runtime.
//!
//! This crate composes core tensor/storage types with kernel dispatch. Keeping
//! this orchestration layer separate prevents `mote-core` and `mote-kernel`
//! from depending on each other.

use mote_kernel::{KernelImpl, KernelKey, KernelRegistry};
use mote_types::Device;

pub struct Runtime {
    kernels: KernelRegistry,
}

impl Runtime {
    pub fn new() -> Self {
        Self {
            kernels: KernelRegistry::default(),
        }
    }

    pub fn kernels(&self) -> &KernelRegistry {
        &self.kernels
    }

    pub fn kernels_mut(&mut self) -> &mut KernelRegistry {
        &mut self.kernels
    }

    pub fn resolve(&self, key: &KernelKey, device: &Device) -> Option<&dyn KernelImpl> {
        self.kernels.resolve(key, device)
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

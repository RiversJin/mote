//! Core runtime types for Mote.

use mote_kernel::{KernelKey, KernelRegistry};

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

    pub fn resolve(&self, key: &KernelKey) -> Option<&dyn mote_kernel::KernelImpl> {
        self.kernels.resolve(key)
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

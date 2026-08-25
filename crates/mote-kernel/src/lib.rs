//! Kernel registry and backend escape-hatch interfaces.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    Cube,
    Cuda,
    Hip,
    Vulkan,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelKey {
    pub op: &'static str,
    pub backend: Backend,
    pub arch: Option<String>,
    pub dtype: &'static str,
    pub shape_class: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub backend: Backend,
    pub arch: Option<String>,
    pub name: String,
}

#[derive(Debug)]
pub struct KernelArgs<'a> {
    pub opaque: &'a [usize],
}

#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error("kernel is not supported on this device")]
    Unsupported,
    #[error("kernel launch failed: {0}")]
    Launch(String),
}

pub trait KernelImpl: Send + Sync {
    fn name(&self) -> &'static str;
    fn supports(&self, device: &DeviceInfo, key: &KernelKey) -> bool;
    fn launch(&self, device: &DeviceInfo, args: &KernelArgs<'_>) -> Result<(), KernelError>;
}

#[derive(Default)]
pub struct KernelRegistry {
    entries: HashMap<KernelKey, Vec<Box<dyn KernelImpl>>>,
}

impl KernelRegistry {
    pub fn register(&mut self, key: KernelKey, kernel: impl KernelImpl + 'static) {
        self.entries.entry(key).or_default().push(Box::new(kernel));
    }

    pub fn resolve(&self, key: &KernelKey) -> Option<&dyn KernelImpl> {
        self.entries
            .get(key)
            .and_then(|kernels| kernels.first())
            .map(Box::as_ref)
    }
}

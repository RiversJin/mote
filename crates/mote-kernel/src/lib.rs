//! Kernel registry and backend escape-hatch interfaces.

use std::collections::HashMap;

use mote_types::{DType, Device};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelKey {
    pub op: &'static str,
    pub dtype: Option<DType>,
    pub shape_class: Option<String>,
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
    fn supports(&self, device: &Device, key: &KernelKey) -> bool;
    fn launch(&self, device: &Device, args: &KernelArgs<'_>) -> Result<(), KernelError>;
}

#[derive(Default)]
pub struct KernelRegistry {
    entries: HashMap<KernelKey, Vec<Box<dyn KernelImpl>>>,
}

impl KernelRegistry {
    pub fn register(&mut self, key: KernelKey, kernel: impl KernelImpl + 'static) {
        self.entries.entry(key).or_default().push(Box::new(kernel));
    }

    /// Resolve the highest-precedence implementation supported by `device`.
    ///
    /// Later registrations take precedence. This lets a portable implementation
    /// be registered first and backend-specific specializations override it
    /// without changing the semantic kernel key.
    pub fn resolve(&self, key: &KernelKey, device: &Device) -> Option<&dyn KernelImpl> {
        self.entries
            .get(key)?
            .iter()
            .rev()
            .map(Box::as_ref)
            .find(|kernel| kernel.supports(device, key))
    }
}

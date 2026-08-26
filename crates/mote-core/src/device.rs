#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Hip,
    Vulkan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device {
    backend: BackendKind,
    ordinal: u32,
}

impl Device {
    pub const fn new(backend: BackendKind, ordinal: u32) -> Self {
        Self { backend, ordinal }
    }

    pub const fn cpu() -> Self {
        Self::new(BackendKind::Cpu, 0)
    }

    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    pub const fn ordinal(&self) -> u32 {
        self.ordinal
    }
}

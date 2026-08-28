use std::{any::Any, marker::PhantomData};

use cubecl::{Runtime, server::Handle};

use mote_core::StorageImpl;
use mote_types::Device;

#[derive(Debug)]
pub(crate) struct CubeStorage<R: Runtime> {
    device: Device,

    // like a pointer to the device
    handle: Handle,
    size_bytes: usize,
    alignment: usize,
    _runtime: PhantomData<R>,
}

impl<R: Runtime> CubeStorage<R> {
    pub(crate) fn new(device: Device, handle: Handle, size_bytes: usize, alignment: usize) -> Self {
        Self {
            device,
            handle,
            size_bytes,
            alignment,
            _runtime: PhantomData,
        }
    }

    pub(crate) fn handle(&self) -> Handle {
        self.handle.clone()
    }
}

impl<R: Runtime> StorageImpl for CubeStorage<R> {
    fn device(&self) -> &Device {
        &self.device
    }
    fn alignment(&self) -> usize {
        self.alignment
    }
    fn size_bytes(&self) -> usize {
        self.size_bytes
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

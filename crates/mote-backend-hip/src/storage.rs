use std::{any::Any, ffi::c_void, sync::Arc};

use mote_core::StorageImpl;
use mote_types::Device;

use crate::{HipError, context::HipContextInner, error::check_hip, ffi::hip};

pub(crate) struct HipStorage {
    pub(crate) context: Arc<HipContextInner>,
    pointer: usize,
    size_bytes: usize,
}

impl HipStorage {
    pub(crate) fn new(
        context: Arc<HipContextInner>,
        size_bytes: usize,
        initial_bytes: Option<&[u8]>,
    ) -> Result<Self, HipError> {
        check_hip(
            hip::set_device(context.ordinal),
            "select device for allocation",
        )?;
        let mut pointer = std::ptr::null_mut();
        check_hip(
            hip::malloc(&mut pointer, size_bytes.max(1)),
            "allocate device storage",
        )?;

        if let Some(bytes) = initial_bytes.filter(|bytes| !bytes.is_empty())
            && let Err(error) = check_hip(
                hip::memcpy_host_to_device(pointer, bytes.as_ptr().cast(), bytes.len()),
                "upload device storage",
            )
        {
            let _ = hip::free(pointer);
            return Err(error);
        }

        Ok(Self {
            context,
            pointer: pointer as usize,
            size_bytes,
        })
    }

    pub(crate) fn pointer(&self) -> *mut c_void {
        self.pointer as *mut c_void
    }

    pub(crate) fn read_bytes(&self) -> Result<Vec<u8>, HipError> {
        if self.size_bytes == 0 {
            return Ok(Vec::new());
        }
        check_hip(
            hip::set_device(self.context.ordinal),
            "select device for readback",
        )?;
        check_hip(
            hip::stream_synchronize(self.context.stream()),
            "synchronize before readback",
        )?;

        let mut bytes = vec![0_u8; self.size_bytes];
        check_hip(
            hip::memcpy_device_to_host(bytes.as_mut_ptr().cast(), self.pointer(), self.size_bytes),
            "read device storage",
        )?;
        Ok(bytes)
    }
}

impl Drop for HipStorage {
    fn drop(&mut self) {
        let _ = hip::set_device(self.context.ordinal);
        let _ = hip::free(self.pointer());
    }
}

impl StorageImpl for HipStorage {
    fn device(&self) -> &Device {
        &self.context.mote_device
    }

    fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    fn alignment(&self) -> usize {
        256
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

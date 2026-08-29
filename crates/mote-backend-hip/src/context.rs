use std::{
    ffi::c_void,
    sync::{Arc, Mutex},
};

use mote_core::{Device as MoteDevice, Storage, Tensor};
use mote_types::{BackendKind, TensorDesc};

use crate::{HipError, error::check_hip, ffi::HipBlasLt, ffi::hip, storage::HipStorage};

pub(crate) const WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

#[derive(Clone)]
pub struct HipContext {
    pub(crate) inner: Arc<HipContextInner>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipMemoryInfo {
    pub free_bytes: usize,
    pub total_bytes: usize,
}

pub(crate) struct HipContextInner {
    pub(crate) mote_device: MoteDevice,
    pub(crate) ordinal: i32,
    stream: hip::Stream,
    blas: Mutex<Option<HipBlasState>>,
}

pub(crate) struct HipBlasState {
    pub(crate) handle: HipBlasLt,
    pub(crate) workspace: usize,
}

impl HipBlasState {
    fn new(ordinal: i32) -> Result<Self, HipError> {
        check_hip(hip::set_device(ordinal), "set device for hipBLASLt")?;
        let handle = HipBlasLt::new()?;
        let mut workspace = std::ptr::null_mut();
        check_hip(
            hip::malloc(&mut workspace, WORKSPACE_BYTES),
            "allocate hipBLASLt workspace",
        )?;
        Ok(Self {
            handle,
            workspace: workspace as usize,
        })
    }
}

impl Drop for HipBlasState {
    fn drop(&mut self) {
        let _ = hip::free(self.workspace as *mut c_void);
    }
}

impl HipContext {
    pub fn new(ordinal: u32) -> Result<Self, HipError> {
        let ordinal_i32 =
            i32::try_from(ordinal).map_err(|_| HipError::DeviceOrdinalTooLarge { ordinal })?;
        let mut available = 0;
        check_hip(hip::get_device_count(&mut available), "enumerate devices")?;
        if ordinal_i32 >= available {
            return Err(HipError::DeviceUnavailable { ordinal, available });
        }

        check_hip(hip::set_device(ordinal_i32), "select device")?;
        let mut stream = hip::Stream::null();
        check_hip(hip::stream_create(&mut stream), "create stream")?;

        Ok(Self {
            inner: Arc::new(HipContextInner {
                mote_device: MoteDevice::new(BackendKind::Hip, ordinal),
                ordinal: ordinal_i32,
                stream,
                blas: Mutex::new(None),
            }),
        })
    }

    pub fn device(&self) -> &MoteDevice {
        &self.inner.mote_device
    }

    pub fn empty(&self, desc: TensorDesc) -> Result<Tensor, HipError> {
        self.validate_desc(&desc)?;
        let implementation = HipStorage::new(self.inner.clone(), desc.required_span_bytes(), None)?;
        let storage = Storage::new(implementation)?;
        Ok(Tensor::new(desc, storage, 0)?)
    }

    pub fn from_bytes(&self, desc: TensorDesc, bytes: &[u8]) -> Result<Tensor, HipError> {
        self.validate_desc(&desc)?;
        let expected = desc.required_span_bytes();
        if bytes.len() != expected {
            return Err(HipError::ByteLengthMismatch {
                expected,
                actual: bytes.len(),
            });
        }

        let implementation = HipStorage::new(self.inner.clone(), expected, Some(bytes))?;
        let storage = Storage::new(implementation)?;
        Ok(Tensor::new(desc, storage, 0)?)
    }

    pub fn read_bytes(&self, tensor: &Tensor) -> Result<Vec<u8>, HipError> {
        self.storage(tensor)?.read_bytes()
    }

    pub fn sync(&self) -> Result<(), HipError> {
        check_hip(
            hip::stream_synchronize(self.inner.stream()),
            "synchronize stream",
        )
    }

    /// Returns the current free and total memory for this context's device.
    pub fn memory_info(&self) -> Result<HipMemoryInfo, HipError> {
        check_hip(
            hip::set_device(self.inner.ordinal),
            "select device for memory info",
        )?;
        let mut free_bytes = 0;
        let mut total_bytes = 0;
        check_hip(
            hip::mem_get_info(&mut free_bytes, &mut total_bytes),
            "query device memory info",
        )?;
        Ok(HipMemoryInfo {
            free_bytes,
            total_bytes,
        })
    }

    pub fn workspace_size_bytes(&self) -> usize {
        WORKSPACE_BYTES
    }

    fn validate_desc(&self, desc: &TensorDesc) -> Result<(), HipError> {
        if !desc.is_contiguous() {
            return Err(HipError::UnsupportedLayout {
                actual: desc.layout().clone(),
            });
        }
        Ok(())
    }

    pub(crate) fn storage<'a>(&self, tensor: &'a Tensor) -> Result<&'a HipStorage, HipError> {
        if tensor.device() != self.device() {
            return Err(HipError::DeviceMismatch {
                expected: *self.device(),
                actual: *tensor.device(),
            });
        }
        if tensor.byte_offset() != 0 {
            return Err(HipError::UnsupportedByteOffset {
                byte_offset: tensor.byte_offset(),
            });
        }
        if !tensor.desc().is_contiguous() {
            return Err(HipError::UnsupportedLayout {
                actual: tensor.desc().layout().clone(),
            });
        }

        let storage = tensor
            .storage()
            .downcast_ref::<HipStorage>()
            .ok_or(HipError::WrongStorage)?;
        if !Arc::ptr_eq(&storage.context, &self.inner) {
            return Err(HipError::WrongStorage);
        }
        Ok(storage)
    }
}

impl HipContextInner {
    pub(crate) fn stream(&self) -> hip::Stream {
        self.stream
    }

    pub(crate) fn with_blas<T>(
        &self,
        callback: impl FnOnce(&mut HipBlasState) -> Result<T, HipError>,
    ) -> Result<T, HipError> {
        let mut state = self.blas.lock().map_err(|_| HipError::StatePoisoned)?;
        if state.is_none() {
            *state = Some(HipBlasState::new(self.ordinal)?);
        }
        callback(state.as_mut().expect("hipBLASLt state was initialized"))
    }
}

impl Drop for HipContextInner {
    fn drop(&mut self) {
        let _ = hip::set_device(self.ordinal);
        let stream = self.stream();
        let _ = hip::stream_synchronize(stream);
        if let Ok(state) = self.blas.get_mut() {
            drop(state.take());
        }
        let _ = hip::stream_destroy(stream);
    }
}

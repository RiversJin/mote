use std::ffi::c_void;

use cubecl::{Runtime, hip::HipRuntime, server::ComputeServer};
use cubecl_hip_sys::hipStream_t;
use cubecl_runtime::{
    native::{NativeError, NativeResource},
    storage::{ComputeStorage, ManagedResource},
};
use mote_core::Tensor;

use crate::{CubeContext, CubeError};

type HipGpuResource = <<HipRuntime as cubecl::Runtime>::Server as ComputeServer>::Storage;
type HipRawResource = <HipGpuResource as ComputeStorage>::Resource;
pub type HipNativeResource = NativeResource<<HipRuntime as Runtime>::Server>;

/// A leased view of CubeCL-managed HIP device memory.
///
/// The allocation remains alive until this value is dropped. Native HIP users
/// must still synchronize their work before dropping it or handing control
/// back to CubeCL.
#[derive(Debug)]
pub struct HipDeviceSlice {
    resource: ManagedResource<HipRawResource>,
}

impl HipDeviceSlice {
    /// Raw HIP device pointer at the tensor's byte offset.
    pub fn as_mut_ptr(&self) -> *mut c_void {
        self.resource.resource().ptr
    }

    /// Number of bytes belonging to this tensor view.
    pub fn len_bytes(&self) -> usize {
        self.resource.resource().size as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len_bytes() == 0
    }
}

impl CubeContext<HipRuntime> {
    /// Enqueue native HIP work on the physical stream managed by CubeCL.
    ///
    /// `callback` borrows the stream and resources only for the duration of
    /// submission. It must enqueue all work on the supplied stream and must
    /// not retain any borrowed native handle after returning.
    pub fn submit_hip_native<F>(&self, tensors: &[&Tensor], callback: F) -> Result<(), CubeError>
    where
        F: FnOnce(&hipStream_t, &[HipNativeResource]) -> Result<(), NativeError> + Send + 'static,
    {
        let handles = tensors
            .iter()
            .map(|tensor| self.handle(tensor))
            .collect::<Result<Vec<_>, _>>()?;

        self.client()
            .submit_native(handles, callback)
            .map_err(|source| CubeError::NativeSubmission { source })
    }

    /// Export a CubeCL allocation for use by native HIP libraries.
    ///
    /// Call [`CubeContext::sync`] before native HIP accesses the pointer, and
    /// synchronize the native HIP stream before CubeCL uses the tensor again.
    pub fn export_hip_device_slice(&self, tensor: &Tensor) -> Result<HipDeviceSlice, CubeError> {
        let handle = self.handle(tensor)?;
        let resource = self
            .client()
            .get_resource(handle)
            .map_err(|source| CubeError::ResourceExport { source })?;

        Ok(HipDeviceSlice { resource })
    }
}

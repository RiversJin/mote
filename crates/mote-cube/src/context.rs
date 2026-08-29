use std::any::type_name;

use cubecl::{client::ComputeClient, server::Handle};
use mote_core::{Device, Storage, StorageError, Tensor};
use mote_types::{Encoding, TensorDesc};

use crate::{CubeError, storage::CubeStorage};

pub struct CubeContext<R: cubecl::Runtime> {
    device: mote_types::Device,
    client: cubecl::prelude::ComputeClient<R>,
    alignment: usize,
}

impl<R: cubecl::Runtime> CubeContext<R> {
    pub fn new(
        device: mote_types::Device,
        client: cubecl::prelude::ComputeClient<R>,
    ) -> Result<Self, CubeError> {
        let raw_align = client.properties().memory.alignment;
        let align = usize::try_from(raw_align)
            .map_err(|_| CubeError::StorageAlignTooLarge { align: raw_align })?;

        if !align.is_power_of_two() {
            return Err(StorageError::InvalidAlignment { alignment: align }.into());
        }

        Ok(Self {
            device,
            client,
            alignment: align,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Whether this runtime reports at least one cooperative-matrix configuration.
    pub fn supports_cooperative_matrix(&self) -> bool {
        !self.client.properties().features.matmul.cmma.is_empty()
    }

    /// Number of cooperative-matrix element/shape combinations reported by the runtime.
    pub fn cooperative_matrix_config_count(&self) -> usize {
        self.client.properties().features.matmul.cmma.len()
    }

    pub fn empty(&self, desc: TensorDesc) -> Result<Tensor, CubeError> {
        self.validate_desc(&desc)?;
        let size_bytes = desc.required_span_bytes();
        let handle = self.client().empty(size_bytes);

        self.tensor_from_handle(desc, handle, size_bytes)
    }

    pub fn from_bytes(&self, desc: TensorDesc, bytes: &[u8]) -> Result<Tensor, CubeError> {
        self.validate_desc(&desc)?;

        let expected_size = desc.required_span_bytes();
        if bytes.len() != expected_size {
            return Err(CubeError::ByteLengthMismatch {
                expected: expected_size,
                actual: bytes.len(),
            });
        }

        let handle = self.client().create_from_slice(bytes);
        self.tensor_from_handle(desc, handle, expected_size)
    }

    pub fn read_bytes(&self, tensor: &Tensor) -> Result<Vec<u8>, CubeError> {
        let handle = self.handle(tensor)?;
        let bytes = self
            .client()
            .read_one(handle)
            .map_err(|source| CubeError::Readback { source })?;
        Ok(bytes.to_vec())
    }

    /// Wait until all work submitted through this context has completed.
    pub fn sync(&self) -> Result<(), CubeError> {
        cubecl::future::block_on(self.client.sync())
            .map_err(|source| CubeError::Synchronization { source })
    }

    fn validate_desc(&self, desc: &TensorDesc) -> Result<(), CubeError> {
        if matches!(desc.encoding(), Encoding::Quantized(_)) {
            return Err(CubeError::UnsupportedEncoding {
                actual: *desc.encoding(),
            });
        }

        if !desc.is_contiguous() {
            return Err(CubeError::UnsupportedLayout {
                actual: desc.layout().clone(),
            });
        }

        Ok(())
    }

    pub(crate) fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    pub(crate) fn handle(&self, tensor: &Tensor) -> Result<Handle, CubeError> {
        if tensor.device() != &self.device {
            return Err(CubeError::DeviceMismatch {
                expected: self.device,
                actual: *tensor.device(),
            });
        }

        if tensor.byte_offset() != 0 {
            return Err(CubeError::UnsupportedByteOffset {
                byte_offset: tensor.byte_offset(),
            });
        }

        if !tensor.desc().is_contiguous() {
            return Err(CubeError::UnsupportedLayout {
                actual: tensor.desc().layout().clone(),
            });
        }

        let storage = tensor.storage().downcast_ref::<CubeStorage<R>>().ok_or(
            CubeError::WrongStorageRuntime {
                expected_runtime: type_name::<R>(),
            },
        )?;

        Ok(storage.handle())
    }

    fn tensor_from_handle(
        &self,
        desc: TensorDesc,
        handle: Handle,
        size_bytes: usize,
    ) -> Result<Tensor, CubeError> {
        let implementation = CubeStorage::<R>::new(self.device, handle, size_bytes, self.alignment);
        let storage = Storage::new(implementation)?;
        Ok(Tensor::new(desc, storage, 0)?)
    }
}

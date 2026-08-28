use std::{any::Any, sync::Arc};

use mote_core::StorageImpl;
use mote_types::Device;
use vulkano::{
    DeviceSize,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::{
        MemoryPropertyFlags,
        allocator::{AllocationCreateInfo, MemoryTypeFilter},
    },
};

use crate::{
    VulkanError,
    context::{VulkanContextInner, VulkanMemoryMode},
};

pub(crate) struct VulkanStorage {
    pub(crate) context: Arc<VulkanContextInner>,
    pub(crate) buffer: Subbuffer<[u8]>,
    pub(crate) size_bytes: usize,
}

impl VulkanStorage {
    pub(crate) fn new(
        context: Arc<VulkanContextInner>,
        size_bytes: usize,
        initial_bytes: Option<&[u8]>,
    ) -> Result<Self, VulkanError> {
        let allocation_size = size_bytes.max(4);
        let allocation_size = DeviceSize::try_from(allocation_size)
            .map_err(|_| VulkanError::StorageSizeTooLarge { size_bytes })?;
        let (usage, memory_type_filter) = match context.memory_mode {
            VulkanMemoryMode::DeviceLocal => (
                BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                MemoryTypeFilter {
                    required_flags: MemoryPropertyFlags::DEVICE_LOCAL,
                    ..Default::default()
                },
            ),
            VulkanMemoryMode::HostVisible => (
                BufferUsage::STORAGE_BUFFER,
                MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ),
        };
        let buffer = Buffer::new_slice::<u8>(
            context.memory_allocator.clone(),
            BufferCreateInfo {
                usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter,
                ..Default::default()
            },
            allocation_size,
        )
        .map_err(|error| VulkanError::backend("storage-buffer allocation", error))?;

        if let Some(initial_bytes) = initial_bytes.filter(|bytes| !bytes.is_empty()) {
            match context.memory_mode {
                VulkanMemoryMode::DeviceLocal => {
                    let staging = host_buffer(
                        &context,
                        initial_bytes.len(),
                        BufferUsage::TRANSFER_SRC,
                        MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    )?;
                    staging
                        .write()
                        .map_err(|error| VulkanError::backend("upload staging mapping", error))?
                        .copy_from_slice(initial_bytes);
                    context.copy_buffer(staging, buffer.clone())?;
                }
                VulkanMemoryMode::HostVisible => {
                    let mut mapped = buffer.write().map_err(|error| {
                        VulkanError::backend("buffer mapping for upload", error)
                    })?;
                    mapped[..size_bytes].copy_from_slice(initial_bytes);
                }
            }
        }

        Ok(Self {
            context,
            buffer,
            size_bytes,
        })
    }

    pub(crate) fn read_bytes(&self) -> Result<Vec<u8>, VulkanError> {
        if self.size_bytes == 0 {
            return Ok(Vec::new());
        }

        match self.context.memory_mode {
            VulkanMemoryMode::DeviceLocal => {
                let readback = host_buffer(
                    &self.context,
                    self.size_bytes,
                    BufferUsage::TRANSFER_DST,
                    MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                )?;
                self.context
                    .copy_buffer(self.buffer.clone(), readback.clone())?;
                let bytes = readback
                    .read()
                    .map_err(|error| VulkanError::backend("readback staging mapping", error))?;
                Ok(bytes.to_vec())
            }
            VulkanMemoryMode::HostVisible => {
                let bytes = self
                    .buffer
                    .read()
                    .map_err(|error| VulkanError::backend("buffer mapping for readback", error))?;
                Ok(bytes[..self.size_bytes].to_vec())
            }
        }
    }
}

fn host_buffer(
    context: &VulkanContextInner,
    size_bytes: usize,
    usage: BufferUsage,
    memory_type_filter: MemoryTypeFilter,
) -> Result<Subbuffer<[u8]>, VulkanError> {
    let size = DeviceSize::try_from(size_bytes)
        .map_err(|_| VulkanError::StorageSizeTooLarge { size_bytes })?;
    Buffer::new_slice::<u8>(
        context.memory_allocator.clone(),
        BufferCreateInfo {
            usage,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter,
            ..Default::default()
        },
        size,
    )
    .map_err(|error| VulkanError::backend("staging-buffer allocation", error))
}

impl StorageImpl for VulkanStorage {
    fn device(&self) -> &Device {
        &self.context.mote_device
    }

    fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    fn alignment(&self) -> usize {
        4
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

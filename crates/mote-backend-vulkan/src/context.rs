use std::sync::{Arc, Mutex};

use mote_core::{Device as MoteDevice, Storage, Tensor};
use mote_types::{BackendKind, Encoding, TensorDesc};
use vulkano::{
    VulkanLibrary,
    buffer::{BufferMemory, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        Device as VkDevice, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue,
        QueueCreateInfo, QueueFlags, physical::PhysicalDeviceType,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::{MemoryPropertyFlags, allocator::StandardMemoryAllocator},
    pipeline::ComputePipeline,
    shader::ShaderStages,
    sync::{self, GpuFuture},
};

use crate::{VulkanError, storage::VulkanStorage};

#[derive(Clone)]
pub struct VulkanContext {
    pub(crate) inner: Arc<VulkanContextInner>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VulkanMemoryMode {
    #[default]
    DeviceLocal,
    HostVisible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VulkanContextOptions {
    pub memory_mode: VulkanMemoryMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VulkanMemoryInfo {
    pub memory_type_index: u32,
    pub heap_index: u32,
    pub heap_size_bytes: u64,
    pub device_local: bool,
    pub host_visible: bool,
    pub host_coherent: bool,
    pub host_cached: bool,
}

pub(crate) struct VulkanContextInner {
    pub(crate) mote_device: MoteDevice,
    pub(crate) device: Arc<VkDevice>,
    pub(crate) queue: Arc<Queue>,
    pub(crate) memory_allocator: Arc<StandardMemoryAllocator>,
    pub(crate) command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub(crate) descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub(crate) vector_add_pipeline: Mutex<Option<Arc<ComputePipeline>>>,
    pub(crate) matmul_pipeline: Mutex<Option<Arc<ComputePipeline>>>,
    pub(crate) matmul_cmma_pipeline: Mutex<Option<Arc<ComputePipeline>>>,
    pub(crate) memory_mode: VulkanMemoryMode,
    cooperative_matrix_supported: bool,
    device_name: String,
}

impl VulkanContext {
    pub fn new(ordinal: u32) -> Result<Self, VulkanError> {
        Self::with_options(ordinal, VulkanContextOptions::default())
    }

    pub fn with_options(ordinal: u32, options: VulkanContextOptions) -> Result<Self, VulkanError> {
        let library = VulkanLibrary::new()
            .map_err(|error| VulkanError::backend("loader initialization", error))?;
        let instance = Instance::new(library, InstanceCreateInfo::default())
            .map_err(|error| VulkanError::backend("instance creation", error))?;

        let mut candidates = instance
            .enumerate_physical_devices()
            .map_err(|error| VulkanError::backend("physical-device enumeration", error))?
            .filter_map(|physical_device| {
                let families = physical_device.queue_family_properties();
                let queue_family_index = families
                    .iter()
                    .position(|family| {
                        family.queue_flags.intersects(QueueFlags::COMPUTE)
                            && !family.queue_flags.intersects(QueueFlags::GRAPHICS)
                    })
                    .or_else(|| {
                        families
                            .iter()
                            .position(|family| family.queue_flags.intersects(QueueFlags::COMPUTE))
                    })? as u32;

                Some((physical_device, queue_family_index))
            })
            .collect::<Vec<_>>();

        candidates.sort_by_key(|(physical_device, _)| {
            match physical_device.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            }
        });

        if candidates.is_empty() {
            return Err(VulkanError::NoComputeDevice);
        }

        let available = candidates.len();
        let (physical_device, queue_family_index) = candidates
            .into_iter()
            .nth(ordinal as usize)
            .ok_or(VulkanError::DeviceUnavailable { ordinal, available })?;
        let device_name = physical_device.properties().device_name.clone();
        let supported_extensions = physical_device.supported_extensions();
        let supported_features = physical_device.supported_features();
        let properties = physical_device.properties();
        let cooperative_matrix_supported = supported_extensions.khr_cooperative_matrix
            && supported_features.cooperative_matrix
            && supported_features.shader_float16
            && supported_features.uniform_and_storage_buffer16_bit_access
            && supported_features.vulkan_memory_model
            && supported_features.vulkan_memory_model_device_scope
            && supported_features.subgroup_size_control
            && properties
                .required_subgroup_size_stages
                .is_some_and(|stages| stages.intersects(ShaderStages::COMPUTE))
            && properties.min_subgroup_size.is_some_and(|size| size <= 64)
            && properties.max_subgroup_size.is_some_and(|size| size >= 64);

        let enabled_extensions = DeviceExtensions {
            khr_cooperative_matrix: cooperative_matrix_supported,
            ..DeviceExtensions::empty()
        };
        let enabled_features = DeviceFeatures {
            cooperative_matrix: cooperative_matrix_supported,
            shader_float16: cooperative_matrix_supported,
            subgroup_size_control: cooperative_matrix_supported,
            uniform_and_storage_buffer16_bit_access: cooperative_matrix_supported,
            vulkan_memory_model: cooperative_matrix_supported,
            vulkan_memory_model_device_scope: cooperative_matrix_supported,
            ..DeviceFeatures::empty()
        };

        let (device, mut queues) = VkDevice::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions,
                enabled_features,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .map_err(|error| VulkanError::backend("logical-device creation", error))?;
        let queue = queues.next().ok_or(VulkanError::NoComputeDevice)?;

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        Ok(Self {
            inner: Arc::new(VulkanContextInner {
                mote_device: MoteDevice::new(BackendKind::Vulkan, ordinal),
                device,
                queue,
                memory_allocator,
                command_buffer_allocator,
                descriptor_set_allocator,
                vector_add_pipeline: Mutex::new(None),
                matmul_pipeline: Mutex::new(None),
                matmul_cmma_pipeline: Mutex::new(None),
                memory_mode: options.memory_mode,
                cooperative_matrix_supported,
                device_name,
            }),
        })
    }

    pub fn device(&self) -> &MoteDevice {
        &self.inner.mote_device
    }

    pub fn device_name(&self) -> &str {
        &self.inner.device_name
    }

    pub fn memory_mode(&self) -> VulkanMemoryMode {
        self.inner.memory_mode
    }

    pub fn supports_cooperative_matrix(&self) -> bool {
        self.inner.cooperative_matrix_supported
    }

    pub fn empty(&self, desc: TensorDesc) -> Result<Tensor, VulkanError> {
        self.validate_desc(&desc)?;
        let size_bytes = desc.required_span_bytes();
        let implementation = VulkanStorage::new(self.inner.clone(), size_bytes, None)?;
        let storage = Storage::new(implementation)?;
        Ok(Tensor::new(desc, storage, 0)?)
    }

    pub fn from_bytes(&self, desc: TensorDesc, bytes: &[u8]) -> Result<Tensor, VulkanError> {
        self.validate_desc(&desc)?;

        let expected = desc.required_span_bytes();
        if bytes.len() != expected {
            return Err(VulkanError::ByteLengthMismatch {
                expected,
                actual: bytes.len(),
            });
        }

        let implementation = VulkanStorage::new(self.inner.clone(), expected, Some(bytes))?;
        let storage = Storage::new(implementation)?;
        Ok(Tensor::new(desc, storage, 0)?)
    }

    pub fn read_bytes(&self, tensor: &Tensor) -> Result<Vec<u8>, VulkanError> {
        self.storage(tensor)?.read_bytes()
    }

    pub fn memory_info(&self, tensor: &Tensor) -> Result<VulkanMemoryInfo, VulkanError> {
        let storage = self.storage(tensor)?;
        let BufferMemory::Normal(memory) = storage.buffer.buffer().memory() else {
            return Err(VulkanError::UnsupportedStorageMemory);
        };
        let memory_type_index = memory.device_memory().memory_type_index();
        let properties = self.inner.device.physical_device().memory_properties();
        let memory_type = properties
            .memory_types
            .get(memory_type_index as usize)
            .ok_or(VulkanError::UnsupportedStorageMemory)?;
        let heap = properties
            .memory_heaps
            .get(memory_type.heap_index as usize)
            .ok_or(VulkanError::UnsupportedStorageMemory)?;
        let flags = memory_type.property_flags;

        Ok(VulkanMemoryInfo {
            memory_type_index,
            heap_index: memory_type.heap_index,
            heap_size_bytes: heap.size,
            device_local: flags.intersects(MemoryPropertyFlags::DEVICE_LOCAL),
            host_visible: flags.intersects(MemoryPropertyFlags::HOST_VISIBLE),
            host_coherent: flags.intersects(MemoryPropertyFlags::HOST_COHERENT),
            host_cached: flags.intersects(MemoryPropertyFlags::HOST_CACHED),
        })
    }

    fn validate_desc(&self, desc: &TensorDesc) -> Result<(), VulkanError> {
        if matches!(desc.encoding(), Encoding::Quantized(_)) {
            return Err(VulkanError::UnsupportedEncoding {
                actual: *desc.encoding(),
            });
        }

        if !desc.is_contiguous() {
            return Err(VulkanError::UnsupportedLayout {
                actual: desc.layout().clone(),
            });
        }

        Ok(())
    }

    pub(crate) fn storage<'a>(&self, tensor: &'a Tensor) -> Result<&'a VulkanStorage, VulkanError> {
        if tensor.device() != &self.inner.mote_device {
            return Err(VulkanError::DeviceMismatch {
                expected: self.inner.mote_device,
                actual: *tensor.device(),
            });
        }

        if tensor.byte_offset() != 0 {
            return Err(VulkanError::UnsupportedByteOffset {
                byte_offset: tensor.byte_offset(),
            });
        }

        if !tensor.desc().is_contiguous() {
            return Err(VulkanError::UnsupportedLayout {
                actual: tensor.desc().layout().clone(),
            });
        }

        let storage = tensor
            .storage()
            .downcast_ref::<VulkanStorage>()
            .ok_or(VulkanError::WrongStorage)?;
        if !Arc::ptr_eq(&storage.context, &self.inner) {
            return Err(VulkanError::WrongStorage);
        }

        Ok(storage)
    }
}

impl VulkanContextInner {
    pub(crate) fn copy_buffer(
        &self,
        source: Subbuffer<[u8]>,
        destination: Subbuffer<[u8]>,
    ) -> Result<(), VulkanError> {
        let mut command_buffer = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|error| VulkanError::backend("transfer command-buffer allocation", error))?;
        command_buffer
            .copy_buffer(CopyBufferInfo::buffers(source, destination))
            .map_err(|error| VulkanError::backend("buffer-copy recording", error))?;
        let command_buffer = command_buffer
            .build()
            .map_err(|error| VulkanError::backend("transfer command-buffer build", error))?;

        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .map_err(|error| VulkanError::backend("transfer queue submission", error))?
            .then_signal_fence_and_flush()
            .map_err(|error| VulkanError::backend("transfer queue flush", error))?
            .wait(None)
            .map_err(|error| VulkanError::backend("transfer completion wait", error))?;

        Ok(())
    }
}

use std::sync::Arc;

use mote_core::Tensor;
use mote_types::{DType, Encoding};
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo},
    sync::{self, GpuFuture},
};

use crate::{VulkanContext, VulkanError};

const WORKGROUP_SIZE: usize = 256;
const VECTOR_ADD_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vector_add.spv"));

/// Add two contiguous plain-F32 Mote tensors using a direct Vulkan compute dispatch.
pub fn vector_add(
    context: &VulkanContext,
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
) -> Result<(), VulkanError> {
    validate_f32(lhs, "lhs")?;
    validate_f32(rhs, "rhs")?;
    validate_f32(output, "output")?;
    validate_shape(lhs, rhs, "rhs")?;
    validate_shape(lhs, output, "output")?;

    let lhs = context.storage(lhs)?;
    let rhs = context.storage(rhs)?;
    let output = context.storage(output)?;
    let numel = lhs.size_bytes / size_of::<f32>();
    if numel == 0 {
        return Ok(());
    }

    let group_count = numel.div_ceil(WORKGROUP_SIZE);
    let group_count =
        u32::try_from(group_count).map_err(|_| VulkanError::LaunchGeometryOverflow { numel })?;

    let (shader_words, remainder) = VECTOR_ADD_SPIRV.as_chunks::<{ size_of::<u32>() }>();
    if !remainder.is_empty() {
        return Err(VulkanError::backend(
            "shader loading",
            "SPIR-V byte length is not a multiple of four",
        ));
    }
    let pipeline = vector_add_pipeline(context, shader_words)?;

    let descriptor_layout = pipeline
        .layout()
        .set_layouts()
        .first()
        .cloned()
        .ok_or_else(|| VulkanError::backend("shader reflection", "descriptor set 0 is missing"))?;
    let descriptor_set = DescriptorSet::new(
        context.inner.descriptor_set_allocator.clone(),
        descriptor_layout,
        [
            WriteDescriptorSet::buffer(0, lhs.buffer.clone()),
            WriteDescriptorSet::buffer(1, rhs.buffer.clone()),
            WriteDescriptorSet::buffer(2, output.buffer.clone()),
        ],
        [],
    )
    .map_err(|error| VulkanError::backend("descriptor-set creation", error))?;

    let mut command_buffer = AutoCommandBufferBuilder::primary(
        context.inner.command_buffer_allocator.clone(),
        context.inner.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .map_err(|error| VulkanError::backend("command-buffer allocation", error))?;
    command_buffer
        .bind_pipeline_compute(pipeline.clone())
        .map_err(|error| VulkanError::backend("compute-pipeline binding", error))?
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            descriptor_set,
        )
        .map_err(|error| VulkanError::backend("descriptor-set binding", error))?;
    // SAFETY: the reflected descriptor layout matches all three bound storage
    // buffers, and the shader bounds-checks every invocation against output.
    unsafe { command_buffer.dispatch([group_count, 1, 1]) }
        .map_err(|error| VulkanError::backend("compute dispatch recording", error))?;
    let command_buffer = command_buffer
        .build()
        .map_err(|error| VulkanError::backend("command-buffer build", error))?;

    sync::now(context.inner.device.clone())
        .then_execute(context.inner.queue.clone(), command_buffer)
        .map_err(|error| VulkanError::backend("queue submission", error))?
        .then_signal_fence_and_flush()
        .map_err(|error| VulkanError::backend("queue flush", error))?
        .wait(None)
        .map_err(|error| VulkanError::backend("dispatch completion wait", error))?;

    Ok(())
}

fn vector_add_pipeline(
    context: &VulkanContext,
    shader_words: &[[u8; size_of::<u32>()]],
) -> Result<Arc<ComputePipeline>, VulkanError> {
    let mut cache = context
        .inner
        .vector_add_pipeline
        .lock()
        .map_err(|error| VulkanError::backend("pipeline-cache locking", error))?;
    if let Some(pipeline) = cache.as_ref() {
        return Ok(pipeline.clone());
    }

    let shader_words = shader_words
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();

    // SAFETY: build.rs emits SPIR-V from a Naga-validated WGSL module. The
    // module stays alive through the entry point used to construct the pipeline.
    let shader = unsafe {
        ShaderModule::new(
            context.inner.device.clone(),
            ShaderModuleCreateInfo::new(&shader_words),
        )
    }
    .map_err(|error| VulkanError::backend("shader-module creation", error))?;
    let entry_point = shader.entry_point("main").ok_or_else(|| {
        VulkanError::backend("shader reflection", "entry point `main` is missing")
    })?;
    let stage = PipelineShaderStageCreateInfo::new(entry_point);
    let layout_create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
        .into_pipeline_layout_create_info(context.inner.device.clone())
        .map_err(|error| VulkanError::backend("pipeline-layout reflection", error))?;
    let layout = PipelineLayout::new(context.inner.device.clone(), layout_create_info)
        .map_err(|error| VulkanError::backend("pipeline-layout creation", error))?;
    let pipeline = ComputePipeline::new(
        context.inner.device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .map_err(|error| VulkanError::backend("compute-pipeline creation", error))?;
    *cache = Some(pipeline.clone());

    Ok(pipeline)
}

fn validate_f32(tensor: &Tensor, name: &'static str) -> Result<(), VulkanError> {
    match *tensor.desc().encoding() {
        Encoding::Plain(DType::F32) => Ok(()),
        Encoding::Plain(actual) => Err(VulkanError::DTypeMismatch {
            tensor: name,
            expected: DType::F32,
            actual,
        }),
        actual @ Encoding::Quantized(_) => Err(VulkanError::UnsupportedEncoding { actual }),
    }
}

fn validate_shape(
    expected: &Tensor,
    actual: &Tensor,
    name: &'static str,
) -> Result<(), VulkanError> {
    if actual.desc().shape() != expected.desc().shape() {
        return Err(VulkanError::ShapeMismatch {
            tensor: name,
            expected: expected.desc().shape().clone(),
            actual: actual.desc().shape().clone(),
        });
    }

    Ok(())
}

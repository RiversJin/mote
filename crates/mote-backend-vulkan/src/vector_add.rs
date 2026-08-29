use std::{sync::Arc, time::Duration};

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
    query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType},
    shader::{ShaderModule, ShaderModuleCreateInfo},
    sync::{self, GpuFuture, PipelineStage},
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
    vector_add_batch(context, lhs, rhs, output, 1)
}

/// Dispatch vector-add repeatedly and wait once after the whole batch.
pub fn vector_add_batch(
    context: &VulkanContext,
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
    dispatches: usize,
) -> Result<(), VulkanError> {
    run_vector_add(context, lhs, rhs, output, dispatches, false).map(|_| ())
}

/// Measure a batch of vector-add dispatches using Vulkan timestamp queries.
pub fn profile_vector_add(
    context: &VulkanContext,
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
    dispatches: usize,
) -> Result<Duration, VulkanError> {
    run_vector_add(context, lhs, rhs, output, dispatches, true)
        .map(|duration| duration.unwrap_or_default())
}

fn run_vector_add(
    context: &VulkanContext,
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
    dispatches: usize,
    profile: bool,
) -> Result<Option<Duration>, VulkanError> {
    validate_f32(lhs, "lhs")?;
    validate_f32(rhs, "rhs")?;
    validate_f32(output, "output")?;
    validate_shape(lhs, rhs, "rhs")?;
    validate_shape(lhs, output, "output")?;

    if dispatches == 0 {
        return Ok(profile.then_some(Duration::ZERO));
    }

    let lhs = context.storage(lhs)?;
    let rhs = context.storage(rhs)?;
    let output = context.storage(output)?;
    let numel = lhs.size_bytes / size_of::<f32>();
    if numel == 0 {
        return Ok(profile.then_some(Duration::ZERO));
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

    let timestamp = if profile {
        let physical_device = context.inner.device.physical_device();
        let queue_family = physical_device
            .queue_family_properties()
            .get(context.inner.queue.queue_family_index() as usize)
            .ok_or(VulkanError::TimestampQueriesUnsupported)?;
        let valid_bits = queue_family
            .timestamp_valid_bits
            .ok_or(VulkanError::TimestampQueriesUnsupported)?;
        let pool = QueryPool::new(
            context.inner.device.clone(),
            QueryPoolCreateInfo {
                query_count: 2,
                ..QueryPoolCreateInfo::query_type(QueryType::Timestamp)
            },
        )
        .map_err(|error| VulkanError::backend("timestamp query-pool creation", error))?;
        Some((pool, valid_bits))
    } else {
        None
    };

    let mut command_buffer = AutoCommandBufferBuilder::primary(
        context.inner.command_buffer_allocator.clone(),
        context.inner.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .map_err(|error| VulkanError::backend("command-buffer allocation", error))?;

    if let Some((query_pool, _)) = timestamp.as_ref() {
        // SAFETY: both query slots are unused by any other command buffer.
        unsafe { command_buffer.reset_query_pool(query_pool.clone(), 0..2) }
            .map_err(|error| VulkanError::backend("timestamp query reset", error))?;
        // SAFETY: query 0 was reset above and the compute queue supports timestamps.
        unsafe {
            command_buffer.write_timestamp(query_pool.clone(), 0, PipelineStage::BottomOfPipe)
        }
        .map_err(|error| VulkanError::backend("start timestamp recording", error))?;
    }

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
    for _ in 0..dispatches {
        // SAFETY: the reflected descriptor layout matches all three bound storage
        // buffers, and the shader bounds-checks every invocation against output.
        unsafe { command_buffer.dispatch([group_count, 1, 1]) }
            .map_err(|error| VulkanError::backend("compute dispatch recording", error))?;
    }

    if let Some((query_pool, _)) = timestamp.as_ref() {
        // SAFETY: query 1 was reset above and follows all profiled dispatches.
        unsafe {
            command_buffer.write_timestamp(query_pool.clone(), 1, PipelineStage::BottomOfPipe)
        }
        .map_err(|error| VulkanError::backend("end timestamp recording", error))?;
    }

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

    let Some((query_pool, valid_bits)) = timestamp else {
        return Ok(None);
    };
    let mut ticks = [0_u64; 2];
    let available = query_pool
        .get_results(0..2, &mut ticks, QueryResultFlags::empty())
        .map_err(|error| VulkanError::backend("timestamp query readback", error))?;
    if !available {
        return Err(VulkanError::TimestampResultsUnavailable);
    }

    let mask = if valid_bits == u64::BITS {
        u64::MAX
    } else {
        (1_u64 << valid_bits) - 1
    };
    let elapsed_ticks = ticks[1].wrapping_sub(ticks[0]) & mask;
    let timestamp_period_ns = context
        .inner
        .device
        .physical_device()
        .properties()
        .timestamp_period as f64;
    let elapsed =
        Duration::from_secs_f64(elapsed_ticks as f64 * timestamp_period_ns / 1_000_000_000.0);

    Ok(Some(elapsed))
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

    // SAFETY: build.rs emits SPIR-V from a Slang-compiled compute module. The
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

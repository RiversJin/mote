use std::{sync::Arc, time::Duration};

use mote_core::Tensor;
use mote_types::{DType, Encoding, Shape};
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet,
        layout::{DescriptorSetLayoutBinding, DescriptorType},
    },
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType},
    shader::{ShaderModule, ShaderModuleCreateInfo, ShaderStages},
    sync::{self, GpuFuture, PipelineStage},
};

use crate::{VulkanContext, VulkanError};

const TILE: usize = 16;
const CMMA_COLUMNS: usize = 64;
const MATMUL_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul.spv"));
const MATMUL_CMMA_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_cmma.spv"));

/// Multiply two contiguous rank-2 F32 matrices using a tiled Slang compute shader.
pub fn matmul(
    context: &VulkanContext,
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
) -> Result<(), VulkanError> {
    run_matmul(context, lhs, rhs, output, false).map(|_| ())
}

/// Measure one tiled F32 matrix multiplication using Vulkan timestamp queries.
pub fn profile_matmul(
    context: &VulkanContext,
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
) -> Result<Duration, VulkanError> {
    run_matmul(context, lhs, rhs, output, true).map(|duration| duration.unwrap_or_default())
}

/// Multiply two contiguous rank-2 F16 matrices using Vulkan cooperative
/// matrices with F32 accumulation and output.
pub fn matmul_cmma_f16_f32(
    context: &VulkanContext,
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
) -> Result<(), VulkanError> {
    run_matmul_cmma(context, lhs, rhs, output, false).map(|_| ())
}

/// Measure one F16/F32 cooperative-matrix multiplication using Vulkan
/// timestamp queries.
pub fn profile_matmul_cmma_f16_f32(
    context: &VulkanContext,
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
) -> Result<Duration, VulkanError> {
    run_matmul_cmma(context, lhs, rhs, output, true).map(|duration| duration.unwrap_or_default())
}

fn run_matmul(
    context: &VulkanContext,
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
    profile: bool,
) -> Result<Option<Duration>, VulkanError> {
    validate_f32_matrix(lhs, "lhs")?;
    validate_f32_matrix(rhs, "rhs")?;
    validate_f32_matrix(output, "output")?;
    let (m, n, k) = validate_shapes(lhs, rhs, output)?;

    if output.desc().numel() == 0 {
        return Ok(profile.then_some(Duration::ZERO));
    }

    let dimensions = [
        u32::try_from(m).map_err(|_| VulkanError::MatmulDimensionsTooLarge { m, n, k })?,
        u32::try_from(n).map_err(|_| VulkanError::MatmulDimensionsTooLarge { m, n, k })?,
        u32::try_from(k).map_err(|_| VulkanError::MatmulDimensionsTooLarge { m, n, k })?,
    ];
    let group_count_x = u32::try_from(n.div_ceil(TILE))
        .map_err(|_| VulkanError::MatmulDimensionsTooLarge { m, n, k })?;
    let group_count_y = u32::try_from(m.div_ceil(TILE))
        .map_err(|_| VulkanError::MatmulDimensionsTooLarge { m, n, k })?;

    let pipeline = matmul_pipeline(context)?;
    dispatch_matmul(
        context,
        lhs,
        rhs,
        output,
        dimensions,
        group_count_x,
        group_count_y,
        pipeline,
        profile,
    )
}

fn run_matmul_cmma(
    context: &VulkanContext,
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
    profile: bool,
) -> Result<Option<Duration>, VulkanError> {
    if !context.supports_cooperative_matrix() {
        return Err(VulkanError::CooperativeMatrixUnsupported);
    }

    validate_plain_matrix(lhs, "lhs", DType::F16)?;
    validate_plain_matrix(rhs, "rhs", DType::F16)?;
    validate_f32_matrix(output, "output")?;
    let (m, n, k) = validate_shapes(lhs, rhs, output)?;

    if output.desc().numel() == 0 {
        return Ok(profile.then_some(Duration::ZERO));
    }
    if !m.is_multiple_of(TILE) || !n.is_multiple_of(CMMA_COLUMNS) || !k.is_multiple_of(TILE) {
        return Err(VulkanError::CooperativeMatrixShapeUnsupported { m, n, k });
    }

    let dimensions = [
        u32::try_from(m).map_err(|_| VulkanError::MatmulDimensionsTooLarge { m, n, k })?,
        u32::try_from(n).map_err(|_| VulkanError::MatmulDimensionsTooLarge { m, n, k })?,
        u32::try_from(k).map_err(|_| VulkanError::MatmulDimensionsTooLarge { m, n, k })?,
    ];
    let group_count_x = u32::try_from(n / CMMA_COLUMNS)
        .map_err(|_| VulkanError::MatmulDimensionsTooLarge { m, n, k })?;
    let group_count_y =
        u32::try_from(m / TILE).map_err(|_| VulkanError::MatmulDimensionsTooLarge { m, n, k })?;
    let pipeline = matmul_cmma_pipeline(context)?;
    dispatch_matmul(
        context,
        lhs,
        rhs,
        output,
        dimensions,
        group_count_x,
        group_count_y,
        pipeline,
        profile,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_matmul(
    context: &VulkanContext,
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
    dimensions: [u32; 3],
    group_count_x: u32,
    group_count_y: u32,
    pipeline: Arc<ComputePipeline>,
    profile: bool,
) -> Result<Option<Duration>, VulkanError> {
    let lhs = context.storage(lhs)?;
    let rhs = context.storage(rhs)?;
    let output = context.storage(output)?;
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

    let timestamp = create_timestamp_pool(context, profile)?;
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
        .map_err(|error| VulkanError::backend("descriptor-set binding", error))?
        .push_constants(pipeline.layout().clone(), 0, dimensions)
        .map_err(|error| VulkanError::backend("push-constant binding", error))?;
    // SAFETY: the reflected descriptor layout and push constants match the
    // Slang shader, and every global access is guarded by M/N/K bounds.
    unsafe { command_buffer.dispatch([group_count_x, group_count_y, 1]) }
        .map_err(|error| VulkanError::backend("compute dispatch recording", error))?;

    if let Some((query_pool, _)) = timestamp.as_ref() {
        // SAFETY: query 1 was reset above and follows the profiled dispatch.
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

    timestamp_duration(context, timestamp)
}

fn matmul_pipeline(context: &VulkanContext) -> Result<Arc<ComputePipeline>, VulkanError> {
    cached_matmul_pipeline(context, &context.inner.matmul_pipeline, MATMUL_SPIRV, false)
}

fn matmul_cmma_pipeline(context: &VulkanContext) -> Result<Arc<ComputePipeline>, VulkanError> {
    cached_matmul_pipeline(
        context,
        &context.inner.matmul_cmma_pipeline,
        MATMUL_CMMA_SPIRV,
        true,
    )
}

fn cached_matmul_pipeline(
    context: &VulkanContext,
    cache: &std::sync::Mutex<Option<Arc<ComputePipeline>>>,
    spirv: &[u8],
    explicit_storage_bindings: bool,
) -> Result<Arc<ComputePipeline>, VulkanError> {
    let mut cache = cache
        .lock()
        .map_err(|error| VulkanError::backend("pipeline-cache locking", error))?;
    if let Some(pipeline) = cache.as_ref() {
        return Ok(pipeline.clone());
    }

    let (shader_words, remainder) = spirv.as_chunks::<{ size_of::<u32>() }>();
    if !remainder.is_empty() {
        return Err(VulkanError::backend(
            "shader loading",
            "SPIR-V byte length is not a multiple of four",
        ));
    }
    let shader_words = shader_words
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();

    // SAFETY: build.rs emits SPIR-V from a Slang-compiled compute module.
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
    let mut stage = PipelineShaderStageCreateInfo::new(entry_point);
    if explicit_storage_bindings {
        stage.required_subgroup_size = Some(64);
    }
    let mut descriptor_layouts = PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage]);
    if explicit_storage_bindings {
        let set = descriptor_layouts
            .set_layouts
            .first_mut()
            .expect("reflection always creates descriptor set 0");
        for binding in 0..3 {
            set.bindings.insert(
                binding,
                DescriptorSetLayoutBinding {
                    stages: ShaderStages::COMPUTE,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
                },
            );
        }
    }
    let layout_create_info = descriptor_layouts
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

fn create_timestamp_pool(
    context: &VulkanContext,
    profile: bool,
) -> Result<Option<(Arc<QueryPool>, u32)>, VulkanError> {
    if !profile {
        return Ok(None);
    }

    let queue_family = context
        .inner
        .device
        .physical_device()
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
    Ok(Some((pool, valid_bits)))
}

fn timestamp_duration(
    context: &VulkanContext,
    timestamp: Option<(Arc<QueryPool>, u32)>,
) -> Result<Option<Duration>, VulkanError> {
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
    Ok(Some(Duration::from_secs_f64(
        elapsed_ticks as f64 * timestamp_period_ns / 1_000_000_000.0,
    )))
}

fn validate_f32_matrix(tensor: &Tensor, name: &'static str) -> Result<(), VulkanError> {
    validate_plain_matrix(tensor, name, DType::F32)
}

fn validate_plain_matrix(
    tensor: &Tensor,
    name: &'static str,
    expected: DType,
) -> Result<(), VulkanError> {
    match *tensor.desc().encoding() {
        Encoding::Plain(actual) if actual == expected => {}
        Encoding::Plain(actual) => {
            return Err(VulkanError::DTypeMismatch {
                tensor: name,
                expected,
                actual,
            });
        }
        actual @ Encoding::Quantized(_) => {
            return Err(VulkanError::UnsupportedEncoding { actual });
        }
    }

    let actual = tensor.desc().rank();
    if actual != 2 {
        return Err(VulkanError::RankMismatch {
            tensor: name,
            expected: 2,
            actual,
        });
    }

    Ok(())
}

fn validate_shapes(
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
) -> Result<(usize, usize, usize), VulkanError> {
    let [m, k] = lhs.desc().shape().dims() else {
        unreachable!("rank was validated above")
    };
    let [rhs_k, n] = rhs.desc().shape().dims() else {
        unreachable!("rank was validated above")
    };

    if rhs_k != k {
        return Err(VulkanError::ShapeMismatch {
            tensor: "rhs",
            expected: Shape::new(&[*k, *n]),
            actual: rhs.desc().shape().clone(),
        });
    }

    let expected = Shape::new(&[*m, *n]);
    if output.desc().shape() != &expected {
        return Err(VulkanError::ShapeMismatch {
            tensor: "output",
            expected,
            actual: output.desc().shape().clone(),
        });
    }

    Ok((*m, *n, *k))
}

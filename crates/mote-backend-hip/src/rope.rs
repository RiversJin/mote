use mote_core::Tensor;
use mote_types::{DType, Encoding, RopeLayout, Shape};

use crate::{HipContext, HipError, error::check_hip, ffi};

/// Launch grid limit: `grid.x` is a 32-bit unsigned quantity on every current
/// HIP runtime, and the kernel maps one block to one (token, head) vector.
const MAX_GRID_VECTORS: u64 = u32::MAX as u64;

/// Enqueue an F16 rotary position embedding with an F32 cache on this
/// context's native HIP stream.
///
/// `input` and `output` may have any rank >= 2: the last two dimensions are
/// `heads` and `head_dim` and the leading dimensions are flattened into the
/// token count (rank 2 means a single token). `cos` and `sin` must have the
/// leading dimensions of `input` plus a trailing `rotary_dim / 2` axis and are
/// shared by all heads of a token. The first `rotary_dim` dimensions of each
/// head vector are rotated in F32; any tail (`rotary_dim < head_dim`) passes
/// through unchanged. `rotary_dim` must be even and at most `head_dim`, and
/// `layout` selects the pairing convention. All four tensors must be plain,
/// contiguous, zero-offset, and backed by storage from `context`; `input` and
/// `output` may be the same tensor, which the kernel handles race-free.
/// Fully validated empty tensors (any zero dimension) return without
/// launching.
pub fn rope_f16(
    context: &HipContext,
    input: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    output: &Tensor,
    rotary_dim: usize,
    layout: RopeLayout,
) -> Result<(), HipError> {
    validate_plain(input, "input", DType::F16)?;
    validate_plain(cos, "cos", DType::F32)?;
    validate_plain(sin, "sin", DType::F32)?;
    validate_plain(output, "output", DType::F16)?;

    let input_rank = input.desc().rank();
    if input_rank < 2 {
        return Err(HipError::RankTooSmall {
            tensor: "input",
            minimum: 2,
            actual: input_rank,
        });
    }

    let dims = input.desc().shape().dims();
    let heads = dims[input_rank - 2];
    let head_dim = dims[input_rank - 1];
    if !rotary_dim.is_multiple_of(2) || rotary_dim > head_dim {
        return Err(HipError::InvalidRotaryDim {
            rotary_dim,
            head_dim,
        });
    }

    if output.desc().shape() != input.desc().shape() {
        return Err(HipError::ShapeMismatch {
            tensor: "output",
            expected: input.desc().shape().clone(),
            actual: output.desc().shape().clone(),
        });
    }

    let mut cache_dims = dims[..input_rank - 2].to_vec();
    cache_dims.push(rotary_dim / 2);
    let expected_cache = Shape::new(&cache_dims);
    if cos.desc().shape() != &expected_cache {
        return Err(HipError::ShapeMismatch {
            tensor: "cos",
            expected: expected_cache.clone(),
            actual: cos.desc().shape().clone(),
        });
    }
    if sin.desc().shape() != &expected_cache {
        return Err(HipError::ShapeMismatch {
            tensor: "sin",
            expected: expected_cache,
            actual: sin.desc().shape().clone(),
        });
    }

    // Storage ownership also enforces contiguity and zero offsets before
    // anything dereferences a device pointer.
    let input_storage = context.storage(input)?;
    let cos_storage = context.storage(cos)?;
    let sin_storage = context.storage(sin)?;
    let output_storage = context.storage(output)?;

    if input.desc().numel() == 0 {
        return Ok(());
    }

    let tokens = dims[..input_rank - 2].iter().copied().product::<usize>();
    let dimensions_too_large = || HipError::RopeDimensionsTooLarge {
        tokens,
        heads,
        head_dim,
        rotary_dim,
    };
    let tokens = u64::try_from(tokens).map_err(|_| dimensions_too_large())?;
    let heads = u64::try_from(heads).map_err(|_| dimensions_too_large())?;
    let head_dim = u64::try_from(head_dim).map_err(|_| dimensions_too_large())?;
    let rotary_dim = u64::try_from(rotary_dim).map_err(|_| dimensions_too_large())?;
    let vectors = tokens.checked_mul(heads).ok_or_else(dimensions_too_large)?;
    if vectors > MAX_GRID_VECTORS {
        return Err(dimensions_too_large());
    }

    let layout = match layout {
        RopeLayout::HalfSplit => ffi::ROPE_LAYOUT_HALF_SPLIT,
        RopeLayout::Interleaved => ffi::ROPE_LAYOUT_INTERLEAVED,
    };

    check_hip(
        ffi::rope_f16(
            input_storage.pointer().cast_const(),
            cos_storage.pointer().cast_const(),
            sin_storage.pointer().cast_const(),
            output_storage.pointer(),
            tokens,
            heads,
            head_dim,
            rotary_dim,
            layout,
            context.inner.stream(),
        ),
        "launch RoPE kernel",
    )
}

fn validate_plain(tensor: &Tensor, name: &'static str, expected: DType) -> Result<(), HipError> {
    match *tensor.desc().encoding() {
        Encoding::Plain(actual) if actual == expected => {}
        Encoding::Plain(actual) => {
            return Err(HipError::DTypeMismatch {
                tensor: name,
                expected,
                actual,
            });
        }
        actual @ Encoding::Quantized(_) => {
            return Err(HipError::UnsupportedEncoding { actual });
        }
    }
    Ok(())
}

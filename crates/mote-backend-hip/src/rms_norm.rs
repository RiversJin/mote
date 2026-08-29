use mote_core::Tensor;
use mote_types::{DType, Encoding, Shape};

use crate::{HipContext, HipError, error::check_hip, ffi};

/// Launch grid limit: `grid.x` is a 32-bit unsigned quantity on every current
/// HIP runtime, and the kernel maps one block to one row.
const MAX_GRID_ROWS: u64 = u32::MAX as u64;

/// Enqueue a row-wise F16 RMSNorm with F32 accumulation on this context's
/// native HIP stream.
///
/// `input` and `output` may have any rank >= 1: the last dimension is the
/// hidden size normalized per row and the leading dimensions are flattened
/// into the row count. `weight` must be rank 1 with length equal to the hidden
/// size, and `output` must have exactly the same shape as `input`. All three
/// tensors must be plain F16, contiguous, zero-offset, and backed by storage
/// from `context`. Fully validated empty tensors (including `hidden == 0`)
/// return without launching.
pub fn rms_norm_f16(
    context: &HipContext,
    input: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    epsilon: f32,
) -> Result<(), HipError> {
    validate_plain_f16(input, "input")?;
    validate_plain_f16(weight, "weight")?;
    validate_plain_f16(output, "output")?;

    let input_rank = input.desc().rank();
    if input_rank < 1 {
        return Err(HipError::RankTooSmall {
            tensor: "input",
            minimum: 1,
            actual: input_rank,
        });
    }
    let weight_rank = weight.desc().rank();
    if weight_rank != 1 {
        return Err(HipError::RankMismatch {
            tensor: "weight",
            expected: 1,
            actual: weight_rank,
        });
    }

    let hidden = input.desc().shape().dims()[input_rank - 1];
    let [weight_hidden] = weight.desc().shape().dims() else {
        unreachable!("rank was validated");
    };
    if *weight_hidden != hidden {
        return Err(HipError::ShapeMismatch {
            tensor: "weight",
            expected: Shape::new(&[hidden]),
            actual: weight.desc().shape().clone(),
        });
    }
    if output.desc().shape() != input.desc().shape() {
        return Err(HipError::ShapeMismatch {
            tensor: "output",
            expected: input.desc().shape().clone(),
            actual: output.desc().shape().clone(),
        });
    }

    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(HipError::InvalidEpsilon { epsilon });
    }

    let input_storage = context.storage(input)?;
    let weight_storage = context.storage(weight)?;
    let output_storage = context.storage(output)?;

    let numel = input.desc().numel();
    if numel == 0 {
        return Ok(());
    }

    let rows = numel / hidden;
    let rows =
        u64::try_from(rows).map_err(|_| HipError::RmsNormDimensionsTooLarge { rows, hidden })?;
    let hidden = u64::try_from(hidden).map_err(|_| HipError::RmsNormDimensionsTooLarge {
        rows: rows as usize,
        hidden,
    })?;
    if rows > MAX_GRID_ROWS {
        return Err(HipError::RmsNormDimensionsTooLarge {
            rows: rows as usize,
            hidden: hidden as usize,
        });
    }

    check_hip(
        ffi::rms_norm_f16(
            input_storage.pointer().cast_const(),
            weight_storage.pointer().cast_const(),
            output_storage.pointer(),
            rows,
            hidden,
            epsilon,
            context.inner.stream(),
        ),
        "launch RMSNorm kernel",
    )
}

fn validate_plain_f16(tensor: &Tensor, name: &'static str) -> Result<(), HipError> {
    let expected = DType::F16;
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

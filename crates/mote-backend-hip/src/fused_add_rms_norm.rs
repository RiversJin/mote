use mote_core::Tensor;
use mote_types::{DType, Encoding, Shape};

use crate::{HipContext, HipError, error::check_hip, ffi};

/// Launch grid limit: `grid.x` is a 32-bit unsigned quantity on every current
/// HIP runtime, and the kernel maps one block to one row.
const MAX_GRID_ROWS: u64 = u32::MAX as u64;

/// Enqueue a fused F16 residual-add + RMSNorm with F32 accumulation on this
/// context's native HIP stream.
///
/// `input`, `residual`, and `output` may have any rank >= 1: the last
/// dimension is the hidden size processed per row and the leading dimensions
/// are flattened into the row count. For every element the kernel forms the
/// F32 sum `input + residual`, publishes the F16-rounded sums back into
/// `residual` (which is in/out), and writes the normalized row
/// `sum * rsqrt(mean_square(sum) + epsilon) * weight` to `output`. `weight`
/// must be rank 1 with length equal to the hidden size, and `residual` and
/// `output` must have exactly the same shape as `input`.
///
/// All four tensors must be plain F16, contiguous, zero-offset, and backed by
/// storage from `context`; `epsilon` must be finite and non-negative.
/// `output` may be (or share storage with) `input` — the kernel computes the
/// statistics before publishing — and `input` may be (or share storage with)
/// `residual`, making every sum `input + input`. `output` must not share
/// storage with `residual`: one allocation cannot hold both results, and the
/// kernel shims reject that pairing. `weight` must not share storage with
/// `output` or `residual`, since the kernel reads it while writing both.
/// Fully validated empty tensors (including `hidden == 0`) return without
/// launching.
pub fn fused_add_rms_norm_f16(
    context: &HipContext,
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    epsilon: f32,
) -> Result<(), HipError> {
    validate_plain_f16(input, "input")?;
    validate_plain_f16(residual, "residual")?;
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
    if residual.desc().shape() != input.desc().shape() {
        return Err(HipError::ShapeMismatch {
            tensor: "residual",
            expected: input.desc().shape().clone(),
            actual: residual.desc().shape().clone(),
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
    let residual_storage = context.storage(residual)?;
    let weight_storage = context.storage(weight)?;
    let output_storage = context.storage(output)?;

    // Aliasing rules: `output` may share storage with `input` (verified by
    // the differential tests) but never with `residual`, and the read-only
    // `weight` may not share storage with either writable result.
    if output.shares_storage_with(residual) {
        return Err(HipError::AliasedStorage {
            tensor: "output",
            other: "residual",
        });
    }
    if weight.shares_storage_with(output) {
        return Err(HipError::AliasedStorage {
            tensor: "weight",
            other: "output",
        });
    }
    if weight.shares_storage_with(residual) {
        return Err(HipError::AliasedStorage {
            tensor: "weight",
            other: "residual",
        });
    }

    let numel = input.desc().numel();
    if numel == 0 {
        return Ok(());
    }

    let rows = numel / hidden;
    let dimensions_too_large = || HipError::FusedAddRmsNormDimensionsTooLarge { rows, hidden };
    let rows = u64::try_from(rows).map_err(|_| dimensions_too_large())?;
    let hidden = u64::try_from(hidden).map_err(|_| dimensions_too_large())?;
    if rows > MAX_GRID_ROWS {
        return Err(dimensions_too_large());
    }

    check_hip(
        ffi::fused_add_rms_norm_f16(
            input_storage.pointer().cast_const(),
            residual_storage.pointer(),
            weight_storage.pointer().cast_const(),
            output_storage.pointer(),
            rows,
            hidden,
            epsilon,
            context.inner.stream(),
        ),
        "launch fused add + RMSNorm kernel",
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

use mote_core::Tensor;
use mote_types::{DType, Encoding, QuantFormat, Shape};

use crate::{HipContext, HipError, error::check_hip, ffi};

const MAX_GRID_BLOCKS: usize = u32::MAX as usize;

/// Enqueue an F16 activation by GGML Q4_0 weight matrix linear operation.
///
/// `input` is plain F16 `[rows, input_features]`; `weights` is contiguous
/// Q4_0 `[output_features, input_features]`, encoded one complete row at a
/// time; and `output` is plain F16 `[rows, output_features]`. The kernel
/// computes `input * weights^T`, accumulating each output element in F32
/// before rounding it to F16. `input_features` may be zero and otherwise must
/// be a multiple of Q4_0's 32-element block size.
///
/// All tensors must be zero-offset allocations from `context`. `output` must
/// not share storage with either input because blocks execute concurrently.
/// Fully validated empty outputs return without launching.
pub fn quantized_linear_q4_0_f16(
    context: &HipContext,
    input: &Tensor,
    weights: &Tensor,
    output: &Tensor,
) -> Result<(), HipError> {
    validate_plain_f16_matrix(input, "input")?;
    validate_q4_0_matrix(weights)?;
    validate_plain_f16_matrix(output, "output")?;

    let [rows, input_features] = input.desc().shape().dims() else {
        unreachable!("rank was validated")
    };
    let [output_features, weight_input_features] = weights.desc().shape().dims() else {
        unreachable!("rank was validated")
    };
    if weight_input_features != input_features {
        return Err(HipError::ShapeMismatch {
            tensor: "weights",
            expected: Shape::new(&[*output_features, *input_features]),
            actual: weights.desc().shape().clone(),
        });
    }
    let expected_output = Shape::new(&[*rows, *output_features]);
    if output.desc().shape() != &expected_output {
        return Err(HipError::ShapeMismatch {
            tensor: "output",
            expected: expected_output,
            actual: output.desc().shape().clone(),
        });
    }

    let input_storage = context.storage(input)?;
    let weight_storage = context.storage(weights)?;
    let output_storage = context.storage(output)?;
    if output.shares_storage_with(input) {
        return Err(HipError::AliasedStorage {
            tensor: "output",
            other: "input",
        });
    }
    if output.shares_storage_with(weights) {
        return Err(HipError::AliasedStorage {
            tensor: "output",
            other: "weights",
        });
    }

    let dimensions_too_large = || HipError::QuantizedLinearDimensionsTooLarge {
        rows: *rows,
        output_features: *output_features,
        input_features: *input_features,
    };
    let grid_blocks = rows
        .checked_mul(*output_features)
        .ok_or_else(dimensions_too_large)?;
    if grid_blocks > MAX_GRID_BLOCKS {
        return Err(dimensions_too_large());
    }
    let rows = u64::try_from(*rows).map_err(|_| dimensions_too_large())?;
    let output_features = u64::try_from(*output_features).map_err(|_| dimensions_too_large())?;
    let input_features = u64::try_from(*input_features).map_err(|_| dimensions_too_large())?;
    if grid_blocks == 0 {
        return Ok(());
    }

    check_hip(
        ffi::quantized_linear_q4_0_f16(
            input_storage.pointer().cast_const(),
            weight_storage.pointer().cast_const(),
            output_storage.pointer(),
            rows,
            output_features,
            input_features,
            context.inner.stream(),
        ),
        "launch Q4_0 linear kernel",
    )
}

fn validate_plain_f16_matrix(tensor: &Tensor, name: &'static str) -> Result<(), HipError> {
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
    validate_rank_two(tensor, name)
}

fn validate_q4_0_matrix(tensor: &Tensor) -> Result<(), HipError> {
    match *tensor.desc().encoding() {
        Encoding::Quantized(QuantFormat::Q4_0) => {}
        actual => return Err(HipError::UnsupportedEncoding { actual }),
    }
    validate_rank_two(tensor, "weights")
}

fn validate_rank_two(tensor: &Tensor, name: &'static str) -> Result<(), HipError> {
    let actual = tensor.desc().rank();
    if actual != 2 {
        return Err(HipError::RankMismatch {
            tensor: name,
            expected: 2,
            actual,
        });
    }
    Ok(())
}

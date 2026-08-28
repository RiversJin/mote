use cubecl::prelude::*;
use mote_core::Tensor;
use mote_types::{DType, Encoding};

use crate::{CubeContext, CubeError};

const CUBE_DIM: u32 = 256;

/// Add two equally sized vectors element by element.
#[cube(launch)]
pub(crate) fn vector_add_kernel<F: Float>(lhs: &Array<F>, rhs: &Array<F>, output: &mut Array<F>) {
    let index = ABSOLUTE_POS;

    if index < output.len() {
        output[index] = lhs[index] + rhs[index];
    }
}

/// Add two contiguous plain-F32 Mote tensors into an equally shaped output tensor.
pub fn vector_add<R: Runtime>(
    context: &CubeContext<R>,
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
) -> Result<(), CubeError> {
    validate_f32(lhs, "lhs")?;
    validate_f32(rhs, "rhs")?;
    validate_f32(output, "output")?;

    validate_shape(lhs, rhs, "rhs")?;
    validate_shape(lhs, output, "output")?;

    let lhs_handle = context.handle(lhs)?;
    let rhs_handle = context.handle(rhs)?;
    let output_handle = context.handle(output)?;

    let numel = lhs.desc().numel();
    if numel == 0 {
        return Ok(());
    }

    let cube_count = numel.div_ceil(CUBE_DIM as usize);
    let cube_count =
        u32::try_from(cube_count).map_err(|_| CubeError::LaunchGeometryOverflow { numel })?;

    // SAFETY: every handle was recovered from CubeStorage<R> matching this
    // context's runtime and device. Tensor construction guarantees the required
    // byte span, while the checks above guarantee contiguous plain-F32 arrays
    // of equal length.
    unsafe {
        vector_add_kernel::launch::<f32, R>(
            context.client(),
            CubeCount::new_1d(cube_count),
            CubeDim::new_1d(CUBE_DIM),
            ArrayArg::from_raw_parts(lhs_handle, numel),
            ArrayArg::from_raw_parts(rhs_handle, numel),
            ArrayArg::from_raw_parts(output_handle, numel),
        );
    }

    Ok(())
}

fn validate_f32(tensor: &Tensor, name: &'static str) -> Result<(), CubeError> {
    match *tensor.desc().encoding() {
        Encoding::Plain(DType::F32) => Ok(()),
        Encoding::Plain(actual) => Err(CubeError::DTypeMismatch {
            tensor: name,
            expected: DType::F32,
            actual,
        }),
        actual @ Encoding::Quantized(_) => Err(CubeError::UnsupportedEncoding { actual }),
    }
}

fn validate_shape(expected: &Tensor, actual: &Tensor, name: &'static str) -> Result<(), CubeError> {
    if actual.desc().shape() != expected.desc().shape() {
        return Err(CubeError::ShapeMismatch {
            tensor: name,
            expected: expected.desc().shape().clone(),
            actual: actual.desc().shape().clone(),
        });
    }

    Ok(())
}

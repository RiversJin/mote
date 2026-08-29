use mote_core::Tensor;
use mote_types::{DType, Encoding, Shape};

use crate::{HipContext, HipError, context::WORKSPACE_BYTES};

/// Enqueue a row-major F16 x F16 matrix multiplication with F32 accumulation
/// and output on this context's native HIP stream.
///
/// The first call for a new `(M, N, K)` shape tunes and caches a hipBLASLt
/// algorithm. Subsequent calls only enqueue the cached algorithm.
pub fn matmul_f16_f32(
    context: &HipContext,
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
) -> Result<(), HipError> {
    validate_plain_matrix(lhs, "lhs", DType::F16)?;
    validate_plain_matrix(rhs, "rhs", DType::F16)?;
    validate_plain_matrix(output, "output", DType::F32)?;
    let (m, n, k) = validate_shapes(lhs, rhs, output)?;
    if output.desc().numel() == 0 {
        return Ok(());
    }

    let m = u64::try_from(m).map_err(|_| HipError::MatmulDimensionsTooLarge { m, n, k })?;
    let n = u64::try_from(n).map_err(|_| HipError::MatmulDimensionsTooLarge {
        m: m as usize,
        n,
        k,
    })?;
    let k = u64::try_from(k).map_err(|_| HipError::MatmulDimensionsTooLarge {
        m: m as usize,
        n: n as usize,
        k,
    })?;
    let lhs = context.storage(lhs)?.pointer();
    let rhs = context.storage(rhs)?.pointer();
    let output = context.storage(output)?.pointer();

    context.inner.with_blas(|state| {
        state.handle.matmul_f16_f32(
            m,
            n,
            k,
            lhs.cast_const(),
            rhs.cast_const(),
            output,
            state.workspace as *mut _,
            WORKSPACE_BYTES,
            context.inner.stream(),
        )
    })
}

fn validate_plain_matrix(
    tensor: &Tensor,
    name: &'static str,
    expected: DType,
) -> Result<(), HipError> {
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

fn validate_shapes(
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
) -> Result<(usize, usize, usize), HipError> {
    let [m, k] = lhs.desc().shape().dims() else {
        unreachable!("rank was validated")
    };
    let [rhs_k, n] = rhs.desc().shape().dims() else {
        unreachable!("rank was validated")
    };
    if rhs_k != k {
        return Err(HipError::ShapeMismatch {
            tensor: "rhs",
            expected: Shape::new(&[*k, *n]),
            actual: rhs.desc().shape().clone(),
        });
    }
    let expected = Shape::new(&[*m, *n]);
    if output.desc().shape() != &expected {
        return Err(HipError::ShapeMismatch {
            tensor: "output",
            expected,
            actual: output.desc().shape().clone(),
        });
    }
    Ok((*m, *n, *k))
}

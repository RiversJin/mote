use cubecl::{
    prelude::{CubePrimitive, Runtime, TensorBinding},
    zspace::{Shape as CubeShape, Strides},
};
use cubek_matmul::{
    definition::MatmulElems,
    launch::{Strategy, launch_ref},
};
use cubek_std::InputBinding;
use mote_core::Tensor;
use mote_types::{DType, Encoding, Shape};

use crate::{CubeContext, CubeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CubeMatmulMode {
    /// Exact F32 inputs, staging, accumulation and output.
    #[default]
    F32,
    /// F16 inputs with F32 cooperative-matrix accumulation and output.
    CmmaF16F32,
}

/// Multiply two contiguous rank-2 F32 matrices.
pub fn matmul<R: Runtime>(
    context: &CubeContext<R>,
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
) -> Result<(), CubeError> {
    matmul_with_mode(context, lhs, rhs, output, CubeMatmulMode::F32)
}

/// Multiply two F16 matrices through CubeCL cooperative-matrix instructions,
/// accumulating and storing the result as F32.
pub fn matmul_cmma_f16_f32<R: Runtime>(
    context: &CubeContext<R>,
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
) -> Result<(), CubeError> {
    matmul_with_mode(context, lhs, rhs, output, CubeMatmulMode::CmmaF16F32)
}

pub fn matmul_with_mode<R: Runtime>(
    context: &CubeContext<R>,
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
    mode: CubeMatmulMode,
) -> Result<(), CubeError> {
    let input_dtype = match mode {
        CubeMatmulMode::F32 => DType::F32,
        CubeMatmulMode::CmmaF16F32 => DType::F16,
    };
    validate_plain_matrix(lhs, "lhs", input_dtype)?;
    validate_plain_matrix(rhs, "rhs", input_dtype)?;
    validate_f32_matrix(output, "output")?;
    validate_shapes(lhs, rhs, output)?;

    if output.desc().numel() == 0 {
        return Ok(());
    }

    let lhs = tensor_binding(context, lhs)?;
    let rhs = tensor_binding(context, rhs)?;
    let output = tensor_binding(context, output)?;
    let f32_dtype = f32::as_type_native_unchecked().storage_type();
    let f16_dtype = half::f16::as_type_native_unchecked().storage_type();
    let (lhs_dtype, rhs_dtype, mut dtypes) = match mode {
        CubeMatmulMode::F32 => (
            f32_dtype,
            f32_dtype,
            MatmulElems::from_single_dtype(f32::as_type_native_unchecked()),
        ),
        CubeMatmulMode::CmmaF16F32 => (
            f16_dtype,
            f16_dtype,
            MatmulElems {
                lhs_global: f16_dtype,
                rhs_global: f16_dtype,
                acc_global: f32_dtype,
                lhs_stage: f16_dtype,
                rhs_stage: f16_dtype,
                acc_stage: f32_dtype,
                lhs_register: f16_dtype,
                rhs_register: f16_dtype,
                acc_register: f32_dtype,
            },
        ),
    };
    let strategy = match mode {
        CubeMatmulMode::F32 => Strategy::Auto,
        CubeMatmulMode::CmmaF16F32 => Strategy::SimpleCyclicCmma(Default::default()),
    };

    launch_ref(
        &strategy,
        context.client(),
        InputBinding::new(lhs, lhs_dtype),
        InputBinding::new(rhs, rhs_dtype),
        output,
        &mut dtypes,
    )
    .map_err(|error| CubeError::MatmulSetup {
        message: format!("{error:?}"),
    })
}

fn tensor_binding<R: Runtime>(
    context: &CubeContext<R>,
    tensor: &Tensor,
) -> Result<TensorBinding<R>, CubeError> {
    let dims = tensor.desc().shape().dims();
    let shape = dims.iter().copied().collect::<CubeShape>();
    let mut stride = 1;
    let mut strides = vec![0; dims.len()];
    for (axis, dimension) in dims.iter().copied().enumerate().rev() {
        strides[axis] = stride;
        stride *= dimension;
    }

    // SAFETY: the handle belongs to this runtime and context. Mote validated
    // the contiguous allocation against the same shape used to derive strides.
    Ok(unsafe {
        TensorBinding::from_raw_parts(context.handle(tensor)?, Strides::new(&strides), shape)
    })
}

fn validate_f32_matrix(tensor: &Tensor, name: &'static str) -> Result<(), CubeError> {
    validate_plain_matrix(tensor, name, DType::F32)
}

fn validate_plain_matrix(
    tensor: &Tensor,
    name: &'static str,
    expected: DType,
) -> Result<(), CubeError> {
    match *tensor.desc().encoding() {
        Encoding::Plain(actual) if actual == expected => {}
        Encoding::Plain(actual) => {
            return Err(CubeError::DTypeMismatch {
                tensor: name,
                expected,
                actual,
            });
        }
        actual @ Encoding::Quantized(_) => {
            return Err(CubeError::UnsupportedEncoding { actual });
        }
    }

    let actual = tensor.desc().rank();
    if actual != 2 {
        return Err(CubeError::RankMismatch {
            tensor: name,
            expected: 2,
            actual,
        });
    }

    Ok(())
}

fn validate_shapes(lhs: &Tensor, rhs: &Tensor, output: &Tensor) -> Result<(), CubeError> {
    let [m, k] = lhs.desc().shape().dims() else {
        unreachable!("rank was validated above")
    };
    let [rhs_k, n] = rhs.desc().shape().dims() else {
        unreachable!("rank was validated above")
    };

    if rhs_k != k {
        return Err(CubeError::ShapeMismatch {
            tensor: "rhs",
            expected: Shape::new(&[*k, *n]),
            actual: rhs.desc().shape().clone(),
        });
    }

    let expected = Shape::new(&[*m, *n]);
    if output.desc().shape() != &expected {
        return Err(CubeError::ShapeMismatch {
            tensor: "output",
            expected,
            actual: output.desc().shape().clone(),
        });
    }

    Ok(())
}

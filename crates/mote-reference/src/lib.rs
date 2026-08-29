//! Reference ("oracle") implementations of Mote operators.
//!
//! This crate contains plain, dependency-light Rust implementations whose only
//! job is to be obviously correct. They are the ground truth that optimized
//! kernels (CUDA, HIP, Vulkan, ...) are validated against, so clarity is
//! preferred over performance here.

#![forbid(unsafe_code)]

mod quantized;

use thiserror::Error;

use mote_types::RopeLayout;

pub use quantized::{QuantizedLinearError, dequantize_quantized_row, quantized_linear_f32};

/// Errors reported by the reference implementations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ReferenceError {
    /// `hidden_size` was zero.
    #[error("hidden_size must be non-zero")]
    ZeroHiddenSize,

    /// `weight.len()` did not equal `hidden_size`.
    #[error("weight length {weight_len} does not match hidden_size {hidden_size}")]
    WeightLengthMismatch {
        /// Length of the `weight` slice.
        weight_len: usize,
        /// The expected row length.
        hidden_size: usize,
    },

    /// `input.len()` was not an exact multiple of `hidden_size`.
    #[error("input length {input_len} is not a multiple of hidden_size {hidden_size}")]
    InputNotMultiple {
        /// Length of the `input` slice.
        input_len: usize,
        /// The row length `input_len` must be divisible by.
        hidden_size: usize,
    },

    /// `residual.len()` did not equal `input.len()`.
    #[error("residual length {residual_len} does not match input length {input_len}")]
    ResidualLengthMismatch {
        /// Length of the `residual` slice.
        residual_len: usize,
        /// Length of the `input` slice `residual` must match.
        input_len: usize,
    },

    /// `epsilon` was negative or not finite (NaN / infinity).
    #[error("epsilon must be finite and non-negative, got {epsilon}")]
    InvalidEpsilon {
        /// The offending epsilon value.
        epsilon: f32,
    },
}

/// Errors reported by the row-major matrix multiplication oracle.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MatmulError {
    /// A dimension product could not be represented by `usize`.
    #[error("matrix dimensions M={m}, N={n}, K={k} overflow the host address space")]
    DimensionsOverflow { m: usize, n: usize, k: usize },

    /// `lhs.len()` did not equal `m * k`.
    #[error("lhs length {actual} does not match M*K={expected}")]
    LhsLengthMismatch { expected: usize, actual: usize },

    /// `rhs.len()` did not equal `k * n`.
    #[error("rhs length {actual} does not match K*N={expected}")]
    RhsLengthMismatch { expected: usize, actual: usize },
}

/// Errors reported by the rotary position embedding oracle.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RopeError {
    #[error("rotary dimension {rotary_dim} must be even")]
    OddRotaryDimension { rotary_dim: usize },

    #[error("rotary dimension {rotary_dim} exceeds head dimension {head_dim}")]
    RotaryDimensionTooLarge { rotary_dim: usize, head_dim: usize },

    #[error(
        "RoPE dimensions tokens={tokens}, heads={heads}, head_dim={head_dim}, rotary_dim={rotary_dim} overflow the host address space"
    )]
    DimensionsOverflow {
        tokens: usize,
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
    },

    #[error("input length {actual} does not match tokens*heads*head_dim={expected}")]
    InputLengthMismatch { expected: usize, actual: usize },

    #[error("cos length {actual} does not match tokens*(rotary_dim/2)={expected}")]
    CosLengthMismatch { expected: usize, actual: usize },

    #[error("sin length {actual} does not match tokens*(rotary_dim/2)={expected}")]
    SinLengthMismatch { expected: usize, actual: usize },
}

/// Row-major `M x K` by `K x N` matrix multiplication in `f32`.
///
/// This deliberately naive implementation is the correctness oracle for
/// vendor-library GEMM paths. Inputs and accumulation are both `f32`; callers
/// testing lower-precision backends should quantize their inputs before
/// widening them for this function.
///
/// # Errors
///
/// Returns [`MatmulError`] when a dimension product overflows or an input
/// slice does not match its declared matrix dimensions.
pub fn matmul_f32(
    lhs: &[f32],
    rhs: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, MatmulError> {
    let lhs_len = m
        .checked_mul(k)
        .ok_or(MatmulError::DimensionsOverflow { m, n, k })?;
    let rhs_len = k
        .checked_mul(n)
        .ok_or(MatmulError::DimensionsOverflow { m, n, k })?;
    let output_len = m
        .checked_mul(n)
        .ok_or(MatmulError::DimensionsOverflow { m, n, k })?;

    if lhs.len() != lhs_len {
        return Err(MatmulError::LhsLengthMismatch {
            expected: lhs_len,
            actual: lhs.len(),
        });
    }
    if rhs.len() != rhs_len {
        return Err(MatmulError::RhsLengthMismatch {
            expected: rhs_len,
            actual: rhs.len(),
        });
    }

    let mut output = vec![0.0_f32; output_len];
    for row in 0..m {
        for inner in 0..k {
            let lhs_value = lhs[row * k + inner];
            for col in 0..n {
                output[row * n + col] += lhs_value * rhs[inner * n + col];
            }
        }
    }
    Ok(output)
}

/// Applies rotary position embeddings to row-major head vectors in `f32`.
///
/// `input` is shaped `[tokens, heads, head_dim]`. `cos` and `sin` are shaped
/// `[tokens, rotary_dim / 2]` and are shared by all heads of a token. Elements
/// after `rotary_dim` pass through unchanged. The pairing convention is
/// explicit because model families use both half-split and interleaved RoPE.
///
/// # Errors
///
/// Returns [`RopeError`] for invalid rotary geometry, overflowing dimension
/// products, or slice lengths that do not match the declared dimensions.
#[allow(clippy::too_many_arguments)]
pub fn rope_f32(
    input: &[f32],
    cos: &[f32],
    sin: &[f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    layout: RopeLayout,
) -> Result<Vec<f32>, RopeError> {
    if !rotary_dim.is_multiple_of(2) {
        return Err(RopeError::OddRotaryDimension { rotary_dim });
    }
    if rotary_dim > head_dim {
        return Err(RopeError::RotaryDimensionTooLarge {
            rotary_dim,
            head_dim,
        });
    }

    let dimensions = RopeError::DimensionsOverflow {
        tokens,
        heads,
        head_dim,
        rotary_dim,
    };
    let input_len = tokens
        .checked_mul(heads)
        .and_then(|rows| rows.checked_mul(head_dim))
        .ok_or_else(|| dimensions.clone())?;
    let frequency_count = rotary_dim / 2;
    let cache_len = tokens.checked_mul(frequency_count).ok_or(dimensions)?;

    if input.len() != input_len {
        return Err(RopeError::InputLengthMismatch {
            expected: input_len,
            actual: input.len(),
        });
    }
    if cos.len() != cache_len {
        return Err(RopeError::CosLengthMismatch {
            expected: cache_len,
            actual: cos.len(),
        });
    }
    if sin.len() != cache_len {
        return Err(RopeError::SinLengthMismatch {
            expected: cache_len,
            actual: sin.len(),
        });
    }

    let mut output = input.to_vec();
    for token in 0..tokens {
        let cache_offset = token * frequency_count;
        for head in 0..heads {
            let row_offset = (token * heads + head) * head_dim;
            for pair in 0..frequency_count {
                let (first, second) = match layout {
                    RopeLayout::HalfSplit => (pair, pair + frequency_count),
                    RopeLayout::Interleaved => (pair * 2, pair * 2 + 1),
                };
                let first_index = row_offset + first;
                let second_index = row_offset + second;
                let first_value = input[first_index];
                let second_value = input[second_index];
                let cosine = cos[cache_offset + pair];
                let sine = sin[cache_offset + pair];
                output[first_index] = first_value * cosine - second_value * sine;
                output[second_index] = second_value * cosine + first_value * sine;
            }
        }
    }
    Ok(output)
}

/// Root-mean-square layer norm over the last dimension, in `f32`.
///
/// `input` is interpreted as `input.len() / hidden_size` rows of length
/// `hidden_size`. For each row `x` the output is
/// `x * rsqrt(mean_square(x) + epsilon) * weight`, where `mean_square(x)` is
/// the mean of `x[i] * x[i]` accumulated in `f32`.
///
/// The function is non-inplace: it returns a freshly allocated `Vec` and never
/// mutates its inputs. An empty `input` yields an empty `Vec` as long as the
/// other parameters are valid.
///
/// # Errors
///
/// Returns a [`ReferenceError`] if `hidden_size` is zero, `weight.len() !=
/// hidden_size`, `input.len()` is not divisible by `hidden_size`, or `epsilon`
/// is negative or not finite.
pub fn rms_norm_f32(
    input: &[f32],
    weight: &[f32],
    hidden_size: usize,
    epsilon: f32,
) -> Result<Vec<f32>, ReferenceError> {
    if hidden_size == 0 {
        return Err(ReferenceError::ZeroHiddenSize);
    }
    if weight.len() != hidden_size {
        return Err(ReferenceError::WeightLengthMismatch {
            weight_len: weight.len(),
            hidden_size,
        });
    }
    if !input.len().is_multiple_of(hidden_size) {
        return Err(ReferenceError::InputNotMultiple {
            input_len: input.len(),
            hidden_size,
        });
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(ReferenceError::InvalidEpsilon { epsilon });
    }

    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(hidden_size) {
        let mut sum_sq = 0.0f32;
        for &x in row {
            sum_sq += x * x;
        }
        let mean_square = sum_sq / hidden_size as f32;
        // rsqrt(v) expressed as 1 / sqrt(v); f32 keeps this an f32 oracle.
        let inv_rms = 1.0f32 / (mean_square + epsilon).sqrt();
        for (&x, &w) in row.iter().zip(weight) {
            output.push(x * inv_rms * w);
        }
    }
    Ok(output)
}

/// Output of [`fused_add_rms_norm_f32`].
#[derive(Debug, Clone, PartialEq)]
pub struct FusedAddRmsNormOutput {
    /// The normed rows: `sum * rsqrt(mean_square(sum) + epsilon) * weight`.
    pub normalized: Vec<f32>,

    /// The element-wise `input + residual` sums in `f32`, kept for the next
    /// residual consumer in the network.
    pub residual: Vec<f32>,
}

/// Fused residual-add + RMS norm over the last dimension, in `f32`.
///
/// `input` and `residual` are interpreted as `input.len() / hidden_size` rows
/// of length `hidden_size`. For each row the element-wise sum
/// `sum = input + residual` is computed in `f32` and returned as
/// [`FusedAddRmsNormOutput::residual`]. The same **un-quantized** `f32` sums
/// feed the statistics: the output row is
/// `sum * rsqrt(mean_square(sum) + epsilon) * weight`, where `mean_square(sum)`
/// is the mean of `sum[i] * sum[i]` accumulated in `f32`.
///
/// The function is non-inplace: both outputs are freshly allocated and the
/// input slices are never mutated. A valid empty `input` yields two empty
/// `Vec`s.
///
/// # Errors
///
/// Returns a [`ReferenceError`] if `hidden_size` is zero, `weight.len() !=
/// hidden_size`, `residual.len() != input.len()`, `input.len()` is not
/// divisible by `hidden_size`, or `epsilon` is negative or not finite.
pub fn fused_add_rms_norm_f32(
    input: &[f32],
    residual: &[f32],
    weight: &[f32],
    hidden_size: usize,
    epsilon: f32,
) -> Result<FusedAddRmsNormOutput, ReferenceError> {
    if hidden_size == 0 {
        return Err(ReferenceError::ZeroHiddenSize);
    }
    if weight.len() != hidden_size {
        return Err(ReferenceError::WeightLengthMismatch {
            weight_len: weight.len(),
            hidden_size,
        });
    }
    if residual.len() != input.len() {
        return Err(ReferenceError::ResidualLengthMismatch {
            residual_len: residual.len(),
            input_len: input.len(),
        });
    }
    if !input.len().is_multiple_of(hidden_size) {
        return Err(ReferenceError::InputNotMultiple {
            input_len: input.len(),
            hidden_size,
        });
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(ReferenceError::InvalidEpsilon { epsilon });
    }

    let sums: Vec<f32> = input.iter().zip(residual).map(|(&x, &r)| x + r).collect();

    let mut normalized = Vec::with_capacity(sums.len());
    for row in sums.chunks_exact(hidden_size) {
        let mut sum_sq = 0.0f32;
        for &x in row {
            sum_sq += x * x;
        }
        let mean_square = sum_sq / hidden_size as f32;
        // rsqrt(v) expressed as 1 / sqrt(v); f32 keeps this an f32 oracle.
        let inv_rms = 1.0f32 / (mean_square + epsilon).sqrt();
        for (&x, &w) in row.iter().zip(weight) {
            normalized.push(x * inv_rms * w);
        }
    }
    Ok(FusedAddRmsNormOutput {
        normalized,
        residual: sums,
    })
}

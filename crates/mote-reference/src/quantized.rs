use half::f16;
use mote_types::QuantFormat;
use thiserror::Error;

/// Errors reported by the block-quantized reference operators.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum QuantizedLinearError {
    /// The format has geometry in `mote-types`, but this oracle cannot decode it yet.
    #[error("quantized format {format:?} is not supported by the reference linear oracle")]
    UnsupportedFormat { format: QuantFormat },

    /// A row must contain an integral number of format-defined blocks.
    #[error(
        "quantized row length {row_elements} is not a multiple of {format:?}'s {block_elements}-element block"
    )]
    RowMisaligned {
        format: QuantFormat,
        row_elements: usize,
        block_elements: usize,
    },

    /// A dimension product or physical byte count did not fit in `usize`.
    #[error(
        "quantized linear dimensions rows={rows}, output_features={output_features}, input_features={input_features} overflow the host address space"
    )]
    DimensionsOverflow {
        rows: usize,
        output_features: usize,
        input_features: usize,
    },

    /// The plain activation slice did not match `[rows, input_features]`.
    #[error("input length {actual} does not match rows*input_features={expected}")]
    InputLengthMismatch { expected: usize, actual: usize },

    /// The encoded weight bytes did not match
    /// `[output_features, input_features]` in the selected block format.
    #[error("weight byte length {actual} does not match the encoded matrix size {expected}")]
    WeightLengthMismatch { expected: usize, actual: usize },

    /// The encoded row bytes did not match one complete quantized row.
    #[error("row byte length {actual} does not match the encoded row size {expected}")]
    RowByteLengthMismatch { expected: usize, actual: usize },
}

/// Dequantizes one contiguous GGML-compatible row into `f32`.
///
/// `Q4_0` blocks contain a little-endian `f16` scale followed by 16 packed
/// bytes. Byte `j` stores element `j` in its low nibble and element `j + 16`
/// in its high nibble; both use a zero point of 8. `Q8_0` blocks contain the
/// same scale followed by 32 signed bytes. A zero-length row is valid.
///
/// # Errors
///
/// Returns [`QuantizedLinearError`] when `format` is not implemented, the
/// logical row is not block aligned, its physical size overflows, or `bytes`
/// does not contain exactly one encoded row.
pub fn dequantize_quantized_row(
    bytes: &[u8],
    format: QuantFormat,
    row_elements: usize,
) -> Result<Vec<f32>, QuantizedLinearError> {
    let row_bytes = checked_row_bytes(format, row_elements, 1, 1)?;
    if bytes.len() != row_bytes {
        return Err(QuantizedLinearError::RowByteLengthMismatch {
            expected: row_bytes,
            actual: bytes.len(),
        });
    }

    let mut output = vec![0.0; row_elements];
    decode_row_into(bytes, format, &mut output);
    Ok(output)
}

/// Applies a block-quantized row-major weight matrix to plain `f32` rows.
///
/// `input` is `[rows, input_features]`, while `weights` is encoded row by row
/// as `[output_features, input_features]`. The returned row-major matrix is
/// `[rows, output_features]` and computes `output = input * weights^T` with
/// `f32` multiplication and accumulation. This is a clarity-first correctness
/// oracle, not an optimized execution path.
///
/// # Errors
///
/// Returns [`QuantizedLinearError`] for unsupported formats, non-block-aligned
/// weight rows, overflowing dimensions, or slices that do not exactly match
/// their declared shapes.
pub fn quantized_linear_f32(
    input: &[f32],
    weights: &[u8],
    format: QuantFormat,
    rows: usize,
    output_features: usize,
    input_features: usize,
) -> Result<Vec<f32>, QuantizedLinearError> {
    let dimensions = || QuantizedLinearError::DimensionsOverflow {
        rows,
        output_features,
        input_features,
    };
    let row_bytes = checked_row_bytes(format, input_features, rows, output_features)?;
    let input_len = rows.checked_mul(input_features).ok_or_else(dimensions)?;
    let weight_len = output_features
        .checked_mul(row_bytes)
        .ok_or_else(dimensions)?;
    let output_len = rows.checked_mul(output_features).ok_or_else(dimensions)?;

    if input.len() != input_len {
        return Err(QuantizedLinearError::InputLengthMismatch {
            expected: input_len,
            actual: input.len(),
        });
    }
    if weights.len() != weight_len {
        return Err(QuantizedLinearError::WeightLengthMismatch {
            expected: weight_len,
            actual: weights.len(),
        });
    }
    if output_len == 0 {
        return Ok(Vec::new());
    }

    let mut output = vec![0.0; output_len];
    let mut decoded_weight = vec![0.0; input_features];
    for output_feature in 0..output_features {
        let byte_start = output_feature * row_bytes;
        decode_row_into(
            &weights[byte_start..byte_start + row_bytes],
            format,
            &mut decoded_weight,
        );
        for row in 0..rows {
            let input_start = row * input_features;
            let mut sum = 0.0f32;
            for inner in 0..input_features {
                sum += input[input_start + inner] * decoded_weight[inner];
            }
            output[row * output_features + output_feature] = sum;
        }
    }
    Ok(output)
}

fn checked_row_bytes(
    format: QuantFormat,
    row_elements: usize,
    rows: usize,
    output_features: usize,
) -> Result<usize, QuantizedLinearError> {
    match format {
        QuantFormat::Q4_0 | QuantFormat::Q8_0 => {}
        QuantFormat::Q4_K | QuantFormat::Q6_K => {
            return Err(QuantizedLinearError::UnsupportedFormat { format });
        }
    }

    let block_elements = format.block_elements();
    if !row_elements.is_multiple_of(block_elements) {
        return Err(QuantizedLinearError::RowMisaligned {
            format,
            row_elements,
            block_elements,
        });
    }

    (row_elements / block_elements)
        .checked_mul(format.block_bytes())
        .ok_or(QuantizedLinearError::DimensionsOverflow {
            rows,
            output_features,
            input_features: row_elements,
        })
}

fn decode_row_into(bytes: &[u8], format: QuantFormat, output: &mut [f32]) {
    let block_elements = format.block_elements();
    let block_bytes = format.block_bytes();
    for (block_index, block) in bytes.chunks_exact(block_bytes).enumerate() {
        let scale = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let output_start = block_index * block_elements;
        match format {
            QuantFormat::Q4_0 => {
                for j in 0..16 {
                    let quants = block[2 + j];
                    output[output_start + j] = f32::from((quants & 0x0f) as i8 - 8) * scale;
                    output[output_start + j + 16] = f32::from((quants >> 4) as i8 - 8) * scale;
                }
            }
            QuantFormat::Q8_0 => {
                for j in 0..32 {
                    output[output_start + j] = f32::from(block[2 + j] as i8) * scale;
                }
            }
            QuantFormat::Q4_K | QuantFormat::Q6_K => {
                unreachable!("unsupported formats are rejected before decoding")
            }
        }
    }
}

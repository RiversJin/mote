use smallvec::SmallVec;

use crate::{Encoding, NumelOverflow, Shape};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Strides(SmallVec<[usize; 4]>);

impl Strides {
    pub fn new(strides: &[usize]) -> Self {
        Self(SmallVec::from_slice(strides))
    }

    pub fn values(&self) -> &[usize] {
        &self.0
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Layout {
    Contiguous,
    Strided(Strides),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LayoutError {
    #[error("layout rank mismatch: shape rank is {shape_rank}, strides rank is {strides_rank}")]
    RankMismatch {
        shape_rank: usize,
        strides_rank: usize,
    },

    #[error(
        "quantized encoding requires Layout::Contiguous; logical element strides are unsupported"
    )]
    UnsupportedQuantizedStrides,

    #[error(transparent)]
    NumelOverflow(#[from] NumelOverflow),

    #[error(
        "strided span overflow at axis {axis}: partial span {partial_span}, dimension {dimension}, stride {stride}"
    )]
    SpanOverflow {
        axis: usize,
        dimension: usize,
        stride: usize,
        partial_span: usize,
    },
}

impl Layout {
    pub fn validate_for(&self, shape: &Shape) -> Result<(), LayoutError> {
        let Self::Strided(strides) = self else {
            return Ok(());
        };

        if strides.rank() != shape.rank() {
            return Err(LayoutError::RankMismatch {
                shape_rank: shape.rank(),
                strides_rank: strides.rank(),
            });
        }

        Ok(())
    }

    pub fn validate_for_encoding(&self, encoding: &Encoding) -> Result<(), LayoutError> {
        if matches!(encoding, Encoding::Quantized(_)) && matches!(self, Self::Strided(_)) {
            return Err(LayoutError::UnsupportedQuantizedStrides);
        }

        Ok(())
    }

    pub fn is_contiguous(&self, shape: &Shape) -> Result<bool, LayoutError> {
        self.validate_for(shape)?;

        let Self::Strided(strides) = self else {
            return Ok(true);
        };

        if shape.dims().contains(&0) {
            return Ok(true);
        }

        shape.checked_numel()?;

        let mut expected_stride = 1;
        for (axis, (&dimension, &stride)) in
            shape.dims().iter().zip(strides.values()).enumerate().rev()
        {
            if dimension > 1 {
                if stride != expected_stride {
                    return Ok(false);
                }
                expected_stride = expected_stride
                    .checked_mul(dimension)
                    .ok_or(NumelOverflow {
                        axis,
                        dimension,
                        partial: expected_stride,
                    })?;
            }
        }

        Ok(true)
    }

    pub fn checked_span_elements(&self, shape: &Shape) -> Result<usize, LayoutError> {
        self.validate_for(shape)?;

        let Self::Strided(strides) = self else {
            return Ok(shape.checked_numel()?);
        };

        if shape.dims().contains(&0) {
            return Ok(0);
        }

        let mut span = 1usize;
        for (axis, (&dimension, &stride)) in shape.dims().iter().zip(strides.values()).enumerate() {
            let overflow = || LayoutError::SpanOverflow {
                axis,
                dimension,
                stride,
                partial_span: span,
            };
            let axis_span = (dimension - 1).checked_mul(stride).ok_or_else(overflow)?;
            span = span.checked_add(axis_span).ok_or_else(overflow)?;
        }

        Ok(span)
    }
}

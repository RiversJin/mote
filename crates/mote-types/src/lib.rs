mod layout;
mod tensor_desc;

pub use layout::{Layout, LayoutError, Strides};
pub use tensor_desc::{TensorDesc, TensorDescError};

use smallvec::SmallVec;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    I8,
    U8,
}

impl DType {
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 | Self::U8 => 1,
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    Q8_0,
    Q4_0,
    Q4_K,
    Q6_K,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Plain(DType),
    Quantized(QuantFormat),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(SmallVec<[usize; 4]>);

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("numel overflow at axis {axis}: partial product {partial} * dimension {dimension}")]
pub struct NumelOverflow {
    pub axis: usize,
    pub dimension: usize,
    pub partial: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("dimension axis {axis} is out of bounds for shape of rank {rank}")]
pub struct DimensionOutOfBounds {
    pub axis: usize,
    pub rank: usize,
}

impl Shape {
    pub fn new(dims: &[usize]) -> Self {
        Self(SmallVec::from_slice(dims))
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn replace_dim(&self, axis: usize, dimension: usize) -> Result<Self, DimensionOutOfBounds> {
        let rank = self.rank();
        let mut dimensions = self.0.clone();
        let current = dimensions
            .get_mut(axis)
            .ok_or(DimensionOutOfBounds { axis, rank })?;
        *current = dimension;

        Ok(Self(dimensions))
    }

    pub fn remove_dim(&self, axis: usize) -> Result<Self, DimensionOutOfBounds> {
        let rank = self.rank();
        if axis >= rank {
            return Err(DimensionOutOfBounds { axis, rank });
        }

        let mut dimensions = self.0.clone();
        dimensions.remove(axis);
        Ok(Self(dimensions))
    }

    pub fn insert_dim(&self, axis: usize, dimension: usize) -> Result<Self, DimensionOutOfBounds> {
        let rank = self.rank();
        if axis > rank {
            return Err(DimensionOutOfBounds { axis, rank });
        }

        let mut dimensions = self.0.clone();
        dimensions.insert(axis, dimension);
        Ok(Self(dimensions))
    }

    pub fn checked_numel(&self) -> Result<usize, NumelOverflow> {
        if self.0.contains(&0) {
            return Ok(0);
        }

        self.0
            .iter()
            .copied()
            .enumerate()
            .try_fold(1usize, |partial, (axis, dimension)| {
                partial.checked_mul(dimension).ok_or(NumelOverflow {
                    axis,
                    dimension,
                    partial,
                })
            })
    }
}

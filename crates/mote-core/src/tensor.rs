use mote_types::TensorDesc;

use crate::{Device, Storage, StorageId};

#[derive(Debug, Clone)]
pub struct Tensor {
    desc: TensorDesc,
    storage: Storage,
    byte_offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TensorError {
    #[error("tensor byte offset {byte_offset} exceeds storage size {storage_size}")]
    OffsetOutOfBounds {
        byte_offset: usize,
        storage_size: usize,
    },

    #[error("tensor byte range overflows usize: offset {byte_offset} + span {span_bytes}")]
    ByteRangeOverflow {
        byte_offset: usize,
        span_bytes: usize,
    },

    #[error("tensor byte range [{byte_offset}, {end_offset}) exceeds storage size {storage_size}")]
    StorageTooSmall {
        byte_offset: usize,
        end_offset: usize,
        storage_size: usize,
    },

    #[error(
        "tensor offset {byte_offset} is not aligned to {required_alignment} bytes; storage guarantees {storage_alignment}-byte base alignment"
    )]
    Misaligned {
        byte_offset: usize,
        required_alignment: usize,
        storage_alignment: usize,
    },
}

impl Tensor {
    pub fn new(
        desc: TensorDesc,
        storage: Storage,
        byte_offset: usize,
    ) -> Result<Self, TensorError> {
        let storage_size = storage.size_bytes();
        if byte_offset > storage_size {
            return Err(TensorError::OffsetOutOfBounds {
                byte_offset,
                storage_size,
            });
        }

        let span_bytes = desc.required_span_bytes();
        let end_offset =
            byte_offset
                .checked_add(span_bytes)
                .ok_or(TensorError::ByteRangeOverflow {
                    byte_offset,
                    span_bytes,
                })?;
        if end_offset > storage_size {
            return Err(TensorError::StorageTooSmall {
                byte_offset,
                end_offset,
                storage_size,
            });
        }

        if span_bytes != 0 {
            let required_alignment = desc.required_alignment();
            let storage_alignment = storage.alignment();
            if storage_alignment < required_alignment
                || !byte_offset.is_multiple_of(required_alignment)
            {
                return Err(TensorError::Misaligned {
                    byte_offset,
                    required_alignment,
                    storage_alignment,
                });
            }
        }

        Ok(Self {
            desc,
            storage,
            byte_offset,
        })
    }

    pub fn desc(&self) -> &TensorDesc {
        &self.desc
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn storage_id(&self) -> StorageId {
        self.storage.id()
    }

    pub fn device(&self) -> &Device {
        self.storage.device()
    }

    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    pub fn shares_storage_with(&self, other: &Self) -> bool {
        self.storage.shares_allocation_with(&other.storage)
    }
}

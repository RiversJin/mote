use crate::{Encoding, Layout, LayoutError, NumelOverflow, QuantFormat, Shape};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDesc {
    shape: Shape,
    encoding: Encoding,
    layout: Layout,
    numel: usize,
    span_bytes: usize,
    contiguous: bool,
    alignment_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TensorDescError {
    #[error(
        "quantized encoding {format:?} cannot describe a scalar: rank-0 shapes have no row dimension to quantize"
    )]
    QuantizedScalarShape { format: QuantFormat },

    #[error(
        "quantized encoding {format:?} row misaligned: last dimension {row_elements} is not a whole multiple of the {block_elements}-element block"
    )]
    QuantizedRowMisaligned {
        format: QuantFormat,
        row_elements: usize,
        block_elements: usize,
    },

    #[error(
        "quantized byte span overflow: {leading_rows} rows * {row_blocks} blocks per row * {block_bytes} bytes per block"
    )]
    QuantizedSpanOverflow {
        leading_rows: usize,
        row_blocks: usize,
        block_bytes: usize,
    },

    #[error(transparent)]
    NumelOverflow(#[from] NumelOverflow),

    #[error(transparent)]
    Layout(#[from] LayoutError),

    #[error("required byte span overflow: {span_elements} logical elements * {element_size} bytes")]
    ByteSpanOverflow {
        span_elements: usize,
        element_size: usize,
    },
}

impl TensorDesc {
    pub fn new(shape: Shape, encoding: Encoding, layout: Layout) -> Result<Self, TensorDescError> {
        layout.validate_for(&shape)?;
        layout.validate_for_encoding(&encoding)?;

        let numel = shape.checked_numel()?;
        let contiguous = layout.is_contiguous(&shape)?;
        let span_bytes = match encoding {
            Encoding::Plain(dtype) => {
                let span_elements = layout.checked_span_elements(&shape)?;
                let element_size = dtype.size_bytes();
                span_elements.checked_mul(element_size).ok_or(
                    TensorDescError::ByteSpanOverflow {
                        span_elements,
                        element_size,
                    },
                )?
            }
            // The last dimension is the contiguous quantized row; every outer
            // dimension counts whole rows of blocks.
            Encoding::Quantized(format) => {
                let block_elements = format.block_elements();
                let block_bytes = format.block_bytes();
                let (&row_elements, _) = shape
                    .dims()
                    .split_last()
                    .ok_or(TensorDescError::QuantizedScalarShape { format })?;
                if row_elements % block_elements != 0 {
                    return Err(TensorDescError::QuantizedRowMisaligned {
                        format,
                        row_elements,
                        block_elements,
                    });
                }
                if numel == 0 {
                    0
                } else {
                    let leading_rows = numel / row_elements;
                    let row_blocks = row_elements / block_elements;
                    let overflow = || TensorDescError::QuantizedSpanOverflow {
                        leading_rows,
                        row_blocks,
                        block_bytes,
                    };
                    leading_rows
                        .checked_mul(row_blocks)
                        .and_then(|total_blocks| total_blocks.checked_mul(block_bytes))
                        .ok_or_else(overflow)?
                }
            }
        };

        let alignment_bytes = match encoding {
            Encoding::Plain(dtype) => dtype.alignment_bytes(),
            Encoding::Quantized(format) => format.alignment_bytes(),
        };

        Ok(Self {
            shape,
            encoding,
            layout,
            numel,
            span_bytes,
            contiguous,
            alignment_bytes,
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn encoding(&self) -> &Encoding {
        &self.encoding
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn numel(&self) -> usize {
        self.numel
    }

    pub fn is_contiguous(&self) -> bool {
        self.contiguous
    }

    pub fn required_span_bytes(&self) -> usize {
        self.span_bytes
    }

    pub fn required_alignment(&self) -> usize {
        self.alignment_bytes
    }
}

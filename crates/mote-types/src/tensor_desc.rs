use crate::{Encoding, Layout, LayoutError, NumelOverflow, QuantFormat, Shape};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDesc {
    shape: Shape,
    encoding: Encoding,
    layout: Layout,
    numel: usize,
    span_bytes: usize,
    contiguous: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TensorDescError {
    #[error("quantized encoding {format:?} is not supported yet")]
    UnsupportedQuantizedEncoding { format: QuantFormat },

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
        let dtype = match encoding {
            Encoding::Plain(dtype) => dtype,
            Encoding::Quantized(format) => {
                return Err(TensorDescError::UnsupportedQuantizedEncoding { format });
            }
        };

        layout.validate_for(&shape)?;
        layout.validate_for_encoding(&encoding)?;

        let numel = shape.checked_numel()?;
        let contiguous = layout.is_contiguous(&shape)?;
        let span_elements = layout.checked_span_elements(&shape)?;
        let element_size = dtype.size_bytes();
        let span_bytes =
            span_elements
                .checked_mul(element_size)
                .ok_or(TensorDescError::ByteSpanOverflow {
                    span_elements,
                    element_size,
                })?;

        Ok(Self {
            shape,
            encoding,
            layout,
            numel,
            span_bytes,
            contiguous,
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
}

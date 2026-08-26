use crate::{DType, Encoding, Layout, LayoutError, NumelOverflow, QuantFormat, Shape};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDesc {
    shape: Shape,
    encoding: Encoding,
    layout: Layout,
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
        if let Encoding::Quantized(format) = encoding {
            return Err(TensorDescError::UnsupportedQuantizedEncoding { format });
        }

        layout.validate_for(&shape)?;
        layout.validate_for_encoding(&encoding)?;
        shape.checked_numel()?;

        let descriptor = Self {
            shape,
            encoding,
            layout,
        };
        descriptor.required_span_bytes()?;

        Ok(descriptor)
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

    pub fn numel(&self) -> Result<usize, NumelOverflow> {
        self.shape.checked_numel()
    }

    pub fn is_contiguous(&self) -> Result<bool, LayoutError> {
        self.layout.is_contiguous(&self.shape)
    }

    pub fn required_span_bytes(&self) -> Result<usize, TensorDescError> {
        let span_elements = self.layout.checked_span_elements(&self.shape)?;
        let element_size = self.plain_dtype()?.size_bytes();

        span_elements
            .checked_mul(element_size)
            .ok_or(TensorDescError::ByteSpanOverflow {
                span_elements,
                element_size,
            })
    }

    pub fn required_alignment(&self) -> Result<usize, TensorDescError> {
        Ok(self.plain_dtype()?.size_bytes())
    }

    fn plain_dtype(&self) -> Result<DType, TensorDescError> {
        match self.encoding {
            Encoding::Plain(dtype) => Ok(dtype),
            Encoding::Quantized(format) => {
                Err(TensorDescError::UnsupportedQuantizedEncoding { format })
            }
        }
    }
}

use mote_types::{
    DType, Encoding, Layout, LayoutError, QuantFormat, Shape, Strides, TensorDesc, TensorDescError,
};

#[test]
fn describes_a_contiguous_plain_tensor() {
    let descriptor = TensorDesc::new(
        Shape::new(&[2, 3, 4]),
        Encoding::Plain(DType::F32),
        Layout::Contiguous,
    )
    .unwrap();

    assert_eq!(descriptor.rank(), 3);
    assert_eq!(descriptor.numel(), 24);
    assert!(descriptor.is_contiguous());
    assert_eq!(descriptor.required_span_bytes(), 96);
    assert_eq!(descriptor.required_alignment(), 4);
}

#[test]
fn uses_the_strided_physical_span_for_plain_tensors() {
    let descriptor = TensorDesc::new(
        Shape::new(&[2, 3]),
        Encoding::Plain(DType::F16),
        Layout::Strided(Strides::new(&[5, 1])),
    )
    .unwrap();

    assert_eq!(descriptor.numel(), 6);
    assert!(!descriptor.is_contiguous());
    assert_eq!(descriptor.required_span_bytes(), 16);
    assert_eq!(descriptor.required_alignment(), 2);
}

#[test]
fn empty_tensor_requires_no_storage_span() {
    let descriptor = TensorDesc::new(
        Shape::new(&[2, 0, 4]),
        Encoding::Plain(DType::F32),
        Layout::Strided(Strides::new(&[usize::MAX, usize::MAX, usize::MAX])),
    )
    .unwrap();

    assert_eq!(descriptor.numel(), 0);
    assert_eq!(descriptor.required_span_bytes(), 0);
}

#[test]
fn rejects_layout_rank_mismatch() {
    assert_eq!(
        TensorDesc::new(
            Shape::new(&[2, 3]),
            Encoding::Plain(DType::F32),
            Layout::Strided(Strides::new(&[1])),
        ),
        Err(TensorDescError::Layout(LayoutError::RankMismatch {
            shape_rank: 2,
            strides_rank: 1,
        }))
    );
}

#[test]
fn rejects_required_byte_span_overflow() {
    assert_eq!(
        TensorDesc::new(
            Shape::new(&[usize::MAX]),
            Encoding::Plain(DType::F32),
            Layout::Contiguous,
        ),
        Err(TensorDescError::ByteSpanOverflow {
            span_elements: usize::MAX,
            element_size: 4,
        })
    );
}

#[test]
fn rejects_quantized_encoding_until_block_geometry_is_implemented() {
    assert_eq!(
        TensorDesc::new(
            Shape::new(&[32]),
            Encoding::Quantized(QuantFormat::Q8_0),
            Layout::Contiguous,
        ),
        Err(TensorDescError::UnsupportedQuantizedEncoding {
            format: QuantFormat::Q8_0,
        })
    );
}

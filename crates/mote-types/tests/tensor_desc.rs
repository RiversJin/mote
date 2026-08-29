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
fn describes_quantized_tensors_with_exact_block_geometry() {
    let cases = [
        (QuantFormat::Q4_0, 32usize, 18usize, 2usize),
        (QuantFormat::Q8_0, 32, 34, 2),
        (QuantFormat::Q4_K, 256, 144, 4),
        (QuantFormat::Q6_K, 256, 210, 2),
    ];

    for &(format, block_elements, block_bytes, alignment_bytes) in &cases {
        assert_eq!(format.block_elements(), block_elements);
        assert_eq!(format.block_bytes(), block_bytes);
        assert_eq!(format.alignment_bytes(), alignment_bytes);

        let descriptor = TensorDesc::new(
            Shape::new(&[block_elements]),
            Encoding::Quantized(format),
            Layout::Contiguous,
        )
        .unwrap();
        assert_eq!(descriptor.numel(), block_elements);
        assert!(descriptor.is_contiguous());
        assert_eq!(descriptor.required_span_bytes(), block_bytes);
        assert_eq!(descriptor.required_alignment(), alignment_bytes);
    }
}

#[test]
fn computes_the_quantized_span_from_leading_rows_of_blocks() {
    let q8_0 = TensorDesc::new(
        Shape::new(&[2, 3, 64]),
        Encoding::Quantized(QuantFormat::Q8_0),
        Layout::Contiguous,
    )
    .unwrap();
    assert_eq!(q8_0.numel(), 384);
    // 6 rows * 2 blocks per row * 34 bytes per block.
    assert_eq!(q8_0.required_span_bytes(), 408);

    let q6_k = TensorDesc::new(
        Shape::new(&[5, 256]),
        Encoding::Quantized(QuantFormat::Q6_K),
        Layout::Contiguous,
    )
    .unwrap();
    assert_eq!(q6_k.numel(), 1280);
    // 5 rows * 1 block per row * 210 bytes per block.
    assert_eq!(q6_k.required_span_bytes(), 1050);
}

#[test]
fn rejects_quantized_rows_that_are_not_whole_blocks() {
    assert_eq!(
        TensorDesc::new(
            Shape::new(&[2, 260]),
            Encoding::Quantized(QuantFormat::Q4_K),
            Layout::Contiguous,
        ),
        Err(TensorDescError::QuantizedRowMisaligned {
            format: QuantFormat::Q4_K,
            row_elements: 260,
            block_elements: 256,
        })
    );

    // A zero outer dimension does not waive row alignment.
    assert_eq!(
        TensorDesc::new(
            Shape::new(&[0, 33]),
            Encoding::Quantized(QuantFormat::Q8_0),
            Layout::Contiguous,
        ),
        Err(TensorDescError::QuantizedRowMisaligned {
            format: QuantFormat::Q8_0,
            row_elements: 33,
            block_elements: 32,
        })
    );
}

#[test]
fn rejects_quantized_scalar_shapes_without_a_row() {
    assert_eq!(
        TensorDesc::new(
            Shape::new(&[]),
            Encoding::Quantized(QuantFormat::Q8_0),
            Layout::Contiguous,
        ),
        Err(TensorDescError::QuantizedScalarShape {
            format: QuantFormat::Q8_0,
        })
    );
}

#[test]
fn quantized_tensors_with_any_zero_dimension_need_no_span() {
    let outer_zero = TensorDesc::new(
        Shape::new(&[0, 64]),
        Encoding::Quantized(QuantFormat::Q8_0),
        Layout::Contiguous,
    )
    .unwrap();
    assert_eq!(outer_zero.numel(), 0);
    assert_eq!(outer_zero.required_span_bytes(), 0);

    // A zero-length row is itself block aligned.
    let row_zero = TensorDesc::new(
        Shape::new(&[2, 0]),
        Encoding::Quantized(QuantFormat::Q4_0),
        Layout::Contiguous,
    )
    .unwrap();
    assert_eq!(row_zero.numel(), 0);
    assert_eq!(row_zero.required_span_bytes(), 0);

    let inner_zero = TensorDesc::new(
        Shape::new(&[3, 0, 256]),
        Encoding::Quantized(QuantFormat::Q6_K),
        Layout::Contiguous,
    )
    .unwrap();
    assert_eq!(inner_zero.numel(), 0);
    assert_eq!(inner_zero.required_span_bytes(), 0);
}

#[test]
fn rejects_quantized_span_overflow_while_logical_numel_fits() {
    // 576460752303423487 * 32 elements still fits, but the same tensor's
    // Q8_0 blocks need 34 bytes per 32 elements and overflow the byte span.
    let leading_rows = usize::MAX / 32;

    assert_eq!(
        TensorDesc::new(
            Shape::new(&[leading_rows, 32]),
            Encoding::Quantized(QuantFormat::Q8_0),
            Layout::Contiguous,
        ),
        Err(TensorDescError::QuantizedSpanOverflow {
            leading_rows,
            row_blocks: 1,
            block_bytes: 34,
        })
    );
}

#[test]
fn quantized_tensors_require_the_contiguous_layout() {
    assert_eq!(
        TensorDesc::new(
            Shape::new(&[2, 32]),
            Encoding::Quantized(QuantFormat::Q8_0),
            Layout::Strided(Strides::new(&[32, 1])),
        ),
        Err(TensorDescError::Layout(
            LayoutError::UnsupportedQuantizedStrides
        ))
    );
}

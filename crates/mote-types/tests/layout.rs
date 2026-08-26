use mote_types::{
    DType, Encoding, Layout, LayoutError, NumelOverflow, QuantFormat, Shape, Strides,
};

#[test]
fn contiguous_layout_uses_the_shape_numel_as_its_span() {
    let shape = Shape::new(&[2, 3, 4]);

    assert_eq!(Layout::Contiguous.validate_for(&shape), Ok(()));
    assert_eq!(Layout::Contiguous.is_contiguous(&shape), Ok(true));
    assert_eq!(Layout::Contiguous.checked_span_elements(&shape), Ok(24));
}

#[test]
fn scalar_and_empty_shapes_have_the_expected_spans() {
    let scalar = Shape::new(&[]);
    let empty = Shape::new(&[2, 0, 4]);
    let scalar_strided = Layout::Strided(Strides::new(&[]));
    let empty_strided = Layout::Strided(Strides::new(&[usize::MAX, usize::MAX, usize::MAX]));

    assert_eq!(Layout::Contiguous.checked_span_elements(&scalar), Ok(1));
    assert_eq!(Layout::Contiguous.checked_span_elements(&empty), Ok(0));
    assert_eq!(scalar_strided.checked_span_elements(&scalar), Ok(1));
    assert_eq!(scalar_strided.is_contiguous(&scalar), Ok(true));
    assert_eq!(empty_strided.checked_span_elements(&empty), Ok(0));
    assert_eq!(empty_strided.is_contiguous(&empty), Ok(true));
}

#[test]
fn calculates_a_strided_layout_span() {
    let shape = Shape::new(&[2, 3]);
    let layout = Layout::Strided(Strides::new(&[5, 1]));

    assert_eq!(layout.checked_span_elements(&shape), Ok(8));
    assert_eq!(layout.is_contiguous(&shape), Ok(false));
}

#[test]
fn recognizes_explicit_contiguous_strides() {
    let shape = Shape::new(&[2, 1, 4]);

    assert_eq!(
        Layout::Strided(Strides::new(&[4, usize::MAX, 1])).is_contiguous(&shape),
        Ok(true)
    );
    assert_eq!(
        Layout::Strided(Strides::new(&[4, 4, 2])).is_contiguous(&shape),
        Ok(false)
    );
}

#[test]
fn rejects_a_strided_layout_with_the_wrong_rank() {
    let shape = Shape::new(&[2, 3]);
    let layout = Layout::Strided(Strides::new(&[1]));

    assert_eq!(
        layout.validate_for(&shape),
        Err(LayoutError::RankMismatch {
            shape_rank: 2,
            strides_rank: 1,
        })
    );
    assert_eq!(
        layout.checked_span_elements(&shape),
        Err(LayoutError::RankMismatch {
            shape_rank: 2,
            strides_rank: 1,
        })
    );
}

#[test]
fn reports_contiguous_numel_overflow() {
    let shape = Shape::new(&[usize::MAX, 2]);

    assert_eq!(
        Layout::Contiguous.checked_span_elements(&shape),
        Err(LayoutError::NumelOverflow(NumelOverflow {
            axis: 1,
            dimension: 2,
            partial: usize::MAX,
        }))
    );
}

#[test]
fn reports_strided_span_overflow() {
    let shape = Shape::new(&[2, 2]);
    let layout = Layout::Strided(Strides::new(&[usize::MAX, 1]));

    assert_eq!(
        layout.checked_span_elements(&shape),
        Err(LayoutError::SpanOverflow {
            axis: 0,
            dimension: 2,
            stride: usize::MAX,
            partial_span: 1,
        })
    );
}

#[test]
fn quantized_encoding_requires_the_explicit_contiguous_layout() {
    let quantized = Encoding::Quantized(QuantFormat::Q4_K);

    assert_eq!(Layout::Contiguous.validate_for_encoding(&quantized), Ok(()));
    assert_eq!(
        Layout::Strided(Strides::new(&[4, 1])).validate_for_encoding(&quantized),
        Err(LayoutError::UnsupportedQuantizedStrides)
    );
}

#[test]
fn plain_encoding_allows_strided_layouts() {
    let plain = Encoding::Plain(DType::F32);

    assert_eq!(
        Layout::Strided(Strides::new(&[4, 1])).validate_for_encoding(&plain),
        Ok(())
    );
}

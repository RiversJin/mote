use mote_reference::{RopeError, rope_f32};
use mote_types::RopeLayout;

#[test]
fn applies_half_split_rotation_and_preserves_tail() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let output = rope_f32(
        &input,
        &[0.0, 1.0],
        &[1.0, 0.0],
        1,
        1,
        6,
        4,
        RopeLayout::HalfSplit,
    )
    .unwrap();

    assert_eq!(output, [-3.0, 2.0, 1.0, 4.0, 5.0, 6.0]);
}

#[test]
fn applies_interleaved_rotation_to_every_head() {
    let input = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];

    let output = rope_f32(
        &input,
        &[0.0, 1.0],
        &[1.0, 0.0],
        1,
        2,
        4,
        4,
        RopeLayout::Interleaved,
    )
    .unwrap();

    assert_eq!(output, [-2.0, 1.0, 3.0, 4.0, -20.0, 10.0, 30.0, 40.0]);
}

#[test]
fn supports_empty_and_zero_rotary_dimensions() {
    assert!(
        rope_f32(&[], &[], &[], 0, 3, 8, 8, RopeLayout::HalfSplit)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        rope_f32(&[1.0, 2.0], &[], &[], 1, 1, 2, 0, RopeLayout::Interleaved,).unwrap(),
        [1.0, 2.0]
    );
}

#[test]
fn rejects_invalid_rotary_geometry() {
    assert_eq!(
        rope_f32(&[], &[], &[], 0, 0, 4, 3, RopeLayout::HalfSplit).unwrap_err(),
        RopeError::OddRotaryDimension { rotary_dim: 3 }
    );
    assert_eq!(
        rope_f32(&[], &[], &[], 0, 0, 4, 6, RopeLayout::HalfSplit).unwrap_err(),
        RopeError::RotaryDimensionTooLarge {
            rotary_dim: 6,
            head_dim: 4,
        }
    );
}

#[test]
fn rejects_mismatched_slice_lengths() {
    assert_eq!(
        rope_f32(&[0.0], &[1.0], &[0.0], 1, 1, 4, 2, RopeLayout::HalfSplit).unwrap_err(),
        RopeError::InputLengthMismatch {
            expected: 4,
            actual: 1,
        }
    );
    assert_eq!(
        rope_f32(&[0.0; 4], &[], &[0.0], 1, 1, 4, 2, RopeLayout::HalfSplit).unwrap_err(),
        RopeError::CosLengthMismatch {
            expected: 1,
            actual: 0,
        }
    );
    assert_eq!(
        rope_f32(&[0.0; 4], &[1.0], &[], 1, 1, 4, 2, RopeLayout::HalfSplit).unwrap_err(),
        RopeError::SinLengthMismatch {
            expected: 1,
            actual: 0,
        }
    );
}

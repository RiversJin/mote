//! Behavioural integration tests for [`mote_reference::rms_norm_f32`].
//!
//! Expected values are produced by an independent, deliberately naive `f64`
//! re-implementation (plus a few hand-computed anchors), not by calling back
//! into the code under test.

use mote_reference::{ReferenceError, rms_norm_f32};

const EPSILON: f32 = 1e-6;
const TOLERANCE: f32 = 1e-5;

/// Independent oracle: plain `f64` RMS norm, written from the math definition.
fn expected_rms_norm(input: &[f32], weight: &[f32], hidden_size: usize, epsilon: f32) -> Vec<f32> {
    let mut expected = Vec::with_capacity(input.len());
    for row in input.chunks(hidden_size) {
        let sum_sq: f64 = row.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        let mean_square = sum_sq / hidden_size as f64;
        let inv_rms = 1.0 / (mean_square + f64::from(epsilon)).sqrt();
        for (&x, &w) in row.iter().zip(weight.iter()) {
            expected.push((f64::from(x) * inv_rms * f64::from(w)) as f32);
        }
    }
    expected
}

#[test]
fn two_rows_match_independent_computation() {
    let hidden_size = 4;
    let input = [1.0, 2.0, 3.0, 4.0, -2.0, 0.0, 2.0, -4.0];
    let weight = [1.0, 2.0, 0.5, 1.5];

    let got = rms_norm_f32(&input, &weight, hidden_size, EPSILON).unwrap();
    let expected = expected_rms_norm(&input, &weight, hidden_size, EPSILON);

    assert_eq!(got.len(), input.len());
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < TOLERANCE,
            "element {i}: got {g}, expected {e}"
        );
    }

    // Hand-computed anchors.
    //
    // Row 0: mean_square = (1 + 4 + 9 + 16) / 4 = 7.5, so
    //   out[0] = 1 * rsqrt(7.5 + 1e-6) * 1.0  ≈  0.3651484
    //   out[1] = 2 * rsqrt(7.5 + 1e-6) * 2.0  ≈  1.4605935
    // Row 1: mean_square = (4 + 0 + 4 + 16) / 4 = 6, so
    //   out[4] = -2 * rsqrt(6 + 1e-6) * 1.0   ≈ -0.8164966
    //   out[6] =  2 * rsqrt(6 + 1e-6) * 0.5   ≈  0.4082483
    //   out[7] = -4 * rsqrt(6 + 1e-6) * 1.5   ≈ -2.4494897
    let anchors = [
        (0usize, 0.365_148_4_f32),
        (1, 1.460_593_5),
        (4, -0.816_496_6),
        (6, 0.408_248_3),
        (7, -2.449_489_7),
    ];
    for (i, want) in anchors {
        assert!(
            (got[i] - want).abs() < 1e-4,
            "anchor element {i}: got {}, expected {want}",
            got[i]
        );
    }
}

#[test]
fn empty_input_returns_empty_vec() {
    let got = rms_norm_f32(&[], &[1.0, 2.0, 3.0, 4.0], 4, EPSILON).unwrap();
    assert!(got.is_empty());
}

#[test]
fn zero_hidden_size_is_rejected() {
    let err = rms_norm_f32(&[1.0, 2.0], &[1.0, 1.0], 0, EPSILON).unwrap_err();
    assert_eq!(err, ReferenceError::ZeroHiddenSize);
}

#[test]
fn weight_length_mismatch_is_rejected() {
    let err = rms_norm_f32(&[1.0, 2.0], &[1.0], 2, EPSILON).unwrap_err();
    assert_eq!(
        err,
        ReferenceError::WeightLengthMismatch {
            weight_len: 1,
            hidden_size: 2
        }
    );
}

#[test]
fn input_not_multiple_of_hidden_size_is_rejected() {
    let err = rms_norm_f32(&[1.0, 2.0, 3.0], &[1.0, 1.0], 2, EPSILON).unwrap_err();
    assert_eq!(
        err,
        ReferenceError::InputNotMultiple {
            input_len: 3,
            hidden_size: 2
        }
    );
}

#[test]
fn negative_epsilon_is_rejected() {
    let err = rms_norm_f32(&[1.0, 2.0], &[1.0, 1.0], 2, -1e-6).unwrap_err();
    assert_eq!(err, ReferenceError::InvalidEpsilon { epsilon: -1e-6 });
}

#[test]
fn non_finite_epsilon_is_rejected() {
    for epsilon in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err = rms_norm_f32(&[1.0, 2.0], &[1.0, 1.0], 2, epsilon).unwrap_err();
        assert!(
            matches!(err, ReferenceError::InvalidEpsilon { .. }),
            "epsilon {epsilon} should be rejected"
        );
    }
}

//! Behavioural integration tests for [`mote_reference::fused_add_rms_norm_f32`].
//!
//! Expected values are produced by an independent, deliberately naive `f64`
//! re-implementation (plus a few hand-computed anchors), not by calling back
//! into the code under test.

use mote_reference::{FusedAddRmsNormOutput, ReferenceError, fused_add_rms_norm_f32};

const EPSILON: f32 = 1e-6;
const TOLERANCE: f32 = 1e-5;

/// Independent oracle: plain `f64` fused add + RMS norm, written from the
/// math definition.
fn expected_fused_add_rms_norm(
    input: &[f32],
    residual: &[f32],
    weight: &[f32],
    hidden_size: usize,
    epsilon: f32,
) -> (Vec<f32>, Vec<f32>) {
    let sums: Vec<f64> = input
        .iter()
        .zip(residual.iter())
        .map(|(&x, &r)| f64::from(x) + f64::from(r))
        .collect();

    let mut normalized = Vec::with_capacity(sums.len());
    for row in sums.chunks(hidden_size) {
        let sum_sq: f64 = row.iter().map(|&x| x * x).sum();
        let mean_square = sum_sq / hidden_size as f64;
        let inv_rms = 1.0 / (mean_square + f64::from(epsilon)).sqrt();
        for (&x, &w) in row.iter().zip(weight.iter()) {
            normalized.push((x * inv_rms * f64::from(w)) as f32);
        }
    }

    let residual_out = sums.iter().map(|&x| x as f32).collect();
    (normalized, residual_out)
}

#[test]
fn two_rows_match_independent_computation() {
    let hidden_size = 4;
    let input = [1.0, 2.0, 3.0, 4.0, -2.0, 0.0, 2.0, -4.0];
    let residual = [0.5, -1.0, 2.0, 1.0, 2.0, 1.0, -1.0, 4.0];
    let weight = [1.0, 2.0, 0.5, 1.5];

    let got = fused_add_rms_norm_f32(&input, &residual, &weight, hidden_size, EPSILON).unwrap();
    let (expected_normalized, expected_residual) =
        expected_fused_add_rms_norm(&input, &residual, &weight, hidden_size, EPSILON);

    assert_eq!(got.normalized.len(), input.len());
    for (i, (g, e)) in got
        .normalized
        .iter()
        .zip(expected_normalized.iter())
        .enumerate()
    {
        assert!(
            (g - e).abs() < TOLERANCE,
            "normalized element {i}: got {g}, expected {e}"
        );
    }

    assert_eq!(got.residual.len(), input.len());
    for (i, (g, e)) in got
        .residual
        .iter()
        .zip(expected_residual.iter())
        .enumerate()
    {
        assert!(
            (g - e).abs() < TOLERANCE,
            "residual element {i}: got {g}, expected {e}"
        );
    }

    // Hand-computed anchors. Sums are row 0 [1.5, 1, 5, 5], row 1 [0, 1, 1, 0].
    //
    // Row 0: mean_square = (2.25 + 1 + 25 + 25) / 4 = 13.3125, so
    //   normalized[0] = 1.5 * rsqrt(13.3125 + 1e-6) * 1.0 ≈ 0.4111132
    //   normalized[3] = 5.0 * rsqrt(13.3125 + 1e-6) * 1.5 ≈ 2.0555661
    // Row 1: mean_square = (0 + 1 + 1 + 0) / 4 = 0.5, so
    //   normalized[4] = 0.0 * rsqrt(0.5 + 1e-6) * 1.0   = 0.0
    //   normalized[5] = 1.0 * rsqrt(0.5 + 1e-6) * 2.0   ≈ 2.8284243
    //   normalized[6] = 1.0 * rsqrt(0.5 + 1e-6) * 0.5   ≈ 0.7071061
    let anchors = [
        (0usize, 0.411_113_2_f32),
        (3, 2.055_566),
        (4, 0.0),
        (5, 2.828_424_3),
        (6, 0.707_106_1),
    ];
    for (i, want) in anchors {
        assert!(
            (got.normalized[i] - want).abs() < 1e-4,
            "anchor normalized element {i}: got {}, expected {want}",
            got.normalized[i]
        );
    }

    // The returned residual is exactly the element-wise input + residual sum;
    // these operands are all exactly representable, so equality is exact.
    assert_eq!(got.residual, [1.5, 1.0, 5.0, 5.0, 0.0, 1.0, 1.0, 0.0]);
}

#[test]
fn zero_weight_normalizes_but_keeps_residual() {
    // weight = 0 keeps the residual path intact while zeroing `normalized`.
    let got = fused_add_rms_norm_f32(
        &[1.0, 2.0, -3.0, 4.0],
        &[4.0, -3.0, 2.0, -1.0],
        &[0.0, 0.0, 0.0, 0.0],
        4,
        EPSILON,
    )
    .unwrap();

    assert_eq!(got.normalized, [0.0, 0.0, 0.0, 0.0]);
    assert_eq!(got.residual, [5.0, -1.0, -1.0, 3.0]);
}

#[test]
fn output_struct_is_comparable() {
    // The public struct supports the derive surface callers rely on.
    // sum = 4, mean_square = 16, inv_rms = 0.25, so normalized = 4 * 0.25 * 2;
    // powers of two only, exact in f32.
    let got = fused_add_rms_norm_f32(&[3.0], &[1.0], &[2.0], 1, 0.0).unwrap();
    assert_eq!(
        got,
        FusedAddRmsNormOutput {
            normalized: vec![2.0],
            residual: vec![4.0],
        }
    );
}

#[test]
fn empty_input_returns_empty_vecs() {
    let got = fused_add_rms_norm_f32(&[], &[], &[1.0, 2.0, 3.0, 4.0], 4, EPSILON).unwrap();
    assert!(got.normalized.is_empty());
    assert!(got.residual.is_empty());
}

#[test]
fn zero_hidden_size_is_rejected() {
    let err =
        fused_add_rms_norm_f32(&[1.0, 2.0], &[1.0, 1.0], &[1.0, 1.0], 0, EPSILON).unwrap_err();
    assert_eq!(err, ReferenceError::ZeroHiddenSize);
}

#[test]
fn weight_length_mismatch_is_rejected() {
    let err = fused_add_rms_norm_f32(&[1.0, 2.0], &[1.0, 1.0], &[1.0], 2, EPSILON).unwrap_err();
    assert_eq!(
        err,
        ReferenceError::WeightLengthMismatch {
            weight_len: 1,
            hidden_size: 2
        }
    );
}

#[test]
fn residual_length_mismatch_is_rejected() {
    let err = fused_add_rms_norm_f32(&[1.0, 2.0], &[1.0], &[1.0, 1.0], 2, EPSILON).unwrap_err();
    assert_eq!(
        err,
        ReferenceError::ResidualLengthMismatch {
            residual_len: 1,
            input_len: 2
        }
    );
}

#[test]
fn input_not_multiple_of_hidden_size_is_rejected() {
    let err = fused_add_rms_norm_f32(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], &[1.0, 1.0], 2, EPSILON)
        .unwrap_err();
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
    let err = fused_add_rms_norm_f32(&[1.0, 2.0], &[1.0, 1.0], &[1.0, 1.0], 2, -1e-6).unwrap_err();
    assert_eq!(err, ReferenceError::InvalidEpsilon { epsilon: -1e-6 });
}

#[test]
fn non_finite_epsilon_is_rejected() {
    for epsilon in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err =
            fused_add_rms_norm_f32(&[1.0, 2.0], &[1.0, 1.0], &[1.0, 1.0], 2, epsilon).unwrap_err();
        assert!(
            matches!(err, ReferenceError::InvalidEpsilon { .. }),
            "epsilon {epsilon} should be rejected"
        );
    }
}

use mote_reference::{MatmulError, matmul_f32};

#[test]
fn multiplies_row_major_matrices() {
    let lhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    let output = matmul_f32(&lhs, &rhs, 2, 2, 3).unwrap();

    assert_eq!(output, [58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn zero_inner_dimension_produces_zero_matrix() {
    assert_eq!(matmul_f32(&[], &[], 2, 3, 0).unwrap(), vec![0.0; 6]);
}

#[test]
fn zero_output_dimension_produces_empty_matrix() {
    assert!(
        matmul_f32(&[], &[1.0, 2.0, 3.0, 4.0], 0, 2, 2)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn rejects_mismatched_input_lengths() {
    assert_eq!(
        matmul_f32(&[1.0], &[1.0; 6], 2, 2, 3).unwrap_err(),
        MatmulError::LhsLengthMismatch {
            expected: 6,
            actual: 1,
        }
    );
    assert_eq!(
        matmul_f32(&[1.0; 6], &[1.0], 2, 2, 3).unwrap_err(),
        MatmulError::RhsLengthMismatch {
            expected: 6,
            actual: 1,
        }
    );
}

#[test]
fn rejects_dimension_overflow() {
    assert_eq!(
        matmul_f32(&[], &[], usize::MAX, 1, 2).unwrap_err(),
        MatmulError::DimensionsOverflow {
            m: usize::MAX,
            n: 1,
            k: 2,
        }
    );
}

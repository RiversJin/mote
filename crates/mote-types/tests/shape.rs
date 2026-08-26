use mote_types::{DimensionOutOfBounds, NumelOverflow, Shape};

#[test]
fn calculates_numel() {
    assert_eq!(Shape::new(&[2, 3, 4]).checked_numel(), Ok(24));
    assert_eq!(Shape::new(&[]).checked_numel(), Ok(1));
}

#[test]
fn reports_the_dimension_that_overflows() {
    assert_eq!(
        Shape::new(&[usize::MAX, 2]).checked_numel(),
        Err(NumelOverflow {
            axis: 1,
            dimension: 2,
            partial: usize::MAX,
        })
    );
}

#[test]
fn zero_dimension_makes_the_shape_empty_without_overflow() {
    assert_eq!(Shape::new(&[usize::MAX, 2, 0]).checked_numel(), Ok(0));
}

#[test]
fn replaces_a_dimension() {
    let shape = Shape::new(&[2, 3, 4]);

    let replaced = shape.replace_dim(1, 5).unwrap();

    assert_eq!(replaced.dims(), &[2, 5, 4]);
    assert_eq!(shape.dims(), &[2, 3, 4]);
}

#[test]
fn removes_a_dimension() {
    let shape = Shape::new(&[2, 3, 4]);

    let removed = shape.remove_dim(1).unwrap();

    assert_eq!(removed.dims(), &[2, 4]);
    assert_eq!(shape.dims(), &[2, 3, 4]);
}

#[test]
fn inserts_a_dimension() {
    let shape = Shape::new(&[2, 4]);

    let inserted = shape.insert_dim(1, 3).unwrap();
    let appended = inserted.insert_dim(inserted.rank(), 5).unwrap();

    assert_eq!(inserted.dims(), &[2, 3, 4]);
    assert_eq!(appended.dims(), &[2, 3, 4, 5]);
    assert_eq!(shape.dims(), &[2, 4]);
}

#[test]
fn rejects_out_of_bounds_dimension_changes() {
    let shape = Shape::new(&[2, 3]);

    assert_eq!(
        shape.replace_dim(2, 4),
        Err(DimensionOutOfBounds { axis: 2, rank: 2 })
    );
    assert_eq!(
        shape.remove_dim(2),
        Err(DimensionOutOfBounds { axis: 2, rank: 2 })
    );
    assert_eq!(
        shape.insert_dim(3, 4),
        Err(DimensionOutOfBounds { axis: 3, rank: 2 })
    );
    assert_eq!(shape.dims(), &[2, 3]);
}

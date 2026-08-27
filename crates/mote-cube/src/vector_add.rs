use cubecl::prelude::*;

/// Add two equally sized vectors element by element.
#[cube(launch)]
pub fn vector_add_kernel<F: Float>(lhs: &Array<F>, rhs: &Array<F>, output: &mut Array<F>) {
    let index = ABSOLUTE_POS;

    if index < output.len() {
        output[index] = lhs[index] + rhs[index];
    }
}

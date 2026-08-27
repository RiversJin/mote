#[cfg(feature = "cpu")]
use cubecl::cpu::{CpuDevice, CpuRuntime};
#[cfg(feature = "hip")]
use cubecl::hip::{AmdDevice, HipRuntime};
use cubecl::prelude::*;
use mote_cube::vector_add::vector_add_kernel;

fn assert_adds_vectors<R: Runtime>(client: ComputeClient<R>) {
    const CUBE_DIM: u32 = 256;

    let lhs = (0..1_000).map(|value| value as f32).collect::<Vec<_>>();
    let rhs = (0..1_000)
        .map(|value| (value as f32) * 0.5)
        .collect::<Vec<_>>();
    let expected = lhs
        .iter()
        .zip(&rhs)
        .map(|(lhs, rhs)| lhs + rhs)
        .collect::<Vec<_>>();

    let lhs_handle = client.create_from_slice(f32::as_bytes(&lhs));
    let rhs_handle = client.create_from_slice(f32::as_bytes(&rhs));
    let output_handle = client.empty(expected.len() * size_of::<f32>());
    let cube_count = u32::try_from(expected.len())
        .expect("test vector length fits in u32")
        .div_ceil(CUBE_DIM);

    unsafe {
        vector_add_kernel::launch::<f32, R>(
            &client,
            CubeCount::new_1d(cube_count),
            CubeDim::new_1d(CUBE_DIM),
            ArrayArg::from_raw_parts(lhs_handle, lhs.len()),
            ArrayArg::from_raw_parts(rhs_handle, rhs.len()),
            ArrayArg::from_raw_parts(output_handle.clone(), expected.len()),
        );
    }

    let actual = client.read_one_unchecked(output_handle);
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual, expected);
}

#[cfg(feature = "cpu")]
#[test]
fn adds_vectors_on_cpu() {
    assert_adds_vectors::<CpuRuntime>(CpuRuntime::client(&CpuDevice));
}

#[cfg(feature = "hip")]
#[test]
fn adds_vectors_on_hip() {
    assert_adds_vectors::<HipRuntime>(HipRuntime::client(&AmdDevice::default()));
}

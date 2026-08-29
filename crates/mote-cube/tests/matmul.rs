#![cfg(feature = "cpu")]

use cubecl::{
    CubeElement,
    cpu::{CpuDevice, CpuRuntime},
    prelude::Runtime,
};
use mote_cube::{CubeContext, matmul::matmul};
use mote_types::{DType, Device, Encoding, Layout, Shape, TensorDesc};

#[test]
fn multiplies_f32_matrices_on_cpu() {
    let context =
        CubeContext::<CpuRuntime>::new(Device::cpu(), CpuRuntime::client(&CpuDevice)).unwrap();
    let lhs_values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs_values = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let lhs = context
        .from_bytes(plain_f32_desc(&[2, 3]), f32::as_bytes(&lhs_values))
        .unwrap();
    let rhs = context
        .from_bytes(plain_f32_desc(&[3, 2]), f32::as_bytes(&rhs_values))
        .unwrap();
    let output = context.empty(plain_f32_desc(&[2, 2])).unwrap();

    matmul(&context, &lhs, &rhs, &output).unwrap();

    let actual = context.read_bytes(&output).unwrap();
    assert_eq!(f32::from_bytes(&actual), vec![58.0, 64.0, 139.0, 154.0]);
}

fn plain_f32_desc(shape: &[usize]) -> TensorDesc {
    TensorDesc::new(
        Shape::new(shape),
        Encoding::Plain(DType::F32),
        Layout::Contiguous,
    )
    .unwrap()
}

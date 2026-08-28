#![cfg(feature = "vulkan-tests")]

use mote_backend_vulkan::{
    VulkanContext, VulkanContextOptions, VulkanMemoryMode, vector_add::vector_add,
};
use mote_types::{DType, Encoding, Layout, Shape, TensorDesc};

#[test]
fn adds_vectors_on_vulkan() {
    let lhs = (0..1_000).map(|value| value as f32).collect::<Vec<_>>();
    let rhs = (0..1_000)
        .map(|value| value as f32 * 0.5)
        .collect::<Vec<_>>();
    let expected = lhs
        .iter()
        .zip(&rhs)
        .map(|(lhs, rhs)| lhs + rhs)
        .collect::<Vec<_>>();

    let context = VulkanContext::new(0).unwrap();
    eprintln!("running raw Vulkan vector_add on {}", context.device_name());
    let desc = plain_f32_desc(lhs.len());
    let lhs = context.from_bytes(desc.clone(), &f32_bytes(&lhs)).unwrap();
    let rhs = context.from_bytes(desc.clone(), &f32_bytes(&rhs)).unwrap();
    let output = context.empty(desc).unwrap();

    let memory = context.memory_info(&output).unwrap();
    assert_eq!(context.memory_mode(), VulkanMemoryMode::DeviceLocal);
    assert!(memory.device_local);
    eprintln!(
        "storage memory type {} on heap {} ({} MiB)",
        memory.memory_type_index,
        memory.heap_index,
        memory.heap_size_bytes / 1024 / 1024
    );

    vector_add(&context, &lhs, &rhs, &output).unwrap();

    let actual = context.read_bytes(&output).unwrap();
    let (actual, remainder) = actual.as_chunks::<{ size_of::<f32>() }>();
    assert!(remainder.is_empty());
    let actual = actual
        .iter()
        .map(|bytes| f32::from_ne_bytes(*bytes))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn supports_host_visible_storage_when_requested() {
    let context = VulkanContext::with_options(
        0,
        VulkanContextOptions {
            memory_mode: VulkanMemoryMode::HostVisible,
        },
    )
    .unwrap();
    let tensor = context.empty(plain_f32_desc(16)).unwrap();
    let memory = context.memory_info(&tensor).unwrap();

    assert_eq!(context.memory_mode(), VulkanMemoryMode::HostVisible);
    assert!(memory.host_visible);
    assert!(memory.host_cached);
}

fn plain_f32_desc(len: usize) -> TensorDesc {
    TensorDesc::new(
        Shape::new(&[len]),
        Encoding::Plain(DType::F32),
        Layout::Contiguous,
    )
    .unwrap()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

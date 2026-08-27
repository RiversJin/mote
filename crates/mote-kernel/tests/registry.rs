use mote_kernel::{KernelArgs, KernelError, KernelImpl, KernelKey, KernelRegistry};
use mote_types::{BackendKind, DType, Device};

struct TestKernel {
    name: &'static str,
    backend: Option<BackendKind>,
}

impl KernelImpl for TestKernel {
    fn name(&self) -> &'static str {
        self.name
    }

    fn supports(&self, device: &Device, _key: &KernelKey) -> bool {
        self.backend
            .is_none_or(|backend| backend == device.backend())
    }

    fn launch(&self, _device: &Device, _args: &KernelArgs<'_>) -> Result<(), KernelError> {
        Ok(())
    }
}

fn rms_norm_key() -> KernelKey {
    KernelKey {
        op: "rms_norm",
        dtype: Some(DType::F16),
        shape_class: None,
    }
}

#[test]
fn resolves_a_kernel_supported_by_the_target_device() {
    let key = rms_norm_key();
    let mut registry = KernelRegistry::default();
    registry.register(
        key.clone(),
        TestKernel {
            name: "portable",
            backend: None,
        },
    );

    let resolved = registry
        .resolve(&key, &Device::new(BackendKind::Hip, 0))
        .unwrap();

    assert_eq!(resolved.name(), "portable");
}

#[test]
fn later_supported_registrations_override_portable_fallbacks() {
    let key = rms_norm_key();
    let mut registry = KernelRegistry::default();
    registry.register(
        key.clone(),
        TestKernel {
            name: "portable",
            backend: None,
        },
    );
    registry.register(
        key.clone(),
        TestKernel {
            name: "cuda-specialized",
            backend: Some(BackendKind::Cuda),
        },
    );

    assert_eq!(
        registry
            .resolve(&key, &Device::new(BackendKind::Cuda, 0))
            .unwrap()
            .name(),
        "cuda-specialized"
    );
    assert_eq!(
        registry
            .resolve(&key, &Device::new(BackendKind::Hip, 0))
            .unwrap()
            .name(),
        "portable"
    );
}

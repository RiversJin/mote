# Mote

**Small models, close to the metal.**

Mote is an experimental on-device inference runtime for small language and multimodal models. Its default path favors portable GPU kernels, while keeping explicit escape hatches for backend-specific specialization when profiling proves they are worth it.

Initial design goals:

- Rust-first runtime and systems layer.
- Portable kernels first, currently centered on CubeCL.
- First-class CUDA, ROCm/HIP, and Vulkan targets.
- Backend-specific kernels as profiled specializations, not the default implementation path.
- Runtime profiling, tuning, and specialization caching.
- A narrow boundary between model/runtime semantics and backend-specific code.

The project is intentionally early. The first milestone is to prove the kernel/runtime architecture before adding a full model frontend.

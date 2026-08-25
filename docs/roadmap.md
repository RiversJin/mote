# Roadmap

## Milestone 0 — prove the boundary

- Build a minimal Rust workspace.
- Define device, kernel, registry, and specialization concepts.
- Wire one real CubeCL kernel.
- Run the same operator through at least two GPU backends.

## Milestone 1 — first inference-shaped kernels

- RMSNorm.
- RoPE.
- GEMV / small GEMM baseline.
- Basic profiling and benchmark harness.
- Differential correctness tests.

## Milestone 2 — specialization

- CUDA NVRTC escape hatch.
- ROCm hipRTC escape hatch.
- Vulkan SPIR-V escape hatch.
- Device fingerprinting.
- Specialization cache and invalidation rules.

## Milestone 3 — model runtime

- Tensor/storage abstraction.
- Memory planner.
- Quantized weights.
- KV cache.
- Minimal decoder-only transformer execution.
- DLPack/Torch interoperability where useful.

## Non-goals for now

- Becoming a general-purpose GPU compute framework.
- Reimplementing every vendor library.
- Premature architecture-specific kernels before profiling proves a need.

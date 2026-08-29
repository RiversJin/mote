# Roadmap

Direction: Rust control plane + vendor-native data planes. Phased order:

```text
Phase 0  freeze CubeCL, prove the control-plane boundary
Phase 1  first inference primitives on vendor-native paths + CPU oracle
Phase 2  exit all remaining CubeCL consumers, then delete mote-cube
Phase 3  model runtime (planner, quantized weights, KV cache, end-to-end)
         Vulkan portable fallback is a parallel track, not a gate
```

Current ROCm lane status:

- native HIP storage/stream plus plan-cached hipBLASLt F16/F32 matmul;
- pure-Rust matmul, RMSNorm, and RoPE correctness oracles;
- native HIP F16 RMSNorm, fused residual-add + RMSNorm, and RoPE
  differential-tested on the GPU;
- GGML-compatible Q4_0/Q8_0/Q4_K/Q6_K block geometry and physical spans,
  plus Q4_0/Q8_0 row decoding and quantized-linear CPU oracles;
- native HIP Q4_0 weight-only linear with device-resident encoded weights,
  F16 activations/output, F32 accumulation, and GPU differential tests;
- repeated decode-shaped Q4_0 benchmarks with validation, sample-spread
  reporting, and deterministic device-storage accounting; the initial native
  baseline is about 0.58--0.66 TOP/s on the larger RX 7900 XTX cases;
- next: inspect the target model's actual GGUF tensor mix, then implement and
  tune its dominant K-quant block-dot path rather than guessing the format.

## Milestone 0 — control plane boundary + CubeCL freeze

- Keep the minimal Rust workspace and the `mote-types` / `mote-core` /
  `mote-kernel` / `mote-runtime` layering; introduce a separate pure-Rust
  reference crate/module rather than putting operators in `mote-core`.
- Derive the operator-semantics boundary from real implementations, starting
  with the existing HIP matmul and its reference implementation. Keep the
  registry prototype small; do not introduce a shared CUDA/HIP context or
  BLAS trait before a real CUDA implementation exposes actual duplication.
- Declare `mote-cube` frozen: no new kernels, features, or consumers.
- Inventory all CubeCL consumers: `mote-cube` portable kernels and HIP
  export, its tests and benches, the CubeCL comparison feature/tests in
  `mote-backend-vulkan`.
- Extract the current test matmul into the pure-Rust reference, then add
  RMSNorm as the first new inference-shaped oracle.
- Wire the differential test harness: oracle vs backend implementation.

## Milestone 1 — first inference primitives, vendor-native first

First batch of inference primitives, in implementation order:

1. **GEMV / small GEMM** — via hipBLASLt (ROCm) and cuBLASLt (CUDA),
   generalizing the existing `mote-backend-hip` plan-cached matmul;
   per-shape algorithm/plan caches.
2. **RMSNorm** (and fused residual + RMSNorm) — pure-Rust oracle first, then
   custom HIP / custom CUDA.
3. **RoPE** — pure-Rust oracle first, then custom HIP / custom CUDA.
4. **Quantized linear** — the first actually-needed `QuantFormat` with block
   geometry, dequant-fused and weight-only paths.
5. **Attention (GQA) with KV-cache append** — hipBLASLt batched GEMM + AITER
   where available on ROCm; cuDNN SDPA / CUTLASS FMHA templates on CUDA.
6. **Fused SiLU/SwiGLU MLP** and KV-cache maintenance operations.

Also in this milestone:

- Basic profiling and benchmark harness (device timestamps, rotated order,
  median reporting) extended to all new primitives.
- Device fingerprinting for backend/architecture-aware selection.
- Differential correctness tests against the CPU oracle for every primitive.
- `mote-backend-cuda` grows from stub to cuBLASLt matmul parity with
  `mote-backend-hip`.
- Backend selection remains runtime-family scoped. Vulkan and the pure-Rust
  reference are not transparent per-operator fallbacks for HIP/CUDA tensors;
  cross-backend execution requires an explicit copy and plan transition.

## Milestone 2 — CubeCL exit and deletion

- Migrate the remaining `mote-cube` consumers (CubeK matmul benchmark, HIP
  export path, vector add, CubeCL-Vulkan comparison tests) onto
  `mote-backend-hip` / `mote-backend-cuda` / `mote-backend-vulkan` and the
  pure-Rust oracle.
- Keep `mote-cube` untouched while consumers remain — the frozen crate is
  the stable baseline for differential comparison during migration.
- When the consumer count reaches zero: delete the `mote-cube` crate, drop
  its features from workspace/CI, and mark `docs/cube-integration-design.md`
  as historical.

## Milestone 3 — model runtime

- Complete the tensor/storage abstraction (mmap/GGUF-backed weights).
- Memory planner (liveness, arenas, alias validation).
- Quantized weights end to end.
- KV-cache manager.
- Minimal decoder-only transformer executed end to end on one vendor-native
  backend, then a second.
- DLPack/Torch interoperability where useful.

## Parallel track — Vulkan portable fallback

- Keep the Slang->SPIR-V path in `mote-backend-vulkan` healthy.
- Implement enough primitives (the same first-batch list) to run a small
  model end to end on Vulkan-only devices.
- Performance parity with vendor-native stacks is explicitly not a goal;
  correctness parity with the oracle is.

## Non-goals for now

- Becoming a general-purpose GPU compute framework.
- A unified kernel DSL or portable runtime — the thing we froze and are
  deleting, not reviving.
- Reimplementing vendor libraries when calling them suffices.
- Premature architecture-specific kernels before profiling proves a need.
- Making Vulkan the primary path on any CUDA or ROCm target.
- Tuning the CPU reference backend for performance; it is an oracle.

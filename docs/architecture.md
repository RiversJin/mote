# Mote architecture

Mote is an on-device inference runtime for small models.

The architecture is a **Rust control plane over vendor-native data planes**:

```text
                  Rust control plane
    model/graph semantics / scheduling / memory planning
    kernel dispatch / registry / profiling / plan caches
              |              |             |          |
           CUDA           ROCm         Vulkan      reference
        cuBLASLt        hipBLASLt     Slang->SPIR-V  pure Rust
        cuDNN           AITER         portable       reference
        CUTLASS         CK            fallback only  oracle only
        custom CUDA     custom HIP
```

The control plane is pure Rust and contains no kernel DSL. Each data plane
maps operators to that vendor's native libraries and kernels. What is shared
across backends is **operator semantics**, not a unified runtime or a portable
kernel language.

## Control plane (Rust)

Mote owns all model and runtime semantics in Rust:

- model/graph semantics and execution orchestration (`mote-runtime`);
- operator semantics: op names, dtypes/encodings, layouts, shape classes,
  numerics contracts, error taxonomy (`mote-kernel`, `mote-types`);
- tensor/storage abstraction with type-erased `StorageImpl` (`mote-core`);
- kernel selection, registry, and dispatch;
- memory planning, activation arenas, KV-cache lifecycle;
- profiling, benchmarking, and per-shape plan/algorithm caches;
- host<->device data movement policy.

The control plane must stay buildable and testable without any GPU toolchain.
Backend-specific types never leak through the `KernelImpl` boundary.

## Data plane (vendor-native)

Each backend maps an operator to the strongest native mechanism available for
it on that vendor. Backend resources (library handles, plans, compiled
modules, device allocations) are private to the backend crate.

### CUDA

- **cuBLASLt** — GEMM/GEMV with heuristic algorithm selection and cached plans;
  the default for matmul-shaped work on NVIDIA.
- **cuDNN** — inference ops that already exist as cuDNN operations.
- **CUTLASS** — template instantiation where cuBLASLt coverage or performance
  is insufficient.
- **custom CUDA** (precompiled or NVRTC) — residual fusion, RoPE, RMSNorm,
  quantized paths, and other glue that libraries do not cover.

### ROCm

- **hipBLASLt** — GEMM/GEMV with heuristic algorithm selection and per-shape
  plan caching; already proven by `mote-backend-hip`.
- **AITER** — AMD-specific fused attention/norm kernels where available.
- **Composable Kernel (CK)** — template ops not covered by hipBLASLt/AITER.
- **custom HIP** (precompiled or hipRTC) — residual fusion, RoPE, quantized
  paths, and glue.

### Vulkan — portable fallback only

- Slang shaders compiled to SPIR-V at build time, cooperative-matrix matmul
  where the device supports it (`mote-backend-vulkan`).
- Exists for devices with no native vendor stack (integrated GPUs, odd
  drivers, portability testing).
- It is never the primary path on NVIDIA or AMD targets, and it is never the
  source of truth for performance decisions there.

### Pure-Rust reference

- A separate reference crate/module with straightforward implementations of
  supported operators, runnable in CI with zero GPU dependencies.
- Purpose: **correctness oracle** and metadata/logic testability, not
  performance. It does not belong in `mote-core`, and it is not a tuned CPU
  execution backend.

## What is shared across backends

Operator semantics — explicitly **not** a runtime, a kernel DSL, or a
single-source portable kernel:

- the semantic `KernelKey` (operator, dtype/quantization, shape class, later
  layout class and feature requirements);
- the argument contract (inputs, outputs, scalars, read/write roles);
- the numerics contract and tolerance policy against the CPU oracle;
- the error taxonomy translated at the `KernelImpl` boundary.

Each backend crate owns its own implementation of that contract using its own
native mechanisms. hipBLASLt plans, cuBLASLt plans, SPIR-V modules, and
NVRTC/hipRTC-compiled kernels are per-backend resources.

## Kernel selection

Runtime-family selection happens before operator selection. HIP allocations
stay on HIP, CUDA allocations stay on CUDA, and Vulkan allocations stay on
Vulkan. Mote never silently falls through from a HIP/CUDA operator to Vulkan
or the CPU reference: that would require an explicit cross-backend copy and a
different execution plan.

Within the selected runtime family, precedence is:

```text
CUDA/ROCm: vendor library -> custom vendor kernel -> unsupported
Vulkan:    native Vulkan implementation -> unsupported
reference: pure-Rust oracle, used by tests and validation
```

Vulkan is selected as the portable backend for a device/session when a vendor
stack is unavailable or deliberately not used; it is not a per-operator
escape hatch for vendor-owned tensors. Selection inside a backend is
deterministic and cached. The specialization key can eventually include
operator, backend, device architecture, dtype/quantization format, shape
class, layout/stride class, and feature requirements; the first prototype
keeps it small (`op`, `dtype`, `shape_class`) and expands it only when real
kernels demand it.

## CubeCL: frozen

CubeCL is **frozen**. It is no longer the portable default kernel path:

- no new kernels, features, or consumers may be added to `mote-cube`;
- existing consumers (`mote-cube` portable kernels and HIP export, its tests
  and benches, the CubeCL comparison feature in `mote-backend-vulkan`)
  migrate to the vendor-native data planes plus the pure-Rust oracle;
- `mote-cube` is **deleted once the last consumer exits** — not before. The
  freeze precedes deletion so migration has a stable baseline and differential
  comparisons remain possible in the meantime.

Rationale: the CubeCL-first tiering assumed one portable kernel source could
be the default execution path with backend-specific kernels as rare
exceptions. Practice inverted this. Vendor libraries already deliver the
performance-critical inference operators, a single-source kernel DSL does not
pay for its complexity across CUDA + ROCm + Vulkan, and the parts worth
sharing turned out to be operator semantics, which live in Rust anyway.

## Human-in-the-loop specialization

A future specialization loop should look like:

```text
vendor-native default path
      |
   profile on target device
      |
 measured hotspot
      |
 candidate: library config / template / custom kernel
      |
 correctness vs pure-Rust CPU oracle
      |
 benchmark on target device
      |
 winning plan/specialization cache
```

The human or agent is not on the inference critical path. It participates in
creating and reviewing new specializations, while runtime selection remains
deterministic and cached.

## Correctness model

```text
pure Rust CPU reference = mathematical oracle

vendor-native impls     = differential-tested vs oracle with tolerances

Vulkan fallback         = tested vs oracle, and vs vendor-native impls
                          where the same hardware is available
```

Exact equality is acceptable only for exact inputs (e.g. integer or
bit-identical library paths); floating-point kernels use tolerance-based
comparison against the oracle.

## Scope

The initial operator set follows actual small-model inference workloads
rather than generic compute benchmarks:

- RMSNorm (and fused residual + RMSNorm)
- RoPE
- GEMV / small GEMM
- quantized linear
- attention (GQA) with KV cache
- KV-cache operations
- fused SiLU/SwiGLU MLP

A full model frontend and Torch/DLPack integration come after the
control-plane / vendor-native boundary has proved itself.

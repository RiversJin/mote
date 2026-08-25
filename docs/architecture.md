# Mote architecture

Mote is an on-device inference runtime for small models.

Its kernel strategy is deliberately tiered:

1. **Portable reference kernel** — implement the operator once using CubeCL.
2. **Compile-time/runtime specialization** — let the portable path specialize and autotune where possible.
3. **Profile-driven backend override** — only for measured hotspots, replace a kernel with CUDA, HIP, or Vulkan-specific code.
4. **Architecture-specific specialization** — reserve device-specific kernels for cases where the performance win justifies the maintenance cost.

The portable implementation is both the default execution path and an executable reference for differential correctness testing of specialized kernels.

## Runtime boundary

Mote owns model/runtime semantics in Rust. Backend-specific code is treated as an implementation detail behind a narrow kernel interface.

A specialization is selected using a key that can eventually include:

- operator
- backend
- device architecture
- dtype / quantization format
- shape class
- layout / stride class
- feature requirements

The first prototype intentionally keeps this key small and expands it only when real kernels demand it.

## Human-in-the-loop specialization

A future specialization loop should look like:

```text
portable kernel
      |
   profile
      |
 measured hotspot
      |
 candidate backend kernel
      |
 correctness vs portable reference
      |
 benchmark on target device
      |
 winning specialization cache
```

The human or agent is not on the inference critical path. It participates in creating and reviewing new specializations, while runtime selection remains deterministic and cached.

## Scope

The initial operator set should follow actual small-model inference workloads rather than generic compute benchmarks:

- RMSNorm
- RoPE
- GEMV / small GEMM
- quantized linear
- attention
- KV-cache operations

A full model frontend and Torch/DLPack integration come after the kernel/runtime boundary has proved itself.

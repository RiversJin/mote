# Mote

**Small models, close to the metal.**

Mote is an experimental on-device inference runtime for small language and
multimodal models. Rust owns the control plane; vendor-native CUDA/ROCm data
planes are the primary accelerated paths, with direct Vulkan as a portable
fallback.

Initial design goals:

- Rust-first runtime and systems layer.
- Vendor libraries first for GEMM/attention-shaped work, custom native kernels
  where fusion or quantized layouts require them.
- First-class CUDA and ROCm/HIP targets, plus a direct Slang/Vulkan fallback.
- CubeCL is frozen as a temporary comparison baseline while its remaining
  consumers are migrated.
- Runtime profiling, tuning, and specialization caching.
- A narrow boundary between model/runtime semantics and backend-specific code.

The project is intentionally early. The first milestone is to prove the kernel/runtime architecture before adding a full model frontend.

## Development

Use `direnv allow` or enter `nix develop` before running Cargo commands. The
development shell provides Rust, the Slang shader compiler, and the Vulkan
loader and tools. Direct Vulkan shaders are compiled from Slang to SPIR-V at
Cargo build time; set `SLANGC` to override the `slangc` executable.

## Matrix multiplication benchmark

Run the CubeK and direct Slang/Vulkan F32 plus F16/F32 cooperative-matrix
comparison with:

```console
cargo bench -p mote-backend-vulkan --features comparison --bench matmul
```

The benchmark uses device timestamp queries, rotates implementation order, and
reports the median of seven samples. CubeCL's `vulkan` feature can lower kernels
to SPIR-V and use `VK_KHR_cooperative_matrix`; this is a native Vulkan path and
does not imply that the portable browser WebGPU/WGSL path exposes cooperative
matrices.

For an F16-input, F32-accumulation/output comparison against ROCm's rocBLAS and
hipBLASLt libraries, use the optional ROCm development shell:

```console
nix develop .#rocm
cargo bench -p mote-backend-hip --features rocm --bench matmul
cargo bench -p mote-backend-hip --features rocm --bench quantized_linear
```

`mote-backend-hip` owns its HIP stream and device allocations and calls
hipBLASLt directly. Its 32 MiB workspace is allocated lazily on the first
matrix multiplication, and the fastest heuristic candidate is cached per
`(M, N, K)` shape. CubeCL is not involved in this path.

The quantized-linear benchmark covers decode-shaped Q4_0 weights that remain
encoded in native HIP device storage. It reports the median of repeated,
synchronization-amortized batches, the full sample spread, deterministic
encoded/storage sizes, and validates sampled output elements against the
pure-Rust block decoder and linear oracle.

The older CubeK and standalone library comparison remains available for
allocator and implementation experiments:

```console
cargo bench -p mote-cube --no-default-features --features hip --bench matmul_hip
cmake -S tools -B target/rocm-bench -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
cmake --build target/rocm-bench
./target/rocm-bench/rocm-matmul-bench
```

Set `MOTE_CUBE_HIP_MEMORY` to `default` (the benchmark default), `bounded`, or
`exclusive` to compare CubeCL's HIP allocation strategies.

The native Mote benchmark reports batched, synchronization-amortized wall time
and validates every output. The older CubeK comparison inserts HIP events on
CubeCL's physical stream through the native submission interface. Its
`bounded` mode uses fixed size-classed pools; the standalone library benchmark
uses HIP events and measures the fastest hipBLASLt heuristic candidate.

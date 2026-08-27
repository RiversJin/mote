# CubeCL integration design draft

> Status: design draft. This document describes the intended boundary between Mote tensors/storage and CubeCL. It is deliberately narrower than a final backend architecture and should evolve from real kernel work.

## Goal

Mote should use CubeCL as the default portable GPU kernel path without making CubeCL types part of Mote's core tensor/runtime semantics.

The target data path is:

```text
Mote Tensor
    |
Mote Storage
    |
CubeStorage<R>
    |
CubeCL Handle / ComputeClient<R>
    |
CUDA / HIP / Vulkan / optional CPU
```

The first integration milestone is to make the existing `vector_add` bring-up kernel consume Mote `Tensor` objects rather than raw CubeCL handles created directly inside the test.

## Non-goals for the first integration

Do not try to solve these yet:

- final `KernelArgs` ABI;
- graph scheduling;
- memory planning;
- asynchronous dependency tracking;
- cross-backend copies;
- arbitrary strided CubeCL allocations;
- quantized storage;
- backend-specific specialization;
- generic external-memory interop;
- making CubeCL CPU a production backend.

The first integration should only prove the Mote -> CubeCL resource boundary.

## Feature policy

CubeCL CPU is useful for GitHub CI, but it should not be a default dependency of normal Mote builds because the CPU backend pulls in the LLVM-based code generation stack.

Recommended feature layout for `mote-cube`:

```toml
[features]
default = []
cpu = ["cubecl/cpu"]
cuda = ["cubecl/cuda"]
hip = ["cubecl/hip"]
vulkan = ["cubecl/vulkan"]
```

CI should explicitly enable the CPU feature for portable-kernel integration tests:

```text
ordinary workspace tests
    -> no CubeCL CPU required

mote-cube portable-kernel tests
    -> --features cpu
```

Local AMD development can explicitly use `--features hip`; NVIDIA can use `--features cuda`.

CPU execution is an integration backend, not the mathematical oracle. Kernel tests should eventually compare against a small pure-Rust reference implementation.

## CubeContext

`mote-cube` should own a context that ties a Mote `Device` to one CubeCL runtime/client.

Conceptual sketch:

```rust
pub struct CubeContext<R: cubecl::Runtime> {
    device: mote_types::Device,
    client: cubecl::prelude::ComputeClient<R>,
}
```

The exact trait bounds/types should follow CubeCL's public API rather than this sketch.

Responsibilities:

- own or clone the `ComputeClient<R>` needed to allocate and launch CubeCL resources;
- expose the corresponding Mote `Device`;
- create Mote-backed CubeCL storage;
- validate that tensors passed to a CubeCL kernel belong to this context/runtime/device;
- provide narrowly scoped access to CubeCL handles for the provider implementation.

It should not own model/runtime semantics, kernel selection, graph scheduling, or memory planning.

Possible initial API:

```text
CubeContext::device()
CubeContext::from_bytes(desc, bytes)
CubeContext::empty(desc)
CubeContext::handle(tensor)
CubeContext::read_bytes(tensor)
```

Names and signatures are intentionally not frozen.

## CubeStorage

CubeCL allocations should implement Mote's `StorageImpl` through a concrete storage type owned by `mote-cube`.

Conceptual sketch:

```rust
pub struct CubeStorage<R: cubecl::Runtime> {
    device: Device,
    handle: /* CubeCL handle type */,
    size_bytes: usize,
    alignment: usize,
    _runtime: PhantomData<R>,
}
```

Why keep the runtime type parameter even if the handle itself is erased/non-generic:

- a CubeCL allocation belongs to a particular runtime/client family;
- `CubeStorage<HipRuntime>` should not accidentally be launched through a CUDA client;
- the type parameter gives us a cheap compile-time guard inside the provider implementation.

`CubeStorage<R>` should implement:

```rust
StorageImpl
```

so the rest of Mote continues to see only:

```text
Storage -> Arc<dyn StorageImpl>
```

Backend handles remain private to `mote-cube`.

## Allocation strategy for v0

Use raw CubeCL buffers for the first integration:

```text
ComputeClient::create_from_slice
ComputeClient::empty
```

and map them to:

```text
Mote Layout::Contiguous
```

Do not use CubeCL tensor-layout allocation helpers yet if they may introduce pitched/padded layouts. Pitched layouts are useful later, but they should enter Mote through an explicit mapping to `Layout::Strided` after the basic boundary works.

Initial invariant:

```text
CubeStorage v0 <-> one contiguous raw CubeCL buffer
```

## Creating Mote tensors

The context should allocate a CubeCL buffer, wrap it in `CubeStorage<R>`, then construct ordinary Mote `Storage` and `Tensor` values.

Conceptually:

```text
TensorDesc
   |
CubeContext::empty
   |
CubeCL Handle
   |
CubeStorage<R>
   |
Storage::new(...)
   |
Tensor::new(desc, storage, 0)
```

For an upload path:

```text
bytes / typed slice
   |
CubeContext::from_bytes
   |
CubeCL create_from_slice
   |
CubeStorage<R>
   |
Mote Tensor
```

The first implementation can require:

- plain encoding;
- contiguous layout;
- byte offset 0 for newly allocated tensors;
- storage span exactly matching or exceeding `TensorDesc::required_span_bytes()`.

## Accessing a CubeCL handle from a Tensor

`mote-cube` may downcast the tensor's storage implementation:

```text
Tensor
  -> Storage
  -> downcast_ref::<CubeStorage<R>>()
  -> CubeCL handle
```

Failure should be explicit and provider-specific, for example:

```text
wrong storage backend/runtime
wrong Mote device
unsupported layout/encoding
```

Do not expose a public `Tensor::cube_handle()` API in `mote-core`.

The dependency direction should remain:

```text
mote-core knows StorageImpl
mote-cube knows CubeStorage
mote-core never knows CubeCL
```

## Vector-add integration milestone

The existing CubeCL test currently proves:

```text
CubeCL client -> raw handles -> vector_add kernel -> result
```

The next version should prove:

```text
CubeContext
   |
Mote input Tensor A
Mote input Tensor B
Mote output Tensor
   |
provider extracts CubeStorage handles
   |
vector_add kernel
   |
read output through CubeContext
```

A temporary provider-facing function is sufficient:

```rust
pub fn vector_add<R: Runtime>(
    ctx: &CubeContext<R>,
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
) -> Result<(), CubeError>
```

This function should validate at least:

- all tensors belong to the same expected Cube runtime/device;
- all tensors use plain `F32` for the bring-up test;
- all tensors are contiguous;
- shapes/numel match;
- output has enough storage by construction.

Do not force this function through `KernelImpl::launch()` yet.

## KernelArgs: intentionally deferred

The current placeholder:

```rust
pub struct KernelArgs<'a> {
    pub opaque: &'a [usize],
}
```

should be treated as temporary.

After one real Mote-backed CubeCL kernel exists, redesign the launch boundary from observed needs.

A likely direction is something structurally similar to:

```rust
pub struct KernelArgs<'a> {
    pub inputs: &'a [&'a Tensor],
    pub outputs: &'a [&'a Tensor],
    pub scalars: &'a [Scalar],
}
```

but this is not yet a decision.

Questions to answer from the first integration:

- does a kernel provider need only `Tensor`, or also an execution/context object?
- how should mutable outputs be represented without leaking eager-style mutation semantics?
- where should stream/queue selection live?
- should launch arguments retain explicit read/write roles for future hazard tracking?
- should scalar arguments be typed values or operator-specific structs?

Do not design the final ABI until these questions have concrete examples.

## Correctness testing

For each real operator, distinguish three layers:

```text
pure Rust reference
    = mathematical oracle

CubeCL CPU
    = portable provider integration test in GitHub CI

CubeCL HIP/CUDA/Vulkan
    = backend/codegen validation on real GPU hardware
```

For the vector-add bring-up test, exact equality is fine for the chosen inputs.

For RMSNorm and later floating-point kernels, use tolerance-based comparison against the pure-Rust reference.

## CI layout

Recommended split:

```text
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --all-targets --locked --no-default-features
cargo test -p mote-cube --features cpu --locked
```

The exact commands may need adjustment depending on how Cargo feature unification affects workspace tests.

The important policy is:

- normal workspace/core tests should not require CubeCL CPU/LLVM;
- the dedicated portable-kernel CI job explicitly pays the LLVM dependency cost;
- GPU backend tests remain opt-in until dedicated runners exist.

## Error boundary

`mote-cube` should have a provider-specific error type rather than turning all failures into strings immediately.

Likely categories:

```text
UnsupportedDevice
WrongStorageRuntime
DeviceMismatch
UnsupportedEncoding
UnsupportedLayout
ShapeMismatch
DTypeMismatch
LaunchFailure
ReadbackFailure
```

Only translate to generic `KernelError` when crossing the eventual `KernelImpl` boundary.

## Expected crate responsibilities after integration

```text
mote-types
    Device / BackendKind
    DType / Encoding
    Shape / Layout / TensorDesc

mote-core
    Storage / StorageImpl
    Tensor

mote-kernel
    semantic KernelKey
    KernelImpl / KernelRegistry
    generic dispatch errors/arguments (after redesign)

mote-cube
    CubeContext<R>
    CubeStorage<R>
    CubeCL kernel implementations
    provider-specific validation/errors

mote-runtime
    kernel selection and execution orchestration
```

## Suggested implementation order

1. Change `mote-cube` default features from `cpu` to empty.
2. Adjust CI so CubeCL CPU is enabled explicitly for portable-kernel tests.
3. Add `CubeContext<R>` around `ComputeClient<R>` and Mote `Device`.
4. Add `CubeStorage<R>` implementing `StorageImpl`.
5. Implement contiguous `empty` and upload helpers returning Mote `Tensor`.
6. Implement provider-private extraction/downcast from `Tensor` to CubeCL handle.
7. Rewrite vector-add integration test to use Mote tensors end to end.
8. Add a pure-Rust vector-add/reference path if useful for the test structure.
9. Only then redesign `KernelArgs` from the observed interface.
10. Move on to RMSNorm as the first inference-shaped operator.

## Completion criterion

This integration milestone is complete when the vector-add test can start with ordinary host values, create Mote tensors through `CubeContext`, launch the CubeCL kernel using only resources recovered from those Mote tensors, and read back the correct result.

At that point the data path is genuinely:

```text
Mote -> CubeCL -> device
```

rather than a standalone CubeCL demo living inside the repository.

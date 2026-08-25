# Tensor and Storage design draft

> Status: design draft. This document defines the intended boundaries and invariants for Mote's first tensor/storage layer. It is deliberately not an implementation spec; details should be tightened by code and tests.

## Context

Mote is an on-device inference runtime, not a general eager tensor framework. The tensor layer should primarily serve:

- static or mostly-static inference graphs;
- model weights, including quantized weights;
- temporary activations planned into reusable arenas;
- stateful buffers such as KV cache;
- CUDA, ROCm/HIP, and Vulkan backends;
- CubeCL/CubeK as the portable kernel path;
- backend-specific escape hatches without leaking backend types through the runtime.

The central split is:

```text
TensorDesc  = what the tensor logically is
Storage     = where the bytes physically live
Tensor      = a typed/layout-aware view into Storage
```

A graph should be able to reason about `TensorDesc` without allocating any physical memory. Physical storage is chosen later by the execution/memory planning layer.

## Goals

1. **Views are first-class.** Slice, reshape, transpose, KV-cache windows, and arena reuse must not require redesigning `Tensor` later.
2. **Storage ownership is separate from tensor metadata.** Multiple tensors may alias one allocation.
3. **Backend resources stay opaque to `mote-core`.** `CudaBuffer`, HIP handles, `VkBuffer`, and CubeCL handles must not become core tensor types.
4. **Quantized representation is not modeled as a scalar dtype.** Q4/Q8 block encodings have different physical semantics from plain scalar tensors.
5. **Byte offsets are canonical.** A tensor's position inside storage is always represented as a byte offset.
6. **Runtime dispatch stays runtime dispatch.** Avoid making the whole runtime generic over `Tensor<B: Backend>`.
7. **Memory planning is possible from metadata alone.** The graph/planner can determine size, alignment, liveness, and reuse before execution.

## Non-goals for v0

- PyTorch-compatible eager semantics.
- Automatic differentiation.
- Arbitrary mutation APIs on tensors.
- Arbitrary strided views over block-quantized encodings.
- Cross-device transparent tensors.
- A universal external-memory interoperability ABI on day one.
- Supporting every exotic layout before a real kernel requires it.

## Conceptual model

```text
                         logical description
                                |
                           TensorDesc
                  shape / encoding / layout
                                |
                                v
                              Tensor
                         /              \
                    TensorDesc       byte_offset
                                          |
                                          v
                                       Storage
                                          |
                              backend-owned allocation
                          /          |           |        \
                        CPU        CUDA         HIP      Vulkan
```

`Tensor` does not own a device allocation directly. `Storage` owns or references the allocation; a `Tensor` is a view into it.

## Device

CubeCL is not a device kind. It is a portable kernel/runtime provider which may target CUDA, HIP, or Vulkan.

Initial shape:

```rust
pub enum BackendKind {
    Cpu,
    Cuda,
    Hip,
    Vulkan,
}

pub struct Device {
    pub backend: BackendKind,
    pub ordinal: u32,
}
```

Keep hardware capabilities out of `Device` itself. A later `DeviceInfo`/`DeviceCaps` can contain architecture, subgroup/warp size, supported dtypes, memory properties, and a specialization fingerprint.

Questions such as "is this gfx1100?" belong to device capability/specialization logic, not tensor identity.

## Scalar dtype and physical encoding

Do not put quantized block formats directly into `DType`.

A scalar dtype describes one independently addressable scalar value:

```rust
pub enum DType {
    F32,
    F16,
    BF16,
    // Add FP8 variants when needed.
    I32,
    I8,
    U8,
}
```

Physical representation is separate:

```rust
pub enum Encoding {
    Plain(DType),
    Quantized(QuantFormat),
}

pub enum QuantFormat {
    Q8_0,
    Q4_0,
    Q4_K,
    Q6_K,
    // Extend from actual model requirements.
}
```

Why:

```text
Plain(F16):
    one logical element <-> one 16-bit scalar

Q4_K:
    a block of logical values <-> scales/mins/packed values
```

For a block format, asking `dtype.size_of()` is conceptually wrong. Instead, `Encoding`/`QuantFormat` must be able to answer physical-layout questions such as required bytes for a logical shape and required alignment/block granularity.

The exact quantization API should be deferred until the first GGUF/quantized-linear implementation, but the core type split should exist from the start.

## Shape

`Shape` should be a small owned value, optimized for typical inference ranks without baking in a maximum rank.

Sketch:

```rust
pub struct Shape(SmallVec<[usize; 4]>);
```

Useful operations:

- rank;
- dimensions;
- checked `numel`;
- dimension replacement/removal/insertion;
- shape compatibility checks.

All size multiplication must be checked for overflow.

## Layout

For v0, keep layout intentionally small:

```rust
pub enum Layout {
    Contiguous,
    Strided(Strides),
}

pub struct Strides(SmallVec<[usize; 4]>);
```

For `Encoding::Plain`, strides may initially be represented in logical-element units. `Tensor` itself still uses a byte offset into storage.

`Contiguous` is worth keeping as an explicit common case rather than eagerly materializing a stride vector everywhere. Kernels can cheaply reject unsupported layouts:

```text
portable RMSNorm v0: contiguous only
later kernel: selected stride patterns
```

### Quantized layouts

Initially, quantized tensors should be treated as format-defined contiguous block layouts. Arbitrary transpose/slice over quantized storage is deferred.

When quantized views are eventually needed, their legality should be defined in terms of the format's block geometry rather than pretending they are normal scalar-strided tensors.

## TensorDesc

Proposed conceptual shape:

```rust
pub struct TensorDesc {
    pub shape: Shape,
    pub encoding: Encoding,
    pub layout: Layout,
}
```

`TensorDesc` contains no device allocation and no backend handle.

It should eventually provide checked helpers such as:

```text
rank()
numel()
is_contiguous()
required_span_bytes()
required_alignment()
```

`required_span_bytes()` is important: for a non-contiguous view the byte span touched by a tensor is not necessarily `numel * element_size`.

For quantized encodings, physical size is format-defined.

## Storage

Core storage is type-erased and shared:

```rust
#[derive(Clone)]
pub struct Storage {
    // Stable identity is useful for alias detection/debugging.
    id: StorageId,
    inner: Arc<dyn StorageImpl>,
}

pub trait StorageImpl: Send + Sync {
    fn device(&self) -> &Device;
    fn size_bytes(&self) -> usize;
    fn alignment(&self) -> usize;

    fn as_any(&self) -> &dyn Any;
}
```

This is a sketch, not a frozen API.

Possible implementations later:

```text
CpuOwnedStorage
MmapStorage
CubeStorage
CudaStorage
HipStorage
VulkanStorage
ExternalStorage
```

`mote-core` should not need to know their concrete resource handles.

### Storage identity

A stable `StorageId` is useful even if two implementations happen to expose the same underlying native allocation. It enables:

- alias detection;
- debug output;
- planner validation;
- tracing;
- future hazard/synchronization tracking.

Whether imported external aliases need a shared identity scheme can be deferred.

### Alignment

Storage reports its base alignment. Tensor construction/planning must additionally satisfy any alignment required by its encoding or selected kernel.

Do not assume the allocation's base alignment implies every tensor view is equally aligned: `byte_offset` matters.

## Tensor

Initial conceptual shape:

```rust
#[derive(Clone)]
pub struct Tensor {
    desc: TensorDesc,
    storage: Storage,
    byte_offset: usize,
}
```

A cloned tensor shares storage. Cloning does not copy data.

Core invariants on construction:

1. `byte_offset` is within storage.
2. The byte span touched by the tensor fits in storage.
3. Required alignment is respected where the descriptor requires it.
4. Layout rank matches shape rank.
5. Encoding/layout combination is valid.
6. All size calculations are checked for overflow.

A tensor gets its device from storage; there should not be a second independently stored `device` field which can disagree with it.

## Views

Views are metadata transformations over shared storage.

Example:

```text
Storage
0 ------------------------------------------------------ N
                  ^
                  byte_offset
                  [ Tensor A ................ ]
                       [ Tensor B view ... ]
```

Important rules:

- `byte_offset` is always in bytes.
- A view does not allocate or copy.
- A view preserves the underlying physical encoding unless an explicit conversion op is run.
- Every view operation re-validates the resulting storage span.

### v0 view operations

Implement metadata semantics before every kernel supports them:

- contiguous reshape where legal;
- slice for plain encoding;
- transpose/permutation for plain encoding;
- narrow/select if useful;
- contiguous check.

This allows the data model to be correct while kernels can initially require contiguous inputs.

### Negative strides

Do not support negative strides in v0. They complicate span calculations and are not important for the initial inference workload. Add only if a real model/runtime path needs them.

### Quantized views

For v0, only whole/contiguous quantized tensors are required. Block-aligned slicing can be added once a concrete quantized kernel needs it.

## Mutation and aliasing

Do not try to encode GPU execution semantics through ordinary Rust `&mut Tensor` ownership.

A tensor may have many cloned/view aliases, and GPU work is asynchronous. Instead, mutation should be expressed by execution operations and access roles.

Conceptually:

```text
Kernel argument A: Read
Kernel argument B: Read
Kernel argument C: Write
KV cache:           ReadWrite
```

The scheduler/runtime is responsible for synchronization and hazard ordering.

Avoid exposing generic `&mut [T]`-style APIs from `Storage`. CPU-specific mapping APIs can later provide safe host access under stricter conditions.

For v0, it is enough that `Tensor` itself provides metadata, not mutation methods.

## Graph boundary

The graph should not allocate a `Tensor` for every intermediate value while being built.

Preferred flow:

```text
Graph build
   |
   | ValueId -> TensorDesc
   v
Graph
   |
   v
liveness / memory planning
   |
   | ValueId -> allocation slice
   v
ExecutionPlan
   |
   v
Storage arenas + Tensor views
```

Inputs, weights, and persistent state may already have physical tensors bound to graph values, while temporary activations remain purely logical until planning.

The exact graph representation is outside this document, but Tensor/Storage must not force eager allocation.

## Memory planner integration

The planner should eventually emit something conceptually like:

```rust
pub struct AllocationSlice {
    pub arena: ArenaId,
    pub byte_offset: usize,
    pub size_bytes: usize,
    pub alignment: usize,
}
```

Example lifetimes:

```text
time ->
A ███████
    B ███████████
             C █████
                  D ████
```

A and C/D may reuse physical regions if their lifetimes do not overlap and alignment/size constraints permit it.

At execution time, temporary tensors become views over one or a small number of large arena storages.

This is why Tensor must support shared storage and byte offsets from the beginning.

## Storage lifetime is policy, not tensor type

Weights, KV cache, activations, and external inputs do not need separate tensor classes.

They differ in allocation/lifetime policy:

```text
weights
  -> persistent storage, model lifetime

KV cache
  -> stateful storage, session lifetime

temporary activations
  -> reusable arena storage, execution lifetime

input/output
  -> owned or externally provided storage
```

The same `Tensor` type can represent all of them.

## mmap / model loading

The Storage/Tensor split should make mmap-backed model weights natural:

```text
GGUF file
   |
 mmap
   |
MmapStorage
   |-- weight A @ byte offset X
   |-- weight B @ byte offset Y
   `-- weight C @ byte offset Z
```

Weights can initially remain CPU/mmap-backed and later be uploaded/copied into GPU storage according to backend/model-loading policy.

GGUF should be a loader, not Mote's internal tensor model.

## Backend interop

Portable kernels and specialized kernels need to obtain backend-native resources without making those resources part of the public tensor model.

Possible direction:

```text
Tensor
  -> Storage
      -> dyn StorageImpl
          -> backend-specific downcast / adapter
              -> native resource handle
```

For example, a CUDA specialization may recognize `CudaStorage` (or a CubeCL CUDA storage adapter) and obtain the device pointer/stream-compatible resource it needs.

The exact native-handle API should be designed together with the first real CubeCL and backend-specialized kernel. Do not freeze it prematurely.

## Why not `Tensor<B: Backend>`

Avoid making backend identity a compile-time generic on every tensor.

Mote needs runtime-selected devices, mixed host/device resources, graph scheduling, model loading, and eventually cross-backend experimentation. A generic tensor tends to spread backend type parameters through graph/model/runtime APIs and makes heterogeneous runtime state awkward.

Prefer:

```text
Tensor
Storage
Device
```

as concrete runtime-dispatched types. Use static generics inside backend/kernel implementations where they actually help.

## Suggested source layout

```text
crates/mote-core/src/
  device.rs
  dtype.rs
  shape.rs
  layout.rs
  storage.rs
  tensor.rs
  lib.rs

# later
  graph.rs
  planner.rs
  execution.rs
```

Do not add all later modules before the tensor/storage invariants have tests.

## Implementation order

### Phase 1: metadata only

- [ ] `DType`
- [ ] `Encoding`
- [ ] `Shape`
- [ ] `Strides` / `Layout`
- [ ] checked size/span calculations
- [ ] contiguous-layout helpers

No GPU dependency required.

### Phase 2: CPU storage and tensor views

- [ ] `Device` / `BackendKind`
- [ ] `Storage` / `StorageImpl`
- [ ] `CpuOwnedStorage`
- [ ] `Tensor`
- [ ] storage bounds/alignment validation
- [ ] reshape
- [ ] slice/narrow
- [ ] transpose/permutation
- [ ] alias/storage identity helpers

Still no CubeCL dependency required.

### Phase 3: first device adapter

- [ ] pick one CubeCL backend available on the development machine
- [ ] implement/wrap device storage behind `StorageImpl`
- [ ] upload/download path sufficient for tests
- [ ] run one trivial portable kernel over Mote tensors

Do not add CUDA/HIP/Vulkan-specific native-handle APIs until this exposes what is actually needed.

### Phase 4: graph/planner

- [ ] logical graph values store `TensorDesc`
- [ ] liveness calculation
- [ ] arena allocation plan
- [ ] materialize execution tensors as storage views
- [ ] validate reuse/aliasing

### Phase 5: quantized model path

- [ ] implement the first actually-needed `QuantFormat`
- [ ] block geometry and physical-size rules
- [ ] mmap GGUF storage/view path
- [ ] quantized linear reference kernel

Avoid implementing a zoo of formats before a target model requires them.

## Minimum test checklist

### Shape / descriptor

- [ ] scalar / rank-1 / rank-N shapes
- [ ] zero-size dimensions: decide and test policy
- [ ] checked `numel` overflow
- [ ] contiguous physical size
- [ ] strided physical span
- [ ] invalid shape/layout rank

### Tensor / storage

- [ ] tensor exactly fills storage
- [ ] tensor begins at non-zero byte offset
- [ ] out-of-bounds construction rejected
- [ ] misaligned construction rejected when required
- [ ] cloned tensors share `StorageId`
- [ ] views share storage and adjust metadata only

### Views

- [ ] legal contiguous reshape
- [ ] illegal reshape rejected
- [ ] slice changes shape/offset correctly
- [ ] transpose changes shape/strides without copying
- [ ] chained view operations remain in bounds

### Quantization boundary

- [ ] quantized physical size is format-defined, not `numel * dtype_size`
- [ ] unsupported arbitrary quantized views are rejected explicitly

## Decisions to make now

These choices are structural and expensive to change later:

1. Tensor and Storage are separate.
2. Tensor offsets are bytes.
3. Storage is shared/type-erased at the core boundary.
4. Device is a runtime value.
5. Quantization is a physical encoding, not a scalar dtype.
6. Views exist in the data model from v0.
7. Graph intermediates do not imply eager allocations.
8. GPU mutation/synchronization is runtime responsibility, not ordinary Rust aliasing semantics.

## Decisions to defer

Do not over-design these until a real kernel/model forces the question:

- exact CubeCL storage adapter API;
- raw CUDA/HIP/Vulkan native-handle escape hatch;
- external-memory/DLPack ownership semantics;
- quantization format trait vs enum shape;
- arbitrary block-quantized views;
- negative strides;
- multi-device graph scheduling;
- asynchronous host mapping;
- storage pooling implementation;
- device capability schema;
- specialized layout taxonomy beyond contiguous/strided.

## First concrete milestone

The tensor layer is ready for the next stage when this can be demonstrated without GPU code:

```text
1. allocate one CPU Storage arena
2. create multiple Tensor views at different byte offsets
3. reshape/slice/transpose plain tensors without copying
4. correctly reject invalid/out-of-bounds views
5. prove aliases share one Storage identity
6. describe a quantized tensor whose physical size is not scalar `numel * sizeof(T)`
```

After that, the next useful step is not more tensor abstraction. It is attaching one CubeCL-backed storage/kernel path and making one real operator consume these tensors.

// Fused residual-add + RMSNorm HIP kernel, compiled to an object by `hipcc`
// in build.rs.
//
// One block of 256 threads normalizes one row. Input, residual, weight, and
// the normalized output are F16 while all arithmetic happens in F32. Each
// thread builds a strided partial sum of squares over the F32 sums
// `input[i] + residual[i]`, so hidden sizes that are not multiples of the
// block size (e.g. 513) are handled exactly; the partials are then combined
// through a shared-memory tree reduction.
//
// `residual` is in/out: the statistics pass only reads it, and a second pass
// overwrites it with the F16-rounded sums only after the statistics are
// final, re-reading the untouched `input`/`residual` so both passes see the
// exact same un-quantized F32 sums. `input`, `residual`, and `output`
// deliberately carry no `__restrict__`: input/output may alias (in-place
// norm) and residual is written after being read, so claiming exclusive
// access would violate the restrict alias contract. `weight` is never
// written and keeps its read-only `__restrict__`.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr int kInternalError = -1;
// dim3 grid.x is a 32-bit unsigned quantity on every current HIP runtime.
constexpr std::uint64_t kMaxGridX = 0xffffffffULL;

__global__ void fused_add_rms_norm_kernel(const __half* input,
                                          __half* residual,
                                          const __half* __restrict__ weight,
                                          __half* output,
                                          std::size_t hidden_size,
                                          float epsilon) {
    __shared__ float partials[kThreadsPerBlock];

    const std::size_t row = blockIdx.x;
    const __half* row_input = input + row * hidden_size;
    __half* row_residual = residual + row * hidden_size;
    __half* row_output = output + row * hidden_size;

    // Pass 1: strided F32 accumulation of (input + residual)^2. residual is
    // only read here; it must stay untouched until the statistics are final
    // so pass 2 can reproduce the exact same F32 sums.
    float partial = 0.0F;
    for (std::size_t i = threadIdx.x; i < hidden_size; i += kThreadsPerBlock) {
        const float sum =
            __half2float(row_input[i]) + __half2float(row_residual[i]);
        partial += sum * sum;
    }
    partials[threadIdx.x] = partial;
    __syncthreads();

    // Shared-memory tree reduction over the full block.
    for (unsigned int stride = kThreadsPerBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partials[threadIdx.x] += partials[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float mean_square = partials[0] / static_cast<float>(hidden_size);
    const float inverse_rms = rsqrtf(mean_square + epsilon);

    // Pass 2: re-read the untouched inputs so the F32 sums below are
    // bit-identical to the ones that fed the statistics (never the rounded
    // residual written here), then publish the normalized row and the
    // updated residual. Every element index is owned by exactly one thread
    // in both passes, all reads of an iteration precede its writes, and the
    // __syncthreads() calls inside the reduction separate the passes, so
    // aliasing input with output or residual stays race-free.
    for (std::size_t i = threadIdx.x; i < hidden_size; i += kThreadsPerBlock) {
        const float sum =
            __half2float(row_input[i]) + __half2float(row_residual[i]);
        const float scale = __half2float(weight[i]);
        row_output[i] = __float2half(sum * inverse_rms * scale);
        row_residual[i] = __float2half(sum);
    }
}

}  // namespace

// Fused residual-add + RMSNorm launch shim exported to Rust.
//
// The Rust side never includes HIP headers or replicates their ABI: pointers
// are untyped, the stream stays an opaque `void*` handle, and the raw
// `hipError_t` status is returned as an `int` where `0` means `hipSuccess`.
// `input` and `output` may alias, but `output` must not alias `residual`
// (one location cannot hold both the normalized row and the updated
// residual); the upper layer rejects that pairing and this shim defends
// independently. A non-finite or negative epsilon is rejected before anything
// is enqueued; non-positive sizes launch nothing and report success.

extern "C" int mote_hip_fused_add_rms_norm_f16(const void* input,
                                               void* residual,
                                               const void* weight,
                                               void* output,
                                               std::uint64_t rows,
                                               std::uint64_t hidden_size,
                                               float epsilon,
                                               void* stream) noexcept {
    if (input == nullptr || residual == nullptr || weight == nullptr ||
        output == nullptr || stream == nullptr) {
        return kInternalError;
    }
    if (!std::isfinite(epsilon) || epsilon < 0.0f) {
        return kInternalError;
    }
    if (output == residual) {
        return kInternalError;
    }
    if (rows == 0 || hidden_size == 0) {
        return 0;
    }
    if (rows > kMaxGridX) {
        return kInternalError;
    }

    const dim3 grid(static_cast<unsigned int>(rows), 1, 1);
    const dim3 block(kThreadsPerBlock, 1, 1);
    hipLaunchKernelGGL(fused_add_rms_norm_kernel,
                       grid,
                       block,
                       0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const __half*>(input),
                       static_cast<__half*>(residual),
                       static_cast<const __half*>(weight),
                       static_cast<__half*>(output),
                       static_cast<std::size_t>(hidden_size),
                       epsilon);
    return static_cast<int>(hipGetLastError());
}

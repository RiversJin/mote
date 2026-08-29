// RMSNorm HIP kernel, compiled to an object by `hipcc` in build.rs.
//
// One block normalizes one row with 256 threads. Input, weight, and output are
// F16 while all accumulation happens in F32. Each thread builds a strided
// partial sum of squares, so hidden sizes that are not multiples of the block
// size (e.g. 513) are handled exactly; the partials are then combined through
// a shared-memory tree reduction.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr int kInternalError = -1;
// dim3 grid.x is a 32-bit unsigned quantity on every current HIP runtime.
constexpr std::uint64_t kMaxGridX = 0xffffffffULL;

__global__ void rms_norm_kernel(const __half* __restrict__ input,
                                const __half* __restrict__ weight,
                                __half* __restrict__ output,
                                std::size_t hidden_size,
                                float epsilon) {
    __shared__ float partials[kThreadsPerBlock];

    const std::size_t row = blockIdx.x;
    const __half* row_input = input + row * hidden_size;
    __half* row_output = output + row * hidden_size;

    // Strided F32 accumulation: thread t covers t, t + 256, t + 512, ...
    float partial = 0.0F;
    for (std::size_t i = threadIdx.x; i < hidden_size; i += kThreadsPerBlock) {
        const float value = __half2float(row_input[i]);
        partial += value * value;
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

    for (std::size_t i = threadIdx.x; i < hidden_size; i += kThreadsPerBlock) {
        const float value = __half2float(row_input[i]);
        const float scale = __half2float(weight[i]);
        row_output[i] = __float2half(value * inverse_rms * scale);
    }
}

}  // namespace

// RMSNorm launch shim exported to Rust.
//
// The Rust side never includes HIP headers or replicates their ABI: pointers
// are untyped, the stream stays an opaque `void*` handle, and the raw
// `hipError_t` status is returned as an `int` where `0` means `hipSuccess`.

extern "C" int mote_hip_rms_norm_f16(const void* input,
                                     const void* weight,
                                     void* output,
                                     std::uint64_t rows,
                                     std::uint64_t hidden_size,
                                     float epsilon,
                                     void* stream) noexcept {
    if (input == nullptr || weight == nullptr || output == nullptr || stream == nullptr) {
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
    hipLaunchKernelGGL(rms_norm_kernel,
                       grid,
                       block,
                       0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const __half*>(input),
                       static_cast<const __half*>(weight),
                       static_cast<__half*>(output),
                       static_cast<std::size_t>(hidden_size),
                       epsilon);
    return static_cast<int>(hipGetLastError());
}

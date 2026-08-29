// Rotary position embedding (RoPE) HIP kernel, compiled by `hipcc` in
// build.rs.
//
// One block rotates one (token, head) vector with 256 threads striding along
// the head dimension. Input and output are F16 while cos/sin are F32 rows of
// `rotary_dim / 2` values per token; the leading `rotary_dim` dimensions are
// rotated in F32 and any tail (`rotary_dim < head_dim`) is copied through
// unchanged. Both pairing conventions shared with `mote_types::RopeLayout`
// are supported, selected by an explicit integer selector; the launch shim
// below rejects unknown selectors before anything is enqueued.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr int kInternalError = -1;
// dim3 grid.x is a 32-bit unsigned quantity on every current HIP runtime.
constexpr std::uint64_t kMaxGridX = 0xffffffffULL;

// Pairing conventions for the rotary prefix, mirroring
// `mote_types::RopeLayout` one level up in Rust.
constexpr std::uint32_t kLayoutHalfSplit = 0;
constexpr std::uint32_t kLayoutInterleaved = 1;

// `input` and `output` deliberately carry no `__restrict__`: the public
// API and GPU tests support in-place operation (input == output), so
// claiming exclusive access would violate the restrict alias contract
// even though the pair-index scheme below has no data races. cos/sin are
// never written and keep their read-only `__restrict__`.
__global__ void rope_kernel(const __half* input,
                            const float* __restrict__ cos,
                            const float* __restrict__ sin,
                            __half* output,
                            std::size_t heads,
                            std::size_t head_dim,
                            std::size_t rotary_dim,
                            std::size_t frequency_count,
                            std::uint32_t layout) {
    // Vectors are token-major rows of `heads` consecutive vectors; each block
    // owns exactly one, so cos/sin are read from this token's cache row.
    const std::size_t vector = blockIdx.x;
    const std::size_t token = vector / heads;
    const __half* vector_input = input + vector * head_dim;
    __half* vector_output = output + vector * head_dim;
    const float* token_cos = cos + token * frequency_count;
    const float* token_sin = sin + token * frequency_count;

    // Pair-indexed strided pass: thread t owns whole rotation pairs t,
    // t + 256, t + 512, ..., so both reads of a pair's inputs and both
    // writes of its outputs come from the same thread. No thread ever
    // reads an element another thread writes, which makes in-place
    // (input == output) operation race-free without shared memory.
    for (std::size_t j = threadIdx.x; j < frequency_count;
         j += kThreadsPerBlock) {
        std::size_t first;
        std::size_t second;
        if (layout == kLayoutHalfSplit) {
            // Pair (j, j + rotary_dim / 2).
            first = j;
            second = j + frequency_count;
        } else {  // kLayoutInterleaved: pair (2 * j, 2 * j + 1).
            first = 2 * j;
            second = 2 * j + 1;
        }

        const float cosine = token_cos[j];
        const float sine = token_sin[j];
        const float lo = __half2float(vector_input[first]);
        const float hi = __half2float(vector_input[second]);
        vector_output[first] = __float2half(lo * cosine - hi * sine);
        vector_output[second] = __float2half(hi * cosine + lo * sine);
    }

    // The non-rotary tail is a plain element-wise passthrough; each
    // element is read and written only by the single thread that copies
    // it, so in-place operation stays race-free here as well.
    for (std::size_t i = threadIdx.x + rotary_dim; i < head_dim;
         i += kThreadsPerBlock) {
        vector_output[i] = vector_input[i];
    }
}

}  // namespace

// RoPE launch shim exported to Rust.
//
// The Rust side never includes HIP headers or replicates their ABI: pointers
// are untyped, the stream stays an opaque `void*` handle, and the raw
// `hipError_t` status is returned as an `int` where `0` means `hipSuccess`.
// `layout` must be one of the selectors above (unknown values), an odd or
// oversized `rotary_dim`, and a grid too large for `grid.x` are all rejected
// before anything is enqueued.

extern "C" int mote_hip_rope_f16(const void* input,
                                 const void* cos,
                                 const void* sin,
                                 void* output,
                                 std::uint64_t tokens,
                                 std::uint64_t heads,
                                 std::uint64_t head_dim,
                                 std::uint64_t rotary_dim,
                                 std::uint32_t layout,
                                 void* stream) noexcept {
    if (input == nullptr || output == nullptr || stream == nullptr) {
        return kInternalError;
    }
    // cos/sin are only dereferenced for the rotary prefix; a zero rotary
    // dimension degenerates to a pure passthrough copy, so zero-length (null)
    // cache pointers are tolerated there.
    if (rotary_dim > 0 && (cos == nullptr || sin == nullptr)) {
        return kInternalError;
    }
    if (layout != kLayoutHalfSplit && layout != kLayoutInterleaved) {
        return kInternalError;
    }
    if ((rotary_dim & 1U) != 0 || rotary_dim > head_dim) {
        return kInternalError;
    }
    if (tokens == 0 || heads == 0 || head_dim == 0) {
        return 0;
    }
    // Also guards the multiplication below against 64-bit overflow.
    if (tokens > kMaxGridX / heads) {
        return kInternalError;
    }
    const std::uint64_t vectors = tokens * heads;

    const dim3 grid(static_cast<unsigned int>(vectors), 1, 1);
    const dim3 block(kThreadsPerBlock, 1, 1);
    hipLaunchKernelGGL(rope_kernel,
                       grid,
                       block,
                       0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const __half*>(input),
                       static_cast<const float*>(cos),
                       static_cast<const float*>(sin),
                       static_cast<__half*>(output),
                       static_cast<std::size_t>(heads),
                       static_cast<std::size_t>(head_dim),
                       static_cast<std::size_t>(rotary_dim),
                       static_cast<std::size_t>(rotary_dim / 2),
                       layout);
    return static_cast<int>(hipGetLastError());
}

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <climits>
#include <cstdint>

namespace {

constexpr unsigned int kThreads = 256;
constexpr std::uint64_t kBlockElements = 32;
constexpr std::uint64_t kBlockBytes = 18;

__global__ void quantized_linear_q4_0_f16_kernel(
    const half* input,
    const unsigned char* weights,
    half* output,
    std::uint64_t output_features,
    std::uint64_t input_features) {
    const std::uint64_t output_index = blockIdx.x;
    const std::uint64_t row = output_index / output_features;
    const std::uint64_t output_feature = output_index % output_features;
    const std::uint64_t weight_row_bytes =
        (input_features / kBlockElements) * kBlockBytes;
    const half* input_row = input + row * input_features;
    const unsigned char* weight_row = weights + output_feature * weight_row_bytes;

    float partial = 0.0f;
    for (std::uint64_t inner = threadIdx.x; inner < input_features;
         inner += blockDim.x) {
        const std::uint64_t block_index = inner / kBlockElements;
        const unsigned int within_block = inner % kBlockElements;
        const unsigned char* block = weight_row + block_index * kBlockBytes;
        const float scale = __half2float(*reinterpret_cast<const half*>(block));
        const unsigned char packed = block[2 + (within_block & 15u)];
        const int quant = within_block < 16 ? packed & 0x0f : packed >> 4;
        const float weight = static_cast<float>(quant - 8) * scale;
        partial += __half2float(input_row[inner]) * weight;
    }

    __shared__ float reduction[kThreads];
    reduction[threadIdx.x] = partial;
    __syncthreads();
    for (unsigned int stride = kThreads / 2; stride != 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[output_index] = __float2half(reduction[0]);
    }
}

}  // namespace

extern "C" int mote_hip_quantized_linear_q4_0_f16(
    const void* input,
    const void* weights,
    void* output,
    std::uint64_t rows,
    std::uint64_t output_features,
    std::uint64_t input_features,
    void* stream) {
    if (input == nullptr || weights == nullptr || output == nullptr || stream == nullptr) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    if (input_features % kBlockElements != 0) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    if (rows == 0 || output_features == 0) {
        return static_cast<int>(hipSuccess);
    }
    if (rows > UINT_MAX || output_features > UINT_MAX ||
        rows > UINT_MAX / output_features) {
        return static_cast<int>(hipErrorInvalidConfiguration);
    }

    const unsigned int grid = static_cast<unsigned int>(rows * output_features);
    hipLaunchKernelGGL(
        quantized_linear_q4_0_f16_kernel,
        dim3(grid),
        dim3(kThreads),
        0,
        reinterpret_cast<hipStream_t>(stream),
        static_cast<const half*>(input),
        static_cast<const unsigned char*>(weights),
        static_cast<half*>(output),
        output_features,
        input_features);
    return static_cast<int>(hipGetLastError());
}

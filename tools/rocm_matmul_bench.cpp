#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string_view>
#include <vector>

namespace {

constexpr std::array<int, 4> kSizes{256, 512, 1024, 2048};
constexpr int kWarmups = 5;
constexpr int kSamples = 7;
constexpr std::size_t kWorkspaceBytes = 32 * 1024 * 1024;
constexpr int kHeuristicCount = 32;
constexpr double kMiB = 1024.0 * 1024.0;

[[noreturn]] void fail(std::string_view operation, int status) {
    std::cerr << operation << " failed with status " << status << '\n';
    std::exit(EXIT_FAILURE);
}

void check_hip(hipError_t status, std::string_view operation) {
    if (status != hipSuccess) {
        std::cerr << operation << " failed: " << hipGetErrorString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void check_rocblas(rocblas_status status, std::string_view operation) {
    if (status != rocblas_status_success) {
        fail(operation, static_cast<int>(status));
    }
}

void check_hipblaslt(hipblasStatus_t status, std::string_view operation) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        fail(operation, static_cast<int>(status));
    }
}

std::size_t free_device_bytes() {
    std::size_t free = 0;
    std::size_t total = 0;
    check_hip(hipMemGetInfo(&free, &total), "hipMemGetInfo");
    return free;
}

double used_mib_since(std::size_t baseline_free) {
    const auto current_free = free_device_bytes();
    return static_cast<double>(baseline_free > current_free ? baseline_free - current_free : 0)
         / kMiB;
}

float lhs_value(int row, int col) {
    return static_cast<float>((row * 17 + col * 13) % 23 - 11) / 16.0F;
}

float rhs_value(int row, int col) {
    return static_cast<float>((row * 7 + col * 19) % 29 - 14) / 16.0F;
}

float as_float(rocblas_half value) {
    return static_cast<float>(value);
}

void launch_gemm(rocblas_handle handle,
                 int size,
                 const rocblas_half* lhs,
                 const rocblas_half* rhs,
                 float* output) {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    check_rocblas(
        rocblas_gemm_ex(handle,
                        rocblas_operation_none,
                        rocblas_operation_none,
                        size,
                        size,
                        size,
                        &alpha,
                        lhs,
                        rocblas_datatype_f16_r,
                        size,
                        rhs,
                        rocblas_datatype_f16_r,
                        size,
                        &beta,
                        output,
                        rocblas_datatype_f32_r,
                        size,
                        output,
                        rocblas_datatype_f32_r,
                        size,
                        rocblas_datatype_f32_r,
                        rocblas_gemm_algo_standard,
                        0,
                        0),
        "rocblas_gemm_ex");
}

struct LtProblem {
    hipblasLtMatmulDesc_t operation{};
    hipblasLtMatrixLayout_t lhs{};
    hipblasLtMatrixLayout_t rhs{};
    hipblasLtMatrixLayout_t output{};
    hipblasLtMatmulAlgo_t algorithm{};
};

hipblasStatus_t launch_lt(hipblasLtHandle_t handle,
                          const LtProblem& problem,
                          const rocblas_half* lhs,
                          const rocblas_half* rhs,
                          float* output,
                          void* workspace,
                          hipStream_t stream) {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    return hipblasLtMatmul(handle,
                          problem.operation,
                          &alpha,
                          lhs,
                          problem.lhs,
                          rhs,
                          problem.rhs,
                          &beta,
                          output,
                          problem.output,
                          output,
                          problem.output,
                          &problem.algorithm,
                          workspace,
                          kWorkspaceBytes,
                          stream);
}

LtProblem create_lt_problem(hipblasLtHandle_t handle,
                            int size,
                            const rocblas_half* lhs,
                            const rocblas_half* rhs,
                            float* output,
                            void* workspace,
                            hipStream_t stream) {
    LtProblem problem;
    check_hipblaslt(
        hipblasLtMatmulDescCreate(&problem.operation, HIPBLAS_COMPUTE_32F, HIP_R_32F),
        "hipblasLtMatmulDescCreate");
    check_hipblaslt(
        hipblasLtMatrixLayoutCreate(&problem.lhs, HIP_R_16F, size, size, size),
        "hipblasLtMatrixLayoutCreate(lhs)");
    check_hipblaslt(
        hipblasLtMatrixLayoutCreate(&problem.rhs, HIP_R_16F, size, size, size),
        "hipblasLtMatrixLayoutCreate(rhs)");
    check_hipblaslt(
        hipblasLtMatrixLayoutCreate(&problem.output, HIP_R_32F, size, size, size),
        "hipblasLtMatrixLayoutCreate(output)");

    hipblasLtMatmulPreference_t preference{};
    check_hipblaslt(hipblasLtMatmulPreferenceCreate(&preference),
                    "hipblasLtMatmulPreferenceCreate");
    const std::uint64_t workspace_bytes = kWorkspaceBytes;
    check_hipblaslt(
        hipblasLtMatmulPreferenceSetAttribute(preference,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &workspace_bytes,
                                              sizeof(workspace_bytes)),
        "hipblasLtMatmulPreferenceSetAttribute");
    std::array<hipblasLtMatmulHeuristicResult_t, kHeuristicCount> candidates{};
    int candidate_count = 0;
    check_hipblaslt(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                    problem.operation,
                                                    problem.lhs,
                                                    problem.rhs,
                                                    problem.output,
                                                    problem.output,
                                                    preference,
                                                    candidates.size(),
                                                    candidates.data(),
                                                    &candidate_count),
                    "hipblasLtMatmulAlgoGetHeuristic");
    check_hipblaslt(hipblasLtMatmulPreferenceDestroy(preference),
                    "hipblasLtMatmulPreferenceDestroy");
    if (candidate_count == 0) {
        fail("hipblasLtMatmulAlgoGetHeuristic returned no algorithms", 0);
    }

    float best_milliseconds = std::numeric_limits<float>::infinity();
    hipblasLtMatmulAlgo_t best_algorithm{};
    hipEvent_t start{};
    hipEvent_t stop{};
    check_hip(hipEventCreate(&start), "hipEventCreate(tuning start)");
    check_hip(hipEventCreate(&stop), "hipEventCreate(tuning stop)");
    for (int index = 0; index < candidate_count; ++index) {
        if (candidates[index].state != HIPBLAS_STATUS_SUCCESS
            || candidates[index].workspaceSize > kWorkspaceBytes) {
            continue;
        }
        problem.algorithm = candidates[index].algo;
        if (launch_lt(handle, problem, lhs, rhs, output, workspace, stream)
            != HIPBLAS_STATUS_SUCCESS) {
            continue;
        }
        check_hip(hipEventRecord(start, stream), "hipEventRecord(tuning start)");
        const auto status = launch_lt(handle, problem, lhs, rhs, output, workspace, stream);
        check_hip(hipEventRecord(stop, stream), "hipEventRecord(tuning stop)");
        check_hip(hipEventSynchronize(stop), "hipEventSynchronize(tuning stop)");
        if (status != HIPBLAS_STATUS_SUCCESS) {
            continue;
        }
        float milliseconds = 0.0F;
        check_hip(hipEventElapsedTime(&milliseconds, start, stop),
                  "hipEventElapsedTime(tuning)");
        if (milliseconds < best_milliseconds) {
            best_milliseconds = milliseconds;
            best_algorithm = candidates[index].algo;
        }
    }
    check_hip(hipEventDestroy(start), "hipEventDestroy(tuning start)");
    check_hip(hipEventDestroy(stop), "hipEventDestroy(tuning stop)");
    if (!std::isfinite(best_milliseconds)) {
        fail("no hipBLASLt heuristic algorithm launched successfully", 0);
    }
    problem.algorithm = best_algorithm;
    return problem;
}

void destroy_lt_problem(const LtProblem& problem) {
    check_hipblaslt(hipblasLtMatrixLayoutDestroy(problem.lhs),
                    "hipblasLtMatrixLayoutDestroy(lhs)");
    check_hipblaslt(hipblasLtMatrixLayoutDestroy(problem.rhs),
                    "hipblasLtMatrixLayoutDestroy(rhs)");
    check_hipblaslt(hipblasLtMatrixLayoutDestroy(problem.output),
                    "hipblasLtMatrixLayoutDestroy(output)");
    check_hipblaslt(hipblasLtMatmulDescDestroy(problem.operation),
                    "hipblasLtMatmulDescDestroy");
}

void validate(int size,
              const std::vector<rocblas_half>& lhs,
              const std::vector<rocblas_half>& rhs,
              const std::vector<float>& output) {
    const std::array<std::array<int, 2>, 5> positions{{
        {0, 0},
        {size / 7, size / 5},
        {size / 2, size / 3},
        {size - 2, size - 3},
        {size - 1, size - 1},
    }};
    for (const auto [row, col] : positions) {
        float expected = 0.0F;
        for (int inner = 0; inner < size; ++inner) {
            expected += as_float(lhs[row + inner * size])
                      * as_float(rhs[inner + col * size]);
        }
        const float actual = output[row + col * size];
        const float tolerance = std::max(0.05F, std::abs(expected) * 0.002F);
        if (std::abs(expected - actual) > tolerance) {
            std::cerr << "validation failed for " << size << " at (" << row << ", " << col
                      << "): expected " << expected << ", got " << actual << '\n';
            std::exit(EXIT_FAILURE);
        }
    }
}

void readback_and_validate(hipStream_t stream,
                           int size,
                           const std::vector<rocblas_half>& lhs,
                           const std::vector<rocblas_half>& rhs,
                           float* device_output,
                           std::vector<float>& output,
                           std::string_view implementation) {
    check_hip(hipMemcpyAsync(output.data(),
                             device_output,
                             output.size() * sizeof(output.front()),
                             hipMemcpyDeviceToHost,
                             stream),
              "hipMemcpyAsync(output)");
    check_hip(hipStreamSynchronize(stream), "readback synchronization");
    validate(size, lhs, rhs, output);
    std::clog << implementation << " " << size << "x" << size << " validation passed\n";
}

}  // namespace

int main() {
    int device = 0;
    hipDeviceProp_t properties{};
    check_hip(hipGetDevice(&device), "hipGetDevice");
    check_hip(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties");
    const auto baseline_free = free_device_bytes();

    hipStream_t stream{};
    hipEvent_t start{};
    hipEvent_t stop{};
    rocblas_handle handle{};
    hipblasLtHandle_t lt_handle{};
    void* workspace{};
    check_hip(hipStreamCreate(&stream), "hipStreamCreate");
    check_hip(hipEventCreate(&start), "hipEventCreate(start)");
    check_hip(hipEventCreate(&stop), "hipEventCreate(stop)");
    check_rocblas(rocblas_create_handle(&handle), "rocblas_create_handle");
    check_rocblas(rocblas_set_stream(handle, stream), "rocblas_set_stream");
    check_hipblaslt(hipblasLtCreate(&lt_handle), "hipblasLtCreate");

    std::cout << "device: " << properties.name << '\n';
    std::cout << "rocBLAS/hipBLASLt: F16 inputs, F32 accumulation/output\n";
    std::cout << "samples per case: " << kSamples << " (median reported)\n\n";
    std::cout << std::setw(8) << "M=N=K" << std::setw(13) << "rocBLAS us"
              << std::setw(11) << "TFLOP/s" << std::setw(15) << "VRAM MiB"
              << std::setw(15) << "hipBLASLt us" << std::setw(11) << "TFLOP/s"
              << std::setw(15) << "VRAM MiB" << '\n';

    for (const int size : kSizes) {
        const std::size_t elements = static_cast<std::size_t>(size) * size;
        std::vector<rocblas_half> lhs(elements);
        std::vector<rocblas_half> rhs(elements);
        std::vector<float> output(elements);
        for (int col = 0; col < size; ++col) {
            for (int row = 0; row < size; ++row) {
                lhs[row + col * size] = static_cast<rocblas_half>(lhs_value(row, col));
                rhs[row + col * size] = static_cast<rocblas_half>(rhs_value(row, col));
            }
        }

        rocblas_half* device_lhs{};
        rocblas_half* device_rhs{};
        float* device_output{};
        check_hip(hipMalloc(&device_lhs, elements * sizeof(*device_lhs)), "hipMalloc(lhs)");
        check_hip(hipMalloc(&device_rhs, elements * sizeof(*device_rhs)), "hipMalloc(rhs)");
        check_hip(hipMalloc(&device_output, elements * sizeof(*device_output)), "hipMalloc(output)");
        check_hip(hipMemcpyAsync(device_lhs,
                                 lhs.data(),
                                 elements * sizeof(*device_lhs),
                                 hipMemcpyHostToDevice,
                                 stream),
                  "hipMemcpyAsync(lhs)");
        check_hip(hipMemcpyAsync(device_rhs,
                                 rhs.data(),
                                 elements * sizeof(*device_rhs),
                                 hipMemcpyHostToDevice,
                                 stream),
                  "hipMemcpyAsync(rhs)");

        for (int i = 0; i < kWarmups; ++i) {
            launch_gemm(handle, size, device_lhs, device_rhs, device_output);
        }
        check_hip(hipStreamSynchronize(stream), "warmup synchronization");

        std::vector<float> samples;
        samples.reserve(kSamples);
        for (int i = 0; i < kSamples; ++i) {
            check_hip(hipEventRecord(start, stream), "hipEventRecord(start)");
            launch_gemm(handle, size, device_lhs, device_rhs, device_output);
            check_hip(hipEventRecord(stop, stream), "hipEventRecord(stop)");
            check_hip(hipEventSynchronize(stop), "hipEventSynchronize(stop)");
            float milliseconds = 0.0F;
            check_hip(hipEventElapsedTime(&milliseconds, start, stop), "hipEventElapsedTime");
            samples.push_back(milliseconds);
        }
        std::sort(samples.begin(), samples.end());
        const double milliseconds = samples[samples.size() / 2];
        const double operations = 2.0 * size * size * size;
        const double tflops = operations / (milliseconds / 1000.0) / 1.0e12;
        const double rocblas_vram_mib = used_mib_since(baseline_free);
        readback_and_validate(stream,
                              size,
                              lhs,
                              rhs,
                              device_output,
                              output,
                              "rocBLAS");

        check_hip(hipMalloc(&workspace, kWorkspaceBytes), "hipMalloc(hipBLASLt workspace)");
        const auto lt_problem = create_lt_problem(lt_handle,
                                                  size,
                                                  device_lhs,
                                                  device_rhs,
                                                  device_output,
                                                  workspace,
                                                  stream);
        for (int i = 0; i < kWarmups; ++i) {
            check_hipblaslt(launch_lt(lt_handle,
                                      lt_problem,
                                      device_lhs,
                                      device_rhs,
                                      device_output,
                                      workspace,
                                      stream),
                            "hipblasLtMatmul(warmup)");
        }
        check_hip(hipStreamSynchronize(stream), "hipBLASLt warmup synchronization");
        samples.clear();
        for (int i = 0; i < kSamples; ++i) {
            check_hip(hipEventRecord(start, stream), "hipEventRecord(hipBLASLt start)");
            check_hipblaslt(launch_lt(lt_handle,
                                      lt_problem,
                                      device_lhs,
                                      device_rhs,
                                      device_output,
                                      workspace,
                                      stream),
                            "hipblasLtMatmul");
            check_hip(hipEventRecord(stop, stream), "hipEventRecord(hipBLASLt stop)");
            check_hip(hipEventSynchronize(stop), "hipEventSynchronize(hipBLASLt stop)");
            float lt_milliseconds = 0.0F;
            check_hip(hipEventElapsedTime(&lt_milliseconds, start, stop),
                      "hipEventElapsedTime(hipBLASLt)");
            samples.push_back(lt_milliseconds);
        }
        std::sort(samples.begin(), samples.end());
        const double lt_milliseconds = samples[samples.size() / 2];
        const double lt_tflops = operations / (lt_milliseconds / 1000.0) / 1.0e12;
        const double lt_vram_mib = used_mib_since(baseline_free);

        readback_and_validate(stream,
                              size,
                              lhs,
                              rhs,
                              device_output,
                              output,
                              "hipBLASLt");

        std::cout << std::setw(8) << size << std::fixed << std::setprecision(3)
                  << std::setw(13) << milliseconds * 1000.0 << std::setprecision(2)
                  << std::setw(11) << tflops << std::setw(15) << rocblas_vram_mib
                  << std::setprecision(3) << std::setw(15) << lt_milliseconds * 1000.0
                  << std::setprecision(2) << std::setw(11) << lt_tflops << std::setw(15)
                  << lt_vram_mib << '\n';

        destroy_lt_problem(lt_problem);
        check_hip(hipFree(workspace), "hipFree(hipBLASLt workspace)");
        workspace = nullptr;
        check_hip(hipFree(device_lhs), "hipFree(lhs)");
        check_hip(hipFree(device_rhs), "hipFree(rhs)");
        check_hip(hipFree(device_output), "hipFree(output)");
    }

    check_hipblaslt(hipblasLtDestroy(lt_handle), "hipblasLtDestroy");
    check_rocblas(rocblas_destroy_handle(handle), "rocblas_destroy_handle");
    check_hip(hipEventDestroy(start), "hipEventDestroy(start)");
    check_hip(hipEventDestroy(stop), "hipEventDestroy(stop)");
    check_hip(hipStreamDestroy(stream), "hipStreamDestroy");
}

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <utility>

namespace {

constexpr int kHeuristicCount = 32;
constexpr int kInternalError = -1;

struct Plan {
    hipblasLtMatmulDesc_t operation{};
    hipblasLtMatrixLayout_t rhs{};
    hipblasLtMatrixLayout_t lhs{};
    hipblasLtMatrixLayout_t output{};
    hipblasLtMatmulAlgo_t algorithm{};

    ~Plan() {
        if (rhs != nullptr) {
            hipblasLtMatrixLayoutDestroy(rhs);
        }
        if (lhs != nullptr) {
            hipblasLtMatrixLayoutDestroy(lhs);
        }
        if (output != nullptr) {
            hipblasLtMatrixLayoutDestroy(output);
        }
        if (operation != nullptr) {
            hipblasLtMatmulDescDestroy(operation);
        }
    }
};

struct Context {
    hipblasLtHandle_t handle{};
    std::map<std::array<std::uint64_t, 3>, std::unique_ptr<Plan>> plans;
    std::string last_error;

    ~Context() {
        plans.clear();
        if (handle != nullptr) {
            hipblasLtDestroy(handle);
        }
    }
};

int fail(Context* context, int status, std::string message) {
    if (context != nullptr) {
        context->last_error = std::move(message);
    }
    return status == 0 ? kInternalError : status;
}

int fail_hip(Context* context, hipError_t status, const char* operation) {
    return fail(context,
                -1000 - static_cast<int>(status),
                std::string(operation) + " failed: " + hipGetErrorString(status));
}

hipblasStatus_t launch(hipblasLtHandle_t handle,
                       const Plan& plan,
                       const void* lhs,
                       const void* rhs,
                       void* output,
                       void* workspace,
                       std::size_t workspace_bytes,
                       hipStream_t stream) {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;

    // Mote tensors are row-major. Interpreting them as column-major transposes
    // each matrix, so evaluate C^T = B^T * A^T and expose C as row-major.
    return hipblasLtMatmul(handle,
                          plan.operation,
                          &alpha,
                          rhs,
                          plan.rhs,
                          lhs,
                          plan.lhs,
                          &beta,
                          output,
                          plan.output,
                          output,
                          plan.output,
                          &plan.algorithm,
                          workspace,
                          workspace_bytes,
                          stream);
}

int create_plan(Context* context,
                std::uint64_t m,
                std::uint64_t n,
                std::uint64_t k,
                const void* lhs,
                const void* rhs,
                void* output,
                void* workspace,
                std::size_t workspace_bytes,
                hipStream_t stream,
                std::unique_ptr<Plan>& result) {
    auto plan = std::make_unique<Plan>();
    auto status = hipblasLtMatmulDescCreate(
        &plan->operation, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        return fail(context, static_cast<int>(status), "hipblasLtMatmulDescCreate failed");
    }
    status = hipblasLtMatrixLayoutCreate(&plan->rhs, HIP_R_16F, n, k, n);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        return fail(context,
                    static_cast<int>(status),
                    "hipblasLtMatrixLayoutCreate(rhs) failed");
    }
    status = hipblasLtMatrixLayoutCreate(&plan->lhs, HIP_R_16F, k, m, k);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        return fail(context,
                    static_cast<int>(status),
                    "hipblasLtMatrixLayoutCreate(lhs) failed");
    }
    status = hipblasLtMatrixLayoutCreate(&plan->output, HIP_R_32F, n, m, n);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        return fail(context,
                    static_cast<int>(status),
                    "hipblasLtMatrixLayoutCreate(output) failed");
    }

    hipblasLtMatmulPreference_t preference{};
    status = hipblasLtMatmulPreferenceCreate(&preference);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        return fail(context,
                    static_cast<int>(status),
                    "hipblasLtMatmulPreferenceCreate failed");
    }
    const auto workspace_limit = static_cast<std::uint64_t>(workspace_bytes);
    status = hipblasLtMatmulPreferenceSetAttribute(
        preference,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_limit,
        sizeof(workspace_limit));
    if (status != HIPBLAS_STATUS_SUCCESS) {
        hipblasLtMatmulPreferenceDestroy(preference);
        return fail(context,
                    static_cast<int>(status),
                    "hipblasLtMatmulPreferenceSetAttribute failed");
    }

    std::array<hipblasLtMatmulHeuristicResult_t, kHeuristicCount> candidates{};
    int candidate_count = 0;
    status = hipblasLtMatmulAlgoGetHeuristic(context->handle,
                                             plan->operation,
                                             plan->rhs,
                                             plan->lhs,
                                             plan->output,
                                             plan->output,
                                             preference,
                                             kHeuristicCount,
                                             candidates.data(),
                                             &candidate_count);
    hipblasLtMatmulPreferenceDestroy(preference);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        return fail(context,
                    static_cast<int>(status),
                    "hipblasLtMatmulAlgoGetHeuristic failed");
    }
    if (candidate_count == 0) {
        return fail(context, kInternalError, "hipBLASLt returned no algorithms");
    }

    hipEvent_t start{};
    hipEvent_t stop{};
    auto hip_status = hipEventCreate(&start);
    if (hip_status != hipSuccess) {
        return fail_hip(context, hip_status, "hipEventCreate(tuning start)");
    }
    hip_status = hipEventCreate(&stop);
    if (hip_status != hipSuccess) {
        [[maybe_unused]] const auto destroy_status = hipEventDestroy(start);
        return fail_hip(context, hip_status, "hipEventCreate(tuning stop)");
    }

    float best_milliseconds = std::numeric_limits<float>::infinity();
    hipblasLtMatmulAlgo_t best_algorithm{};
    for (int index = 0; index < candidate_count; ++index) {
        const auto& candidate = candidates[static_cast<std::size_t>(index)];
        if (candidate.state != HIPBLAS_STATUS_SUCCESS
            || candidate.workspaceSize > workspace_bytes) {
            continue;
        }

        plan->algorithm = candidate.algo;
        if (launch(context->handle,
                   *plan,
                   lhs,
                   rhs,
                   output,
                   workspace,
                   workspace_bytes,
                   stream)
            != HIPBLAS_STATUS_SUCCESS) {
            continue;
        }
        if (hipEventRecord(start, stream) != hipSuccess) {
            continue;
        }
        status = launch(context->handle,
                        *plan,
                        lhs,
                        rhs,
                        output,
                        workspace,
                        workspace_bytes,
                        stream);
        if (status != HIPBLAS_STATUS_SUCCESS) {
            continue;
        }
        if (hipEventRecord(stop, stream) != hipSuccess
            || hipEventSynchronize(stop) != hipSuccess) {
            continue;
        }
        float milliseconds = 0.0F;
        if (hipEventElapsedTime(&milliseconds, start, stop) != hipSuccess) {
            continue;
        }
        if (milliseconds < best_milliseconds) {
            best_milliseconds = milliseconds;
            best_algorithm = candidate.algo;
        }
    }
    [[maybe_unused]] const auto destroy_start_status = hipEventDestroy(start);
    [[maybe_unused]] const auto destroy_stop_status = hipEventDestroy(stop);

    if (!std::isfinite(best_milliseconds)) {
        return fail(context,
                    kInternalError,
                    "no hipBLASLt heuristic algorithm launched successfully");
    }
    plan->algorithm = best_algorithm;
    result = std::move(plan);
    return 0;
}

}  // namespace

// HIP runtime wrappers.
//
// The Rust side never includes HIP headers; it consumes this narrow C ABI
// instead. Streams are opaque `void*` handles and every entry point returns
// the raw `hipError_t` status as an `int` where `0` means `hipSuccess`.

extern "C" int mote_hip_get_device_count(int* count) noexcept {
    if (count == nullptr) {
        return kInternalError;
    }
    return static_cast<int>(hipGetDeviceCount(count));
}

extern "C" int mote_hip_set_device(int ordinal) noexcept {
    return static_cast<int>(hipSetDevice(ordinal));
}

extern "C" int mote_hip_malloc(void** pointer, std::size_t bytes) noexcept {
    if (pointer == nullptr) {
        return kInternalError;
    }
    *pointer = nullptr;
    return static_cast<int>(hipMalloc(pointer, bytes));
}

extern "C" int mote_hip_free(void* pointer) noexcept {
    return static_cast<int>(hipFree(pointer));
}

extern "C" int mote_hip_memcpy_host_to_device(void* destination,
                                               const void* source,
                                               std::size_t bytes) noexcept {
    return static_cast<int>(
        hipMemcpy(destination, source, bytes, hipMemcpyHostToDevice));
}

extern "C" int mote_hip_memcpy_device_to_host(void* destination,
                                               const void* source,
                                               std::size_t bytes) noexcept {
    return static_cast<int>(
        hipMemcpy(destination, source, bytes, hipMemcpyDeviceToHost));
}

extern "C" int mote_hip_stream_create(void** stream) noexcept {
    if (stream == nullptr) {
        return kInternalError;
    }
    *stream = nullptr;
    hipStream_t native = nullptr;
    const auto status = hipStreamCreate(&native);
    *stream = native;
    return static_cast<int>(status);
}

extern "C" int mote_hip_stream_destroy(void* stream) noexcept {
    return static_cast<int>(hipStreamDestroy(static_cast<hipStream_t>(stream)));
}

extern "C" int mote_hip_stream_synchronize(void* stream) noexcept {
    return static_cast<int>(
        hipStreamSynchronize(static_cast<hipStream_t>(stream)));
}

extern "C" int mote_hip_mem_get_info(std::size_t* free_bytes,
                                      std::size_t* total_bytes) noexcept {
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return kInternalError;
    }
    *free_bytes = 0;
    *total_bytes = 0;
    return static_cast<int>(hipMemGetInfo(free_bytes, total_bytes));
}

extern "C" int mote_hipblaslt_create(void** output) noexcept {
    if (output == nullptr) {
        return kInternalError;
    }
    try {
        auto context = std::make_unique<Context>();
        const auto status = hipblasLtCreate(&context->handle);
        if (status != HIPBLAS_STATUS_SUCCESS) {
            return static_cast<int>(status);
        }
        *output = context.release();
        return 0;
    } catch (...) {
        return kInternalError;
    }
}

extern "C" void mote_hipblaslt_destroy(void* opaque) noexcept {
    delete static_cast<Context*>(opaque);
}

extern "C" const char* mote_hipblaslt_last_error(void* opaque) noexcept {
    const auto* context = static_cast<Context*>(opaque);
    return context == nullptr ? nullptr : context->last_error.c_str();
}

extern "C" int mote_hipblaslt_matmul_f16_f32(void* opaque,
                                               std::uint64_t m,
                                               std::uint64_t n,
                                               std::uint64_t k,
                                               const void* lhs,
                                               const void* rhs,
                                               void* output,
                                               void* workspace,
                                               std::size_t workspace_bytes,
                                               void* stream) noexcept {
    auto* context = static_cast<Context*>(opaque);
    if (context == nullptr || lhs == nullptr || rhs == nullptr || output == nullptr
        || workspace == nullptr || stream == nullptr) {
        return fail(context, kInternalError, "hipBLASLt matmul received a null handle");
    }
    const auto native_stream = static_cast<hipStream_t>(stream);
    try {
        const std::array<std::uint64_t, 3> key{m, n, k};
        auto found = context->plans.find(key);
        if (found == context->plans.end()) {
            std::unique_ptr<Plan> plan;
            const auto status = create_plan(context,
                                            m,
                                            n,
                                            k,
                                            lhs,
                                            rhs,
                                            output,
                                            workspace,
                                            workspace_bytes,
                                            native_stream,
                                            plan);
            if (status != 0) {
                return status;
            }
            found = context->plans.emplace(key, std::move(plan)).first;
        }

        const auto status = launch(context->handle,
                                   *found->second,
                                   lhs,
                                   rhs,
                                   output,
                                   workspace,
                                   workspace_bytes,
                                   native_stream);
        if (status != HIPBLAS_STATUS_SUCCESS) {
            return fail(context, static_cast<int>(status), "hipblasLtMatmul failed");
        }
        context->last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        return fail(context, kInternalError, error.what());
    } catch (...) {
        return fail(context, kInternalError, "unknown C++ exception");
    }
}

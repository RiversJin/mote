cmake_minimum_required(VERSION 3.25)
project(mote_rocm_bench LANGUAGES CXX)

find_package(hip CONFIG REQUIRED)
find_package(hipblas-common CONFIG REQUIRED)
find_package(hipblaslt CONFIG REQUIRED)
find_package(rocblas CONFIG REQUIRED)

add_executable(rocm-matmul-bench rocm_matmul_bench.cpp)
target_compile_features(rocm-matmul-bench PRIVATE cxx_std_20)
target_compile_definitions(rocm-matmul-bench PRIVATE ROCM_USE_FLOAT16)
target_compile_options(rocm-matmul-bench PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(rocm-matmul-bench PRIVATE hip::host roc::hipblaslt roc::rocblas)

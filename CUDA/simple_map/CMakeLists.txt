cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

# CUDA NVCC Flags
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    # Add NVCC flags here
    -Wno-deprecated-gpu-targets
    )

include_directories(
    )

link_directories(
    )

cuda_add_executable(
    kernel
    kernel.cu
    )

target_link_libraries(
    kernel
    )
cuda_add_library(bb_cuda STATIC
    forward_pass.cu
    plasticity_kernels.cu
    reward_kernels.cu
    structural_kernels.cu
)

target_include_directories(bb_cuda PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

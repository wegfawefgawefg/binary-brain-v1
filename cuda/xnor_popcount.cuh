#pragma once
#include "bb_types.hpp"

// Host and device compatible popcount helper
#ifdef __CUDA_ARCH__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace bb::cuda {
    /** Warp-level XNOR + popcount on 64-bit lanes. Works on both host and device. */
    __host__ __device__ inline int warp_xnor_popcount(weight_word_t a, weight_word_t b) {
#ifdef __CUDA_ARCH__
        return __popcll(~(a ^ b));
#else
        return __builtin_popcountll(~(a ^ b));
#endif
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    template <typename TileShape>
    __device__ void mma_b1() {
        // placeholder for real WMMA implementation
    }
#endif
} // namespace bb::cuda

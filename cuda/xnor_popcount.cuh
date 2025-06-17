#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "bb_types.hpp"

namespace bb::cuda {
    /** Warp-level XNOR + popcount on 64-bit lanes. */
    __device__ inline int warp_xnor_popcount(weight_word_t a, weight_word_t b) {
        return __popcll(~(a ^ b));
    }

#if __CUDA_ARCH__ >= 750
    template <typename TileShape>
    __device__ void mma_b1() {
        // placeholder
    }
#endif
} // namespace bb::cuda

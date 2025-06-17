#include "bb_core/synapse.hpp"
#ifdef BB_WITH_CUDA
#include <cuda_runtime.h>
#endif
#include "xnor_popcount.cuh"

__host__ __device__ void bb::SynapseBlock::dot_xnor_popcount(const weight_word_t* a, int32_t* out_accum) const {
#ifdef __CUDA_ARCH__
    for (unsigned i = 0; i < word_count; ++i) {
        weight_word_t x = a[i] ^ weights[i];
        int pc = __popcll(~x);
        atomicAdd(out_accum, pc);
    }
#else
    for (unsigned i = 0; i < word_count; ++i) {
        weight_word_t x = a[i] ^ weights[i];
        int pc = bb::cuda::warp_xnor_popcount(a[i], weights[i]);
        *out_accum += pc;
    }
#endif
}

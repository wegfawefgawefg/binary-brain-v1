#include "bb_core/synapse.hpp"
#include <cuda_runtime.h>

__device__ void bb::SynapseBlock::dot_xnor_popcount(const weight_word_t* a, int32_t* out_accum) const {
    for (unsigned i = 0; i < word_count; ++i) {
        weight_word_t x = a[i] ^ weights[i]; // XOR
        int pc = __popcll(~x); // XNOR then popcount
        atomicAdd(out_accum, pc);
    }
}

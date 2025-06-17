#include "bb_core/structural_plasticity.hpp"

#ifdef __CUDACC__
__global__ void bb::kernels::structural_pass(weight_word_t* synapse_words,
                                             uint32_t*      usage_counters,
                                             float          prune_threshold) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (usage_counters[idx] < prune_threshold) {
        synapse_words[idx] = 0ull;
    }
}
#endif

#ifndef __CUDACC__
void bb::kernels::structural_pass(weight_word_t* synapse_words,
                                  uint32_t*      usage_counters,
                                  float          prune_threshold,
                                  unsigned       word_count)
{
    for (unsigned idx = 0; idx < word_count; ++idx) {
        if (usage_counters[idx] < prune_threshold) {
            synapse_words[idx] = 0ull;
        }
    }
}
#endif

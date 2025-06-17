#include "bb_core/structural_plasticity.hpp"

__global__ void bb::kernels::structural_pass(weight_word_t* synapse_words,
                                             uint32_t*      usage_counters,
                                             float          prune_threshold) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (usage_counters[idx] < prune_threshold) {
        synapse_words[idx] = 0ull;
    }
}

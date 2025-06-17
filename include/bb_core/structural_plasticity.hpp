#include "bb_types.hpp"
#pragma once
namespace bb::kernels {
    /**
     * CUDA kernel: prune under-used synapses and allocate new ones where
     * pre & S are frequent.
     */
    __global__ void structural_pass(weight_word_t* synapse_words,
                                    uint32_t*      usage_counters,
                                    float          prune_threshold);
#ifndef __CUDACC__
    void structural_pass(weight_word_t* synapse_words,
                         uint32_t*      usage_counters,
                         float          prune_threshold,
                         unsigned       word_count);
#endif
} // namespace bb::kernels

#pragma once
#include "bb_types.hpp"
namespace bb::kernels {
    /**
     * CUDA kernel: apply three-factor bit-flip rule.
     *  - Each thread owns one weight_word
     *  - Uses RewardBus bits broadcast via CUDA constant memory.
     */
    __global__ void plasticity_step(const weight_word_t* pre_spike,
                                    weight_word_t*       synapse_words,
                                    unsigned             word_count,
                                    uint8_t              flip_prob256);
} // namespace bb::kernels

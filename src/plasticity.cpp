#include "bb_core/plasticity.hpp"

__global__ void bb::kernels::plasticity_step(const weight_word_t* pre_spike,
                                             weight_word_t*       synapse_words,
                                             unsigned             word_count,
                                             uint8_t              flip_prob256) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= word_count) return;
    weight_word_t w = synapse_words[idx];
    weight_word_t p = pre_spike[idx];
    if ((p ^ w) & 1ull && (flip_prob256 > 0)) {
        w ^= 1ull;
    }
    synapse_words[idx] = w;
}

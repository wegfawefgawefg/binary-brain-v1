#include "bb_core/plasticity.hpp"
#include <cstdlib>

using namespace bb;
using namespace bb::kernels;

#ifdef __CUDACC__
__global__ void bb::kernels::plasticity_step(const weight_word_t* pre_spike,
                                             weight_word_t*       synapse_words,
                                             unsigned             word_count,
                                             uint8_t              flip_prob256)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= word_count) return;
    weight_word_t w    = synapse_words[idx];
    weight_word_t pre  = pre_spike[idx];
    weight_word_t mask = 1ull;
    for (int b = 0; b < 64; ++b, mask <<= 1) {
        bool wrong = ((pre & mask) != (w & mask));
        if (wrong && flip_prob256 > 0) {
            w ^= mask;
        }
    }
    synapse_words[idx] = w;
}
#endif

#ifndef __CUDACC__
void bb::kernels::plasticity_step(const weight_word_t* pre_spike,
                                  weight_word_t*       synapse_words,
                                  unsigned             word_count,
                                  uint8_t              flip_prob256)
{
    for (unsigned idx = 0; idx < word_count; ++idx) {
        weight_word_t w    = synapse_words[idx];
        weight_word_t pre  = pre_spike[idx];
        weight_word_t mask = 1ull;
        for (int b = 0; b < 64; ++b, mask <<= 1) {
            bool wrong = ((pre & mask) != (w & mask));
            if (wrong && flip_prob256 > 0) {
                w ^= mask;
            }
        }
        synapse_words[idx] = w;
    }
}
#endif


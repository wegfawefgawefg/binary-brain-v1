#pragma once
#include "bb_types.hpp"

namespace bb {
struct SynapseBlock final {
    weight_word_t* weights;  // device pointer (length = N_words)
    unsigned       word_count;

    /**
     * Compute popcount(XNOR(a,b)) on packed 64-bit words.
     * @param a Pointer to presynaptic packed activity.
     * @param out_accum Int32 accumulator in shared memory.
     */
    __host__ __device__ void dot_xnor_popcount(const weight_word_t* a, int32_t* out_accum) const;
};
} // namespace bb

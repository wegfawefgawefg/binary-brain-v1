#pragma once
#include "bb_types.hpp"

namespace bb {
struct Neuron final {
    neuron_id_t id;
    coord_t     x, y, z;   // immutable position
    phase_t     phase;     // oscillator tag
    uint8_t     target_rate; // homeostatic target (0-255)

    // spike state buffers (double-buffered)
    uint64_t    spike_mask_curr;
    uint64_t    spike_mask_prev;

    /** Compute intrinsic prediction and write to spike_mask_curr. */
    __host__ __device__ void forward_intrinsic();
};
} // namespace bb

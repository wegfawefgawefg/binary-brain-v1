#include "bb_core/neuron.hpp"

__host__ __device__ void bb::Neuron::forward_intrinsic() {
    // simplistic oscillator toggle
    spike_mask_prev = spike_mask_curr;
    spike_mask_curr ^= 1ull;
}

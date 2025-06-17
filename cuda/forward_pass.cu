#include "xnor_popcount.cuh"
#include "bb_core/neuron.hpp"
#include "bb_core/synapse.hpp"

__global__ void forward_kernel(bb::Neuron* neurons, const bb::SynapseBlock block) {
    // stub kernel
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < block.word_count) {
        // example: toggle spike
        neurons[idx].spike_mask_curr ^= 1ull;
    }
}

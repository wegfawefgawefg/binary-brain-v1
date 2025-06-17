#include "bb_core/network.hpp"
#include "bb_config.hpp"
#include "bb_core/structural_plasticity.hpp"
#include <random>

using namespace bb;

Network::Network(unsigned num_neurons)
    : neurons_(num_neurons),
      weight_storage_(num_neurons, 0ull),
      usage_counters_(num_neurons, 1u),
      sensors_(num_neurons),
      actuators_(num_neurons) {
    synapses_.weights     = weight_storage_.data();
    synapses_.word_count  = static_cast<unsigned>(weight_storage_.size());
    for (unsigned i = 0; i < num_neurons; ++i) {
        neurons_[i].id            = i;
        neurons_[i].spike_mask_curr = 0ull;
        neurons_[i].spike_mask_prev = 0ull;
    }
}

void Network::step(float dt_sec) {
    sense();
    compute_reward();
    forward_pass_device();
    apply_plasticity_device();
    structural_timer_ += dt_sec;
    if (structural_timer_ >= cfg::PRUNE_FREQ) {
        structural_pass_device();
        structural_timer_ = 0.f;
    }
}

RewardBus& Network::reward() { return reward_; }
const RewardBus& Network::reward() const { return reward_; }

void Network::sense() {
    // For demonstration, copy sensor bytes directly into spike masks.
    for (unsigned i = 0; i < neurons_.size(); ++i) {
        neurons_[i].spike_mask_prev = neurons_[i].spike_mask_curr;
        neurons_[i].spike_mask_curr = sensors_.data()[i % sensors_.size()] & 1u;
    }
}

void Network::compute_reward() {
    // Very naive statistics for now: prediction error as fraction of mismatched spikes
    unsigned mismatch = 0;
    for (const auto& n : neurons_) {
        mismatch += static_cast<unsigned>((n.spike_mask_curr ^ n.spike_mask_prev) & 1ull);
    }
    float pred_err = static_cast<float>(mismatch) / neurons_.size();
    reward_.update(pred_err, false, 0.0f);
}

void Network::forward_pass_device() {
#ifdef BB_WITH_CUDA
    // GPU path would launch forward_kernel (omitted)
#else
    // CPU fallback simply toggles each neuron's spike bit
    for (auto& n : neurons_) {
        n.forward_intrinsic();
    }
#endif
    for (auto& counter : usage_counters_) {
        counter++;
    }
}

void Network::apply_plasticity_device() {
#ifdef BB_WITH_CUDA
    // would launch plasticity kernels with proper grid dimensions
#else
    if (reward_.S || reward_.R_neg) {
        for (auto& w : weight_storage_) {
            w ^= 1ull; // basic bit flip for demo
        }
    }
#endif
}

void Network::structural_pass_device() {
#ifdef BB_WITH_CUDA
    // structural kernels on the GPU
#else
    kernels::structural_pass(weight_storage_.data(), usage_counters_.data(),
                             0.5f,
                             static_cast<unsigned>(weight_storage_.size()));
#endif
}

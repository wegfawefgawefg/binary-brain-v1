#include "bb_core/network.hpp"
#include "bb_config.hpp"

using namespace bb;

Network::Network(unsigned num_neurons) : neurons_(num_neurons) {}

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

void Network::sense() { /* stub read sensors */ }
void Network::compute_reward() { reward_.update(0.0f, false, 0.0f); }
void Network::forward_pass_device() { /* launch CUDA kernels (stub) */ }
void Network::apply_plasticity_device() { /* launch CUDA kernels (stub) */ }
void Network::structural_pass_device() { /* launch CUDA kernels (stub) */ }

#include "bb_core/network.hpp"
#include "bb_config.hpp"
#include <cstdlib>
#include <ctime>
#include <algorithm>

using namespace bb;

Network::Network(unsigned num_neurons)
    : neurons_(num_neurons),
      weight_storage_(num_neurons),
      usage_counters_(num_neurons, 0),
      sensors_(num_neurons),
      actuators_(num_neurons) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    synapses_.weights     = weight_storage_.data();
    synapses_.word_count  = num_neurons;

    for (unsigned i = 0; i < num_neurons; ++i) {
        neurons_[i].id            = i;
        neurons_[i].x             = static_cast<coord_t>(i);
        neurons_[i].y             = 0;
        neurons_[i].z             = 0;
        neurons_[i].phase         = 0;
        neurons_[i].target_rate   = 128;
        neurons_[i].spike_mask_curr = 0;
        neurons_[i].spike_mask_prev = 0;
        weight_storage_[i] = (static_cast<weight_word_t>(std::rand()) << 32) |
                             static_cast<weight_word_t>(std::rand());
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
    for (unsigned i = 0; i < sensors_.size(); ++i) {
        sensors_.data()[i] = static_cast<uint8_t>(std::rand() & 1);
    }
}

void Network::compute_reward() {
    float pred_err = 0.f;
    float rate_dev = 0.f;
    reward_.update(pred_err, false, rate_dev);
}

void Network::forward_pass_device() {
    weight_word_t pre_word = 0;
    unsigned max_bits = std::min<unsigned>(cfg::TILE_BITS, sensors_.size());
    for (unsigned j = 0; j < max_bits; ++j) {
        pre_word |= static_cast<weight_word_t>(sensors_.data()[j] & 1) << j;
    }

    for (unsigned i = 0; i < neurons_.size(); ++i) {
        weight_word_t w = weight_storage_[i];
        int pc = __builtin_popcountll(~(pre_word ^ w));
        bool spike = pc > static_cast<int>(cfg::TILE_BITS / 2);
        neurons_[i].spike_mask_prev = neurons_[i].spike_mask_curr;
        neurons_[i].spike_mask_curr = spike ? 1ull : 0ull;
        if (spike) usage_counters_[i]++;
    }
}

void Network::apply_plasticity_device() {
    weight_word_t pre_word = 0;
    unsigned max_bits = std::min<unsigned>(cfg::TILE_BITS, sensors_.size());
    for (unsigned j = 0; j < max_bits; ++j) {
        pre_word |= static_cast<weight_word_t>(sensors_.data()[j] & 1) << j;
    }

    for (unsigned i = 0; i < neurons_.size(); ++i) {
        bool post = neurons_[i].spike_mask_curr & 1ull;
        weight_word_t wrong_mask = pre_word ^ (post ? ~0ull : 0ull);
        for (unsigned b = 0; b < cfg::TILE_BITS; ++b) {
            if (wrong_mask & (1ull << b)) {
                if (static_cast<float>(std::rand()) / RAND_MAX < cfg::FLIP_PROB) {
                    weight_storage_[i] ^= (1ull << b);
                }
            }
        }
    }
}

void Network::structural_pass_device() {
    for (unsigned i = 0; i < weight_storage_.size(); ++i) {
        if (usage_counters_[i] == 0) {
            weight_storage_[i] = (static_cast<weight_word_t>(std::rand()) << 32) |
                                 static_cast<weight_word_t>(std::rand());
        }
        usage_counters_[i] = 0;
    }
}

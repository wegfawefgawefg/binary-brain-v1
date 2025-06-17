#pragma once
#include <vector>
#include "neuron.hpp"
#include "synapse.hpp"
#include "reward_bus.hpp"
#include "sensor_pool.hpp"
#include "actuator_pool.hpp"

namespace bb {
class Network final {
public:
    explicit Network(unsigned num_neurons);

    // Clock‐tick entry point (host)
    void step(float dt_sec);

    // Accessors
    RewardBus&       reward();
    const RewardBus& reward() const;

private:
    // === phases split out for clarity ===
    void sense();                       // read SensorPool → neuron inputs
    void compute_reward();              // populate RewardBus
    void forward_pass_device();         // CUDA kernel launch
    void apply_plasticity_device();     // plasticity kernels
    void structural_pass_device();      // prune/grow kernels (≤1 Hz)

    // --- state ---
    std::vector<Neuron>         neurons_;
    std::vector<weight_word_t>  weight_storage_;
    SynapseBlock                synapses_{};
    std::vector<uint32_t>       usage_counters_;
    SensorPool                  sensors_;
    ActuatorPool                actuators_;
    RewardBus                   reward_;
    float                       structural_timer_{0.f};
};
} // namespace bb

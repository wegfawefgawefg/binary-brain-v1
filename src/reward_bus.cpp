#include "bb_core/reward_bus.hpp"

void bb::RewardBus::update(float prediction_err, bool danger_flag, float firing_rate_dev) noexcept {
    S = prediction_err > 0.5f;
    R_neg = danger_flag;
    H = firing_rate_dev < 0.1f;
    R_pos = S && !R_neg && H;
}

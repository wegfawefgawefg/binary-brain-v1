#pragma once
namespace bb {
struct RewardBus {
    bool R_pos {false};   // pleasure
    bool R_neg {false};   // pain / danger
    bool S     {false};   // surprise / novelty
    bool H     {true};    // homeostasis-ok

    /** Update reward bits from sensor & prediction stats. */
    void update(float prediction_err, bool danger_flag, float firing_rate_dev) noexcept;
};
} // namespace bb

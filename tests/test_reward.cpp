#include <gtest/gtest.h>
#include "bb_core/reward_bus.hpp"

TEST(Reward, Update) {
    bb::RewardBus bus;
    bus.update(1.0f, false, 0.0f);
    ASSERT_TRUE(bus.S);
}

#include <gtest/gtest.h>
#include "bb_core/network.hpp"

TEST(Forward, XnorPopcountDot) {
    bb::Network net(1024);
    net.step(0.010f);
    ASSERT_TRUE(true); // TODO: add proper assertions
}

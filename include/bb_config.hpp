#pragma once
namespace bb::cfg {
    inline constexpr unsigned TILE_BITS   = 64;   // compile-time pack size
    inline constexpr float    TICK_SEC    = 0.010f;
    inline constexpr float    PRUNE_FREQ  = 1.0f;  // structural pass (s)
    inline constexpr float    FLIP_PROB   = 0.01f; // base stochastic flip
    // add more tunables here
}

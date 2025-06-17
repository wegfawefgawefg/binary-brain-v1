#pragma once
#include <cstdint>

namespace bb {
    using neuron_id_t   = uint32_t;    // supports up to 4 billion neurons
    using coord_t       = uint16_t;    // fixed-point 16-bit coordinate per axis
    using phase_t       = uint16_t;    // 8-bit freq | 8-bit phase
    using weight_word_t = uint64_t;    // 64 binary synapses packed
} // namespace bb

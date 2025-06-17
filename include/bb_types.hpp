#pragma once
#include <cstdint>

// Provide CUDA attributes as no-ops when compiling without nvcc so that the
// headers can be consumed by a plain C++ compiler. This allows the core logic
// to build even if CUDA is unavailable.
#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#endif

namespace bb {
    using neuron_id_t   = uint32_t;    // supports up to 4 billion neurons
    using coord_t       = uint16_t;    // fixed-point 16-bit coordinate per axis
    using phase_t       = uint16_t;    // 8-bit freq | 8-bit phase
    using weight_word_t = uint64_t;    // 64 binary synapses packed
} // namespace bb

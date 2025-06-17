# Codebase Skeleton: One‑Bit Self‑Organising Neural Substrate

This document defines the **directory layout, header/API stubs, build tooling, and Python bindings** with enough detail for downstream implementers to fill in logic without needing further architectural clarification.

> **Language choices**\
> • C++‑17 for host logic\
> • CUDA 11.x for device kernels (or HIP with trivial macro swap)\
> • pybind11 for Python bindings\
> • CMake ≥ 3.25 to glue it all together

```
project_root/
│
├─ CMakeLists.txt                # top‑level build script
├─ cmake/                        # toolchain & FindXXX helpers
│
├─ include/                      # public headers (installable)
│   ├─ bb_types.hpp
│   └─ bb_config.hpp
│
├─ include/bb_core/              # core data‑structures
│   ├─ neuron.hpp
│   ├─ synapse.hpp
│   ├─ reward_bus.hpp
│   ├─ sensor_pool.hpp
│   ├─ actuator_pool.hpp
│   ├─ network.hpp
│   ├─ plasticity.hpp
│   └─ structural_plasticity.hpp
│
├─ src/                           # host‑side implementations
│   ├─ neuron.cpp
│   ├─ synapse.cpp
│   ├─ reward_bus.cpp
│   ├─ sensor_pool.cpp
│   ├─ actuator_pool.cpp
│   ├─ network.cpp
│   ├─ plasticity.cpp
│   └─ structural_plasticity.cpp
│
├─ cuda/                          # device kernels / templates
│   ├─ xnor_popcount.cuh          # templated warp‑level primitives
│   ├─ forward_pass.cu
│   ├─ plasticity_kernels.cu
│   ├─ reward_kernels.cu
│   └─ structural_kernels.cu
│
├─ python/
│   ├─ CMakeLists.txt             # builds pybind11 module
│   ├─ bb_pybind.cpp              # bindings
│   └─ brain.py                   # high‑level Python wrapper
│
├─ tests/
│   ├─ test_forward.cpp
│   ├─ test_plasticity.cpp
│   ├─ test_reward.cpp
│   └─ test_structural.cpp
│
└─ scripts/
    ├─ simulate.py
    └─ visualise_grid_cells.ipynb
```

---

## 1 Top‑Level CMakeLists.txt (stub)

```cmake
cmake_minimum_required(VERSION 3.25)
project(BinaryBrain LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(cuda)
add_subdirectory(python)
add_subdirectory(tests)
```

*Down‑stream implementers fill in **``**, compute capability flags, etc.*

---

## 2 Public Headers (API stubs)

### 2.1 `include/bb_types.hpp`

```cpp
#pragma once
#include <cstdint>

namespace bb {
    using neuron_id_t   = uint32_t;    // supports up to 4 billion neurons
    using coord_t       = uint16_t;    // fixed‑point 16‑bit coordinate per axis
    using phase_t       = uint16_t;    // 8‑bit freq | 8‑bit phase
    using weight_word_t = uint64_t;    // 64 binary synapses packed
} // namespace bb
```

### 2.2 `include/bb_config.hpp`

```cpp
#pragma once
namespace bb::cfg {
    inline constexpr unsigned TILE_BITS   = 64;   // compile‑time pack size
    inline constexpr float    TICK_SEC    = 0.010f;
    inline constexpr float    PRUNE_FREQ  = 1.0f;  // structural pass (s)
    inline constexpr float    FLIP_PROB   = 0.01f; // base stochastic flip
    // add more tunables here
}
```

### 2.3 `include/bb_core/neuron.hpp`

```cpp
#pragma once
#include "bb_types.hpp"

namespace bb {
struct Neuron final {
    neuron_id_t id;
    coord_t     x, y, z;   // immutable position
    phase_t     phase;     // oscillator tag
    uint8_t     target_rate; // homeostatic target (0‑255)

    // spike state buffers (double‑buffered)
    uint64_t    spike_mask_curr;
    uint64_t    spike_mask_prev;

    /** Compute intrinsic prediction and write to spike_mask_curr. */
    __host__ __device__ void forward_intrinsic();
};
} // namespace bb
```

### 2.4 `include/bb_core/synapse.hpp`

```cpp
#pragma once
#include "bb_types.hpp"

namespace bb {
struct SynapseBlock final {
    weight_word_t* weights;  // device pointer (length = N_words)
    unsigned       word_count;

    /**
     * Compute popcount(XNOR(a,b)) on packed 64‑bit words.
     * @param a Pointer to presynaptic packed activity.
     * @param out_accum Int32 accumulator in shared memory.
     */
    __device__ void dot_xnor_popcount(const weight_word_t* a, int32_t* out_accum) const;
};
} // namespace bb
```

### 2.5 `include/bb_core/reward_bus.hpp`

```cpp
#pragma once
namespace bb {
struct RewardBus {
    bool R_pos {false};   // pleasure
    bool R_neg {false};   // pain / danger
    bool S     {false};   // surprise / novelty
    bool H     {true};    // homeostasis‑ok

    /** Update reward bits from sensor & prediction stats. */
    void update(float prediction_err, bool danger_flag, float firing_rate_dev) noexcept;
};
} // namespace bb
```

### 2.6 `include/bb_core/network.hpp`

```cpp
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
    SynapseBlock                synapses_;
    RewardBus                   reward_;
    float                       structural_timer_ {0.f};
};
} // namespace bb
```

### 2.7 `include/bb_core/plasticity.hpp`

```cpp
#pragma once
#include "bb_types.hpp"
namespace bb::kernels {
    /**
     * CUDA kernel: apply three‑factor bit‑flip rule.
     *  - Each thread owns one weight_word
     *  - Uses RewardBus bits broadcast via CUDA constant memory.
     */
    __global__ void plasticity_step(const weight_word_t* pre_spike,
                                    weight_word_t*       synapse_words,
                                    unsigned             word_count,
                                    uint8_t              flip_prob256);
} // namespace bb::kernels
```

### 2.8 `include/bb_core/structural_plasticity.hpp`

```cpp
#pragma once
namespace bb::kernels {
    /**
     * CUDA kernel: prune under‑used synapses and allocate new ones where
     * pre & S are frequent.
     */
    __global__ void structural_pass(weight_word_t* synapse_words,
                                    uint32_t*      usage_counters,
                                    float          prune_threshold);
} // namespace bb::kernels
```

---

## 3 CUDA Kernel Stubs (headers only)

### 3.1 `cuda/xnor_popcount.cuh`

```cpp
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "bb_types.hpp"

namespace bb::cuda {
    /** Warp‑level XNOR + popcount on 64‑bit lanes. */
    __device__ inline int warp_xnor_popcount(weight_word_t a, weight_word_t b);

    /** Cooperative MMA wrapper if WMMA.B1 is available. */
    #if __CUDA_ARCH__ >= 750
    template <typename TileShape>
    __device__ void mma_b1(/* params */);
    #endif
} // namespace bb::cuda
```

> Other `.cu` files simply include the stubs above and expose `__global__` kernels—omitted for brevity.

---

## 4 Python Binding Stub (`python/bb_pybind.cpp`)

```cpp
#include <pybind11/pybind11.h>
#include "bb_core/network.hpp"

PYBIND11_MODULE(_binarybrain, m) {
    namespace py = pybind11;
    py::class_<bb::Network>(m, "Network")
        .def(py::init<unsigned>())
        .def("step", &bb::Network::step)
        .def_property_readonly("reward", &bb::Network::reward);
}
```

### 4.1 High‑level Python Helper (`python/brain.py`)

```python
import _binarybrain as _bb

class Brain:
    def __init__(self, n):
        self._net = _bb.Network(n)

    def tick(self, dt=0.010):
        """Advance simulation by dt seconds."""
        self._net.step(dt)

    @property
    def reward(self):
        return self._net.reward
```

---

## 5 Unit‑Test Skeleton (`tests/test_forward.cpp`)

```cpp
#include <gtest/gtest.h>
#include "bb_core/network.hpp"

TEST(Forward, XnorPopcountDot) {
    bb::Network net(1024);
    net.step(0.010f);
    ASSERT_TRUE(true); // TODO: add proper assertions
}
```

*(Similar stubs exist for plasticity, reward, structural passes.)*

---

## 6 Utility Scripts (names only)

| Script                         | Purpose                                                |
| ------------------------------ | ------------------------------------------------------ |
| `scripts/simulate.py`          | CLI loop to run the brain on gym‑like envs.            |
| `scripts/visualise_grid_cells` | Jupyter notebook: FFT/PCA plots of neuron activations. |

---

## 7 Implementation Notes for Down‑Stream Agents

- **No function bodies** included above—replace each with actual CUDA/CPU code.
- Stick to **idempotent kernels** (no warp divergence) in device code.
- Keep **global memory accesses coalesced**; favour shared memory tiles.
- Re‑run `clang‑format` with NVIDIA style on all `.hpp/.cu` files.

---

*End of Skeleton*


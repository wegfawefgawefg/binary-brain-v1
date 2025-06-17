# Project Spec: One‑Bit Self‑Organising Neural Substrate

*Version 0.1 — June 17 2025*

---

## 1 Goal & Scope

Design an always‑on neural system with **1‑bit weights** that can run efficiently on commodity GPUs/SoCs (Turing/​Ampere, GH200 fallback) while learning online in arbitrary real or virtual environments **without handcrafted task rewards**. The network must:

1. Operate continuously on a fixed clock (e.g. 10 ms ticks).
2. Learn using **purely local, three‑factor plasticity**—no gradient back‑prop.
3. Receive only four global 1‑bit neuromodulator lines (pleasure, pain, surprise, homeostasis).
4. Self‑organise topology and features (grid cells, conv‑like filters, attention hops) through structural plasticity.
5. Fit memory‑bandwidth constraints via packed XNOR+POPCNT kernels (INT1 when available, INT4 fallback).

---

## 2 Core Concepts

| Component                  | Summary                                                                                          |
| -------------------------- | ------------------------------------------------------------------------------------------------ |
| **Neuron ID & Coordinate** | Each neuron has immutable `(x,y[,z])` 16‑bit coords and a 16‑bit oscillator phase/frequency tag. |
| **Synapse (binary)**       | Weight `w ∈ {−1,+1}` stored as 1 bit in a packed 64‑bit lane.                                    |
| **Activity Bit**           | Spike state for the current tick.                                                                |
| **Global Reward Bus**      | 4 single‑bit lanes: `R⁺` (dopamine), `R⁻` (pain), `S` (surprise), `H` (homeostasis).             |
| **Local Plasticity**       | `Δw = f(pre, post, Reward)`, implemented as stochastic bit‑flip (`w ^= flip_mask`).              |
| **Structural Plasticity**  | Background prune/grow that reallocates sparse long‑range links based on statistics.              |

---

## 3 Data Representation

### 3.1 Weight Packing

- 64 synapses → one `uint64_t`.
- Encoding: `1 → +1`, `0 → −1`.
- Dot‑like operation: `dot(x,w) ≈ 2·popcnt(XNOR(x,w)) − N`.

### 3.2 Activation Buffers

- Current spike mask and a one‑tick delay for prediction.
- Stored per warp in registers / shared memory.

---

## 4 Topology Construction

1. **Local Mesh**: connect neurons with probability `P(d)=exp(−d²/σ²)`.
2. **Sparse Long Hops**: add `ρ_long ≪ 1` random far edges ⇒ small‑world graph.
3. **Boot Algorithm**:
   ```pseudo
   for each neuron i:
       for j in Neurons:
           if rand() < P(d(i,j)) OR rand() < ρ_long:
               add_synapse(i,j, bit_rand())
   ```

---

## 5 Reward/Drive Computation

### 5.1 Local Signals

- **Prediction Error**: `err_i = popcnt(a_i ⊕ â_i)`.
- **Firing‑Rate Deviation**: `|ρ_i − ρ*|`.
- **Sensor Health**: battery %, collision, temperature.

### 5.2 Global Bits (per tick)

| Bit  | Logic                                  | Purpose                           |
| ---- | -------------------------------------- | --------------------------------- |
| `S`  | `mean(err) > θ_S`                      | Novelty / surprise                |
| `R⁺` | `S & (Δ_loss < 0)` or external “score” | Pleasure / positive learning gate |
| `R⁻` | Any danger flag                        | Pain / negative gate              |
| `H`  | Majority of neurons within rate band   | Homeostatic comfort               |

---

## 6 Local Plasticity Rule

```c
wrong = (pre ^ post) ^ (R⁻ | !R⁺);  // sign of correlation
flip  = wrong & (S | R⁻) & bernoulli(p);
w    ^= flip;                         // atomic in register
```

- All operands live in the same 64‑bit lane → no global sync.

---

## 7 Clock Cycle (10 ms example)

1. **Sense**: read sensors, update internal variables.
2. **Reward Bits**: compute `S, R⁺, R⁻, H` (device kernel).
3. **Forward Pass**: XNOR+POPCNT for each neuron’s in‑tile synapses; sparse hops fetched from global mem.
4. **Plasticity**: apply bit‑flip rule inside each thread block.
5. **Structural Pass** (every 1 s): prune dead links, grow new using `(pre & S)` stats.

---

## 8 Hardware Path

| GPU                  | INT1 Tensor Core  | Kernel Path                       |
| -------------------- | ----------------- | --------------------------------- |
| RTX 2080 Ti (Turing) | ✔ `WMMA.B1`       | CUTLASS‑based binary GEMM         |
| GH200 (Hopper)       | ✖ (INT4/FP8 only) | Shared‑mem XNOR + `popc` fallback |

> **Optimization:** Keep both weight & activation tiles in shared memory; double‑buffer loading for full occupancy.

---

## 9 Expected Emergent Phenomena

- **Grid / Place cells** via coordinate‑tag interference.
- **Conv‑like feature maps** via local receptive field reuse.
- **Attention‑style gating** from sparse long hops.
- **Self‑balancing activity** through homeostatic bit.

---

## 10 Interfaces

- **Sensors**: any binary/quantised streams mapped to neuron IDs.
- **Actuators**: designate motor neuron groups whose spiking is read by host code.
- **Host API**: C++/CUDA core with Python bindings (PyO3 or ctypes).

---

## 11 Testing & Metrics

| Metric             | How collected                  | Note                        |
| ------------------ | ------------------------------ | --------------------------- |
| Power draw         | nvidia‑smi / NVML              | Stable < TDP at 60 Hz clock |
| Novel state ratio  | `S` duty cycle                 | Higher ⇒ better exploration |
| Mean reward        | `R⁺` minus `R⁻` rate           | Tracks overall “happiness”  |
| Emergent cell maps | offline PCA/FFT on activations | Detect grid frequency       |

---

## 12 Future Extensions

- **2‑bit or FP4 weights** on Hopper/Blackwell.
- **Spiking neuron timing jitter** (phase code).
- **Distributed multi‑GPU mesh** via NVLink.
- **FPGA/ASIC off‑load** for edge devices.

---

## 13 Glossary

- **XNOR‑Popcount:** Bitwise agreement count ≈ dot‑product.
- **Three‑factor Rule:** Local Hebbian term gated by neuromodulator.
- **Structural Plasticity:** Synaptic growth/pruning cycle.

---

## 14 Key References

1. Courbariaux & Bengio, *BinaryConnect* (2015)
2. Hubara et al., *Binarized Neural Networks* (2016)
3. Xie et al., *Bi‑Real Net* (2018)
4. Bellec et al., *Three‑Factor Hebbian Learning* (2020)
5. Cutlass Examples: `binary_gemm` (NVIDIA, 2024)


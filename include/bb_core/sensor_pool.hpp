#pragma once
#include <vector>
#include <cstdint>

namespace bb {
class SensorPool {
public:
    explicit SensorPool(unsigned count) : sensors_(count, 0) {}
    uint8_t* data() { return sensors_.data(); }
    const uint8_t* data() const { return sensors_.data(); }
    unsigned size() const { return static_cast<unsigned>(sensors_.size()); }

private:
    std::vector<uint8_t> sensors_;
};
} // namespace bb

#pragma once
#include <vector>
#include <cstdint>

namespace bb {
class ActuatorPool {
public:
    explicit ActuatorPool(unsigned count) : actions_(count, 0) {}
    uint8_t* data() { return actions_.data(); }
    const uint8_t* data() const { return actions_.data(); }
    unsigned size() const { return static_cast<unsigned>(actions_.size()); }

private:
    std::vector<uint8_t> actions_;
};
} // namespace bb

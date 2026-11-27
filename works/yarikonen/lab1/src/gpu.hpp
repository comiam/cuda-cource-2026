#pragma once

#include <array>
#include <vector>

struct Input {
    float alpha;
    std::vector<float> hX;
    std::vector<float> hY;
};

using Output = std::array<std::vector<float>, 3>;

Output compute(const Input &input, bool noncoalesced, unsigned block, unsigned repeat);

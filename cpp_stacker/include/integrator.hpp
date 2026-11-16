#pragma once

#include "types.hpp"
#include <vector>

namespace astro {

// Iter 0 Baseline: Integer shift-and-add unweighted arithmetic mean integration
std::vector<float> integrate_frames(
    const std::vector<std::vector<float>>& rgb_frames,
    const std::vector<FrameTransform>& transforms
);

} // namespace astro

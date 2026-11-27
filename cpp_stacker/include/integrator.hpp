#pragma once

#include "types.hpp"
#include <vector>

namespace astro {

// Integrate registered RGB frames onto the reference grid using partition-of-unity
// Lanczos-3 resampling, differential planar sky matching, and per-pixel temporal
// Winsorized Sigma Clipping.
std::vector<float> integrate_frames(
    const std::vector<std::vector<float>>& rgb_frames,
    const std::vector<FrameTransform>& transforms
);

} // namespace astro

#pragma once

#include "types.hpp"
#include <vector>

namespace astro {

// Register all frames against qualities[ref_index].
// Iter 0 Baseline: 2D bright-star translation consensus rounded to nearest integer pixel shift.
std::vector<FrameTransform> register_all_frames(
    const std::vector<FrameQuality>& qualities,
    int ref_index
);

} // namespace astro

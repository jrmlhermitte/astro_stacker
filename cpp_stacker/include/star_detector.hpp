#pragma once

#include "types.hpp"
#include <vector>

namespace astro {

// Extract stars and frame quality metrics from a single linear luminance frame (ACTIVE_H x ACTIVE_W)
FrameQuality analyze_frame_stars(
    const std::vector<float>& lum,
    int frame_index,
    const std::string& filename,
    int valid_rows = ACTIVE_H
);

// Automatically select the sharpest clean exposure as the registration reference frame
int select_sharpest_reference_frame(const std::vector<FrameQuality>& qualities);

} // namespace astro

#pragma once

#include "types.hpp"
#include <vector>

namespace astro {

// Register all frames against qualities[ref_index] using Valdes chiral triangle
// geometric hashing, 4-DOF similarity / 6-DOF affine refinement, photometric flux
// scaling, and PSF/SNR frame weighting.
std::vector<FrameTransform> register_all_frames(
    const std::vector<FrameQuality>& qualities,
    int ref_index
);

} // namespace astro

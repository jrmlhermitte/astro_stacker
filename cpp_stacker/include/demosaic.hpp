#pragma once

#include "types.hpp"
#include <vector>

namespace astro {

// Exact Malvar-He-Cutler (2004) 5x5 linear demosaicing on unrotated RGGB CFA
void demosaic_mhc_rggb(
    const float* cfa,
    int height,
    int width,
    float* rgb_out,
    bool use_omp = false
);

// Compute canonical RGGB photon-weighted linear luminance: L = 0.25*R + 0.50*G + 0.25*B
void compute_luminance_from_rgb(
    const float* rgb,
    int height,
    int width,
    std::vector<float>& lum_out,
    bool use_omp = false
);

} // namespace astro

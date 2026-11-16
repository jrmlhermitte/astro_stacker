#pragma once

#include "types.hpp"
#include <vector>

namespace astro {

// Build master dark (ACTIVE_H * ACTIVE_W float32) from clean dark frames
std::vector<float> build_master_dark(const std::vector<RawSensorFrame>& clean_darks);

// Calibrate a single raw light frame into linear float32 CFA (zero floor clipping)
void calibrate_light_cfa(
    const RawSensorFrame& light,
    const std::vector<float>& master_dark,
    std::vector<float>& cfa_out,
    bool use_omp = false
);

} // namespace astro

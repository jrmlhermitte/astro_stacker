#pragma once

#include "types.hpp"
#include <string>
#include <vector>

namespace astro {

enum class FlatMode {
    None,
    MotesSelf,   // Sensor-space temporal median + dip-masked band-pass dust motes only
    MotesIter,   // Alternating sky-model division dust-mote solver (cancels stars + extended targets)
    FullSelf,    // Smooth radial vignetting + MotesSelf
    FullIter     // Smooth radial vignetting + MotesIter
};

enum class FlatChromaMode {
    Coupled,     // Shared high-SNR luminance dust-mote map + per-channel smooth radial vignetting
    PerChannel   // Independent R, G, B estimation for both vignetting and dust motes
};

struct FlatFieldConfig {
    FlatMode mode = FlatMode::None;
    FlatChromaMode chroma = FlatChromaMode::Coupled;
    int iter_passes = 3;
    double mote_sigma_lo_px = 10.0;
    double mote_sigma_hi_px = 180.0;
    double mote_min_val = 0.80;
    double mote_max_val = 1.005;
    double mote_wiener_sigma = 0.003; // Wiener shrinkage noise floor (0.3% RMS) to suppress sky noise
};

struct FlatFieldStats {
    double v_min = 1.0;
    double v_max = 1.0;
    double v_median = 1.0;
    double vig_center_to_corner_drop_pct = 0.0;
    double mote_min_val = 1.0;
    double mote_rms_pct = 0.0;
    double vig_x0_px = ACTIVE_W * 0.5;
    double vig_y0_px = ACTIVE_H * 0.5;
};

FlatMode parse_flat_mode(const std::string& s);
FlatChromaMode parse_flat_chroma_mode(const std::string& s);

// Estimate empirical synthetic flat field V(x) in physical active sensor coordinates (ACTIVE_H * ACTIVE_W * 3)
std::vector<float> estimate_synthetic_flat(
    const std::vector<std::vector<float>>& rgb_frames,
    const std::vector<FrameTransform>& transforms,
    const FlatFieldConfig& cfg,
    FlatFieldStats& stats_out
);

// Apply flat field V(x) in-place to all calibrated RGB frames in sensor coordinates: I_m(x) /= V(x)
void apply_flat_field_inplace(
    std::vector<std::vector<float>>& rgb_frames,
    const std::vector<float>& flat_rgb
);

// Compute summary statistics for an existing (loaded) flat field V(x)
FlatFieldStats summarize_flat_field(const std::vector<float>& flat_rgb);

} // namespace astro

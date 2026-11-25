#pragma once

#include "types.hpp"
#include <string>
#include <vector>

namespace astro {

// Save (ACTIVE_H, ACTIVE_W, 3) float32 array in standard NumPy .npy v1.0 format
void write_npy_f32_rgb(
    const std::string& path,
    const std::vector<float>& rgb,
    int height = ACTIVE_H,
    int width = ACTIVE_W
);

// Load (ACTIVE_H, ACTIVE_W, 3) float32 array from standard NumPy .npy v1.0/v2.0 format
std::vector<float> read_npy_f32_rgb(
    const std::string& path,
    int height = ACTIVE_H,
    int width = ACTIVE_W
);

// Save per-frame registration transforms & weights in JSON format for scripts/evaluate.py
void write_transforms_json(
    const std::string& path,
    const std::string& ref_filename,
    const std::vector<FrameTransform>& transforms
);

// Save 32-bit IEEE-754 floating-point linear RGB FITS cube (NAXIS1=W, NAXIS2=H, NAXIS3=3)
void write_fits_f32_rgb(
    const std::string& path,
    const std::vector<float>& rgb,
    int height = ACTIVE_H,
    int width = ACTIVE_W
);

// Save 32-bit IEEE-754 floating-point linear RGB TIFF (SampleFormat=3, BitsPerSample=32)
void write_tiff_f32_rgb(
    const std::string& path,
    const std::vector<float>& rgb,
    int height = ACTIVE_H,
    int width = ACTIVE_W
);

// Save 16-bit auto-stretched RGB visual preview TIFF (strictly for human visual inspection)
void write_tiff_u16_preview_rgb(
    const std::string& path,
    const std::vector<float>& rgb,
    int height = ACTIVE_H,
    int width = ACTIVE_W
);

} // namespace astro

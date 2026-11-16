#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace astro {

inline constexpr int RAW_W = 4352;
inline constexpr int RAW_H = 2874;
inline constexpr int TOP_MARGIN = 18;
inline constexpr int LEFT_MARGIN = 62;
inline constexpr int ACTIVE_W = 4290;
inline constexpr int ACTIVE_H = 2856;
inline constexpr size_t ACTIVE_PIXELS = static_cast<size_t>(ACTIVE_W) * ACTIVE_H;
inline constexpr int OB_LEFT_START = 2;
inline constexpr int OB_LEFT_END = 60; // [2, 60) -> 58 cols
inline constexpr int ROI_MARGIN = 64;

struct RawFrameMeta {
    std::string filepath;
    std::string filename;
    int rc = 0;
    int data_error_count = 0;
    int first_data_error_offset = -1;
    uint16_t active_min = 0;
    uint16_t active_max = 0;
    double active_p1 = 0.0;
    double ob_left_median = 2048.0;
    float iso_speed = 0.0f;
    float shutter_s = 0.0f;
    float camera_temp_c = 0.0f;
    int64_t timestamp = 0;
    int flip_ignored = 0;
    bool cfa_is_rggb = false;
    bool is_clean = false;
    std::string reject_reason;
};

struct RawSensorFrame {
    RawFrameMeta meta;
    std::vector<uint16_t> active_cfa;    // ACTIVE_H * ACTIVE_W
    std::vector<float> ob_row_medians;   // ACTIVE_H per-row optical black medians
};

struct DetectedStar {
    double x = 0.0;          // column in [0, ACTIVE_W - 1]
    double y = 0.0;          // row in [0, ACTIVE_H - 1]
    double flux = 0.0;       // integrated flux above local background
    double peak = 0.0;       // peak luminance above background
    double fwhm_px = 0.0;    // geometric FWHM in pixels
    double eccentricity = 0.0;
    bool is_corner = false;
};

struct FrameQuality {
    int frame_index = -1;
    std::string filename;
    double sky_median = 0.0;
    double sky_rms = 0.0;
    double median_fwhm_px = 999.0;
    double corner_fwhm_px = 999.0;
    double median_eccentricity = 1.0;
    int star_count = 0;
    std::vector<DetectedStar> stars;
};

// Coordinate convention centered at (cx = 2144.5, cy = 1427.5):
// [x_frame - cx; y_frame - cy] = scale * R(dtheta) * [x_ref - cx; y_ref - cy] + [dx; dy]
// (or general 6-DOF affine matrix A_2x2 * [rx; ry] + [dx; dy])
struct FrameTransform {
    std::string filename;
    bool integrated = true;
    bool is_reference = false;
    double weight = 1.0;
    double dx = 0.0;
    double dy = 0.0;
    double dtheta_deg = 0.0;
    double scale = 1.0;
    // Direct inverse mapping matrix from output reference coords (rx, ry) relative to center
    // to source frame coords (fx, fy) relative to center:
    // fx = m00 * rx + m01 * ry + tx
    // fy = m10 * rx + m11 * ry + ty
    double m00 = 1.0;
    double m01 = 0.0;
    double m10 = 0.0;
    double m11 = 1.0;
    double tx = 0.0;
    double ty = 0.0;
    double flux_scale = 1.0; // Photometric transparency/exposure ratio F_i relative to reference
    int inlier_matches = 0;
    double residual_rms_px = 0.0;
};

} // namespace astro

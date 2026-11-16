#include "integrator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace astro {

namespace {

constexpr int LANCZOS_A = 3;
constexpr int LUT_RES = 4096; // 4096 subdivisions per unit pixel -> 24,577 table entries

struct Lanczos3Lut {
    std::array<double, 2 * LANCZOS_A * LUT_RES + 1> table{};

    Lanczos3Lut() {
        constexpr double PI = 3.14159265358979323846;
        const int center = LANCZOS_A * LUT_RES;
        for (int i = -center; i <= center; ++i) {
            const double x = static_cast<double>(i) / static_cast<double>(LUT_RES);
            if (std::abs(x) < 1e-12) {
                table[center + i] = 1.0;
            } else if (std::abs(x) >= static_cast<double>(LANCZOS_A)) {
                table[center + i] = 0.0;
            } else {
                const double px = PI * x;
                const double px_a = px / static_cast<double>(LANCZOS_A);
                table[center + i] = (std::sin(px) / px) * (std::sin(px_a) / px_a);
            }
        }
    }

    inline double eval(double x) const {
        if (x <= -LANCZOS_A || x >= LANCZOS_A) return 0.0;
        const double pos = (x + LANCZOS_A) * LUT_RES;
        const int idx = static_cast<int>(pos);
        const double frac = pos - idx;
        return table[idx] + frac * (table[idx + 1] - table[idx]);
    }
};

const Lanczos3Lut& get_lanczos3_lut() {
    static const Lanczos3Lut lut;
    return lut;
}

inline int clamp_coord(int v, int limit) {
    if (v < 0) return 0;
    if (v >= limit) return limit - 1;
    return v;
}

// Unified 2D Lanczos-3 (6x6 stencil, partition-of-unity normalized) inverse coordinate sampler.
// Used identically for sub-pixel translation AND 4-DOF/6-DOF geometric warps (Refinement 3).
inline void sample_rgb_lanczos3_2d(
    const float* src_rgb,
    double fx,
    double fy,
    double& out_r,
    double& out_g,
    double& out_b
) {
    const int ix0 = static_cast<int>(std::floor(fx));
    const int iy0 = static_cast<int>(std::floor(fy));
    const double dx_frac = fx - ix0;
    const double dy_frac = fy - iy0;

    if (std::abs(dx_frac) < 1e-9 && std::abs(dy_frac) < 1e-9) {
        const int sc = clamp_coord(ix0, ACTIVE_W);
        const int sr = clamp_coord(iy0, ACTIVE_H);
        const float* p = src_rgb + (static_cast<size_t>(sr) * ACTIVE_W + sc) * 3;
        out_r = p[0];
        out_g = p[1];
        out_b = p[2];
        return;
    }

    const auto& lut = get_lanczos3_lut();
    double wx[6];
    double wy[6];
    double sum_wx = 0.0;
    double sum_wy = 0.0;
    for (int k = 0; k < 6; ++k) {
        const int offset = k - 2; // {-2, -1, 0, +1, +2, +3}
        wx[k] = lut.eval(dx_frac - offset);
        wy[k] = lut.eval(dy_frac - offset);
        sum_wx += wx[k];
        sum_wy += wy[k];
    }
    const double inv_wx = 1.0 / sum_wx;
    const double inv_wy = 1.0 / sum_wy;
    for (int k = 0; k < 6; ++k) {
        wx[k] *= inv_wx;
        wy[k] *= inv_wy;
    }

    int cols[6];
    for (int k = 0; k < 6; ++k) {
        cols[k] = clamp_coord(ix0 + k - 2, ACTIVE_W) * 3;
    }

    double acc_r = 0.0;
    double acc_g = 0.0;
    double acc_b = 0.0;

    for (int ky = 0; ky < 6; ++ky) {
        const int sr = clamp_coord(iy0 + ky - 2, ACTIVE_H);
        const float* row_ptr = src_rgb + static_cast<size_t>(sr) * ACTIVE_W * 3;
        double row_r = 0.0;
        double row_g = 0.0;
        double row_b = 0.0;
        for (int kx = 0; kx < 6; ++kx) {
            const float* px = row_ptr + cols[kx];
            const double w = wx[kx];
            row_r += w * static_cast<double>(px[0]);
            row_g += w * static_cast<double>(px[1]);
            row_b += w * static_cast<double>(px[2]);
        }
        const double wy_k = wy[ky];
        acc_r += wy_k * row_r;
        acc_g += wy_k * row_g;
        acc_b += wy_k * row_b;
    }

    out_r = acc_r;
    out_g = acc_g;
    out_b = acc_b;
}

} // namespace

std::vector<float> integrate_frames(
    const std::vector<std::vector<float>>& rgb_frames,
    const std::vector<FrameTransform>& transforms
) {
    const int n_frames = static_cast<int>(rgb_frames.size());
    if (n_frames == 0 || static_cast<int>(transforms.size()) != n_frames) {
        throw std::runtime_error("Invalid frame/transform count in integrate_frames");
    }

    int active_count = 0;
    for (const auto& t : transforms) {
        if (t.integrated && t.weight > 0.0) ++active_count;
    }
    if (active_count == 0) {
        throw std::runtime_error("Zero integrated frames in integrate_frames");
    }

    // Touch static LUT before entering parallel region
    (void)get_lanczos3_lut();

    std::vector<float> stack_out(ACTIVE_PIXELS * 3, 0.0f);
    const double inv_count = 1.0 / static_cast<double>(active_count);
    const double cx_center = (ACTIVE_W - 1) * 0.5;
    const double cy_center = (ACTIVE_H - 1) * 0.5;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < ACTIVE_H; ++r) {
        const double ry = static_cast<double>(r) - cy_center;
        float* dst_row = stack_out.data() + static_cast<size_t>(r) * ACTIVE_W * 3;

        for (int c = 0; c < ACTIVE_W; ++c) {
            const double rx = static_cast<double>(c) - cx_center;
            double sum_r = 0.0;
            double sum_g = 0.0;
            double sum_b = 0.0;

            for (int f = 0; f < n_frames; ++f) {
                const auto& tr = transforms[f];
                if (!tr.integrated || tr.weight <= 0.0) continue;

                const double fx = cx_center + (tr.m00 * rx + tr.m01 * ry + tr.tx);
                const double fy = cy_center + (tr.m10 * rx + tr.m11 * ry + tr.ty);

                double sr = 0.0, sg = 0.0, sb = 0.0;
                sample_rgb_lanczos3_2d(rgb_frames[f].data(), fx, fy, sr, sg, sb);
                sum_r += sr;
                sum_g += sg;
                sum_b += sb;
            }

            dst_row[3 * c + 0] = static_cast<float>(sum_r * inv_count);
            dst_row[3 * c + 1] = static_cast<float>(sum_g * inv_count);
            dst_row[3 * c + 2] = static_cast<float>(sum_b * inv_count);
        }
    }

    return stack_out;
}

} // namespace astro

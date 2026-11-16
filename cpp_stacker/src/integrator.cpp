#include "integrator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace astro {

namespace {

constexpr int LANCZOS_A = 3;
constexpr int LUT_RES = 4096;

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
        const int offset = k - 2;
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

// Per-channel 1st-degree planar differential sky-glow model (Guardrail 4 compliant: ONLY 3 scalar
// plane coefficients [c0, cx, cy] per channel, fitted to I_frame/F_i - I_ref so extended nebulae
// and galaxy disks cancel identically!).
struct SkyDiffPlane {
    double c0[3] = {0.0, 0.0, 0.0};
    double cx[3] = {0.0, 0.0, 0.0};
    double cy[3] = {0.0, 0.0, 0.0};
};

bool solve_3x3_plane(double A[3][3], const double b[3], double x[3]) {
    double M[3][4];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) M[i][j] = A[i][j];
        M[i][3] = b[i];
    }
    for (int col = 0; col < 3; ++col) {
        int piv = col;
        for (int r = col + 1; r < 3; ++r) {
            if (std::abs(M[r][col]) > std::abs(M[piv][col])) piv = r;
        }
        if (std::abs(M[piv][col]) < 1e-12) return false;
        if (piv != col) {
            for (int j = col; j < 4; ++j) std::swap(M[col][j], M[piv][j]);
        }
        const double inv = 1.0 / M[col][col];
        for (int j = col; j < 4; ++j) M[col][j] *= inv;
        for (int r = 0; r < 3; ++r) {
            if (r == col) continue;
            const double f = M[r][col];
            for (int j = col; j < 4; ++j) M[r][j] -= f * M[col][j];
        }
    }
    for (int i = 0; i < 3; ++i) x[i] = M[i][3];
    return true;
}

SkyDiffPlane fit_frame_sky_diff_plane(
    const float* frame_rgb,
    const float* ref_rgb,
    const FrameTransform& tr
) {
    SkyDiffPlane plane;
    if (tr.is_reference) return plane;

    const double cx_center = (ACTIVE_W - 1) * 0.5;
    const double cy_center = (ACTIVE_H - 1) * 0.5;
    const double inv_flux = 1.0 / std::max(0.25, tr.flux_scale);

    struct BlockSample {
        double nx;
        double ny;
        double diff_med[3];
    };
    std::vector<BlockSample> blocks;

    constexpr int BSIZE = 96;
    std::vector<double> d_buf[3];
    for (int ch = 0; ch < 3; ++ch) d_buf[ch].reserve(64);

    for (int r0 = ROI_MARGIN; r0 + BSIZE <= ACTIVE_H - ROI_MARGIN; r0 += BSIZE) {
        for (int c0 = ROI_MARGIN; c0 + BSIZE <= ACTIVE_W - ROI_MARGIN; c0 += BSIZE) {
            for (int ch = 0; ch < 3; ++ch) d_buf[ch].clear();

            for (int r = r0 + 6; r < r0 + BSIZE; r += 12) {
                const double ry = static_cast<double>(r) - cy_center;
                for (int c = c0 + 6; c < c0 + BSIZE; c += 12) {
                    const double rx = static_cast<double>(c) - cx_center;
                    const double fx = cx_center + (tr.m00 * rx + tr.m01 * ry + tr.tx);
                    const double fy = cy_center + (tr.m10 * rx + tr.m11 * ry + tr.ty);
                    if (fx < 8.0 || fx >= ACTIVE_W - 8.0 || fy < 8.0 || fy >= ACTIVE_H - 8.0) {
                        continue;
                    }
                    double sr = 0.0, sg = 0.0, sb = 0.0;
                    sample_rgb_lanczos3_2d(frame_rgb, fx, fy, sr, sg, sb);
                    const float* rp = ref_rgb + (static_cast<size_t>(r) * ACTIVE_W + c) * 3;
                    // Reject saturated or bright star core samples where PSF FWHM differences exist
                    const double lum_r = 0.25 * rp[0] + 0.50 * rp[1] + 0.25 * rp[2];
                    if (lum_r > 800.0) continue;

                    d_buf[0].push_back(sr * inv_flux - static_cast<double>(rp[0]));
                    d_buf[1].push_back(sg * inv_flux - static_cast<double>(rp[1]));
                    d_buf[2].push_back(sb * inv_flux - static_cast<double>(rp[2]));
                }
            }

            if (d_buf[0].size() >= 24) {
                BlockSample bs{};
                bs.nx = ((c0 + 0.5 * BSIZE) - cx_center) / ACTIVE_W;
                bs.ny = ((r0 + 0.5 * BSIZE) - cy_center) / ACTIVE_H;
                for (int ch = 0; ch < 3; ++ch) {
                    const size_t mid = d_buf[ch].size() / 2;
                    std::nth_element(d_buf[ch].begin(), d_buf[ch].begin() + mid, d_buf[ch].end());
                    bs.diff_med[ch] = d_buf[ch][mid];
                }
                blocks.push_back(bs);
            }
        }
    }

    if (blocks.size() < 16) return plane;

    // 3-pass Huber IRLS fit of 1st-degree plane [c0, cx, cy] per channel
    for (int ch = 0; ch < 3; ++ch) {
        double p[3] = {0.0, 0.0, 0.0};
        for (int pass = 0; pass < 3; ++pass) {
            double A[3][3] = {};
            double rhs[3] = {};
            for (const auto& b : blocks) {
                const double pred = p[0] + p[1] * b.nx + p[2] * b.ny;
                const double resid = b.diff_med[ch] - pred;
                const double w = (pass == 0 || std::abs(resid) <= 4.0) ? 1.0 : (4.0 / std::abs(resid));
                const double u[3] = {1.0, b.nx, b.ny};
                for (int r = 0; r < 3; ++r) {
                    rhs[r] += w * u[r] * b.diff_med[ch];
                    for (int c = 0; c < 3; ++c) {
                        A[r][c] += w * u[r] * u[c];
                    }
                }
            }
            double sol[3];
            if (solve_3x3_plane(A, rhs, sol)) {
                p[0] = sol[0]; p[1] = sol[1]; p[2] = sol[2];
            }
        }
        plane.c0[ch] = p[0];
        plane.cx[ch] = p[1];
        plane.cy[ch] = p[2];
    }
    return plane;
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

    int ref_idx = 0;
    double weight_sum = 0.0;
    for (int i = 0; i < n_frames; ++i) {
        if (transforms[i].is_reference) ref_idx = i;
        if (transforms[i].integrated && transforms[i].weight > 0.0) {
            weight_sum += transforms[i].weight;
        }
    }
    if (weight_sum <= 0.0) {
        throw std::runtime_error("Zero total weight in integrate_frames");
    }

    (void)get_lanczos3_lut();

    // Fit per-frame 1st-degree differential sky-glow planes in parallel
    std::vector<SkyDiffPlane> sky_planes(n_frames);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int f = 0; f < n_frames; ++f) {
        if (transforms[f].integrated && transforms[f].weight > 0.0) {
            sky_planes[f] = fit_frame_sky_diff_plane(
                rgb_frames[f].data(),
                rgb_frames[ref_idx].data(),
                transforms[f]
            );
        }
    }

    std::vector<float> stack_out(ACTIVE_PIXELS * 3, 0.0f);
    const double inv_w_sum = 1.0 / weight_sum;
    const double cx_center = (ACTIVE_W - 1) * 0.5;
    const double cy_center = (ACTIVE_H - 1) * 0.5;
    const double inv_w_norm = 1.0 / static_cast<double>(ACTIVE_W);
    const double inv_h_norm = 1.0 / static_cast<double>(ACTIVE_H);

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < ACTIVE_H; ++r) {
        const double ry = static_cast<double>(r) - cy_center;
        const double ny = ry * inv_h_norm;
        float* dst_row = stack_out.data() + static_cast<size_t>(r) * ACTIVE_W * 3;

        for (int c = 0; c < ACTIVE_W; ++c) {
            const double rx = static_cast<double>(c) - cx_center;
            const double nx = rx * inv_w_norm;
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

                const double inv_f = 1.0 / std::max(0.25, tr.flux_scale);
                const auto& sp = sky_planes[f];
                const double cal_r = sr * inv_f - (sp.c0[0] + sp.cx[0] * nx + sp.cy[0] * ny);
                const double cal_g = sg * inv_f - (sp.c0[1] + sp.cx[1] * nx + sp.cy[1] * ny);
                const double cal_b = sb * inv_f - (sp.c0[2] + sp.cx[2] * nx + sp.cy[2] * ny);

                const double w = tr.weight;
                sum_r += w * cal_r;
                sum_g += w * cal_g;
                sum_b += w * cal_b;
            }

            dst_row[3 * c + 0] = static_cast<float>(sum_r * inv_w_sum);
            dst_row[3 * c + 1] = static_cast<float>(sum_g * inv_w_sum);
            dst_row[3 * c + 2] = static_cast<float>(sum_b * inv_w_sum);
        }
    }

    return stack_out;
}

} // namespace astro

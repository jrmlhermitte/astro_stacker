#include "flat_field.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace astro {

namespace {

constexpr int BLK = 4;
constexpr int GH = (ACTIVE_H + BLK - 1) / BLK; // 714
constexpr int GW = (ACTIVE_W + BLK - 1) / BLK; // 1073
constexpr size_t GPIX = static_cast<size_t>(GH) * GW;

inline double median_small_d(double* a, int n) {
    if (n <= 0) return 1.0;
    for (int i = 1; i < n; ++i) {
        const double key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = key;
    }
    const int mid = n / 2;
    return ((n & 1) == 1) ? a[mid] : 0.5 * (a[mid - 1] + a[mid]);
}

inline double percentile_small_d(double* a, int n, double q) {
    if (n <= 0) return 1.0;
    std::sort(a, a + n);
    const double pos = std::clamp(q, 0.0, 1.0) * (n - 1);
    const int idx = static_cast<int>(pos);
    const double frac = pos - idx;
    if (idx + 1 >= n) return a[n - 1];
    return a[idx] + frac * (a[idx + 1] - a[idx]);
}

// Separable normalized-convolution Gaussian blur on (GH x GW) float64 grid
std::vector<double> gaussian_blur_masked(
    const std::vector<double>& src,
    const std::vector<uint8_t>& mask,
    double sigma
) {
    if (sigma <= 0.1) return src;
    const int rad = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
    std::vector<double> k(2 * rad + 1);
    double ksum = 0.0;
    const double inv2s2 = 0.5 / (sigma * sigma);
    for (int i = -rad; i <= rad; ++i) {
        k[i + rad] = std::exp(-static_cast<double>(i * i) * inv2s2);
        ksum += k[i + rad];
    }
    for (double& v : k) v /= ksum;

    std::vector<double> tmp_num(GPIX, 0.0);
    std::vector<double> tmp_den(GPIX, 0.0);

    // Horizontal pass
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < GH; ++r) {
        const size_t roff = static_cast<size_t>(r) * GW;
        for (int c = 0; c < GW; ++c) {
            double sn = 0.0;
            double sd = 0.0;
            for (int d = -rad; d <= rad; ++d) {
                const int cc = std::clamp(c + d, 0, GW - 1);
                const size_t idx = roff + cc;
                if (mask[idx]) {
                    const double w = k[d + rad];
                    sn += w * src[idx];
                    sd += w;
                }
            }
            tmp_num[roff + c] = sn;
            tmp_den[roff + c] = sd;
        }
    }

    // Vertical pass
    std::vector<double> out(GPIX, 1.0);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < GH; ++r) {
        const size_t roff = static_cast<size_t>(r) * GW;
        for (int c = 0; c < GW; ++c) {
            double sn = 0.0;
            double sd = 0.0;
            for (int d = -rad; d <= rad; ++d) {
                const int rr = std::clamp(r + d, 0, GH - 1);
                const size_t idx = static_cast<size_t>(rr) * GW + c;
                const double w = k[d + rad];
                sn += w * tmp_num[idx];
                sd += w * tmp_den[idx];
            }
            out[roff + c] = (sd > 1e-9) ? (sn / sd) : src[roff + c];
        }
    }
    return out;
}

// Bilinear sampling on (GH x GW) grid at fractional grid coords (gx, gy)
inline double sample_grid_bilinear(const std::vector<double>& g, double gx, double gy) {
    const double cx = std::clamp(gx, 0.0, static_cast<double>(GW - 1));
    const double cy = std::clamp(gy, 0.0, static_cast<double>(GH - 1));
    const int x0 = static_cast<int>(cx);
    const int y0 = static_cast<int>(cy);
    const int x1 = std::min(x0 + 1, GW - 1);
    const int y1 = std::min(y0 + 1, GH - 1);
    const double fx = cx - x0;
    const double fy = cy - y0;
    const double v00 = g[static_cast<size_t>(y0) * GW + x0];
    const double v10 = g[static_cast<size_t>(y0) * GW + x1];
    const double v01 = g[static_cast<size_t>(y1) * GW + x0];
    const double v11 = g[static_cast<size_t>(y1) * GW + x1];
    return (1.0 - fy) * ((1.0 - fx) * v00 + fx * v10) +
           fy * ((1.0 - fx) * v01 + fx * v11);
}

// Compute per-frame, per-channel sky medians and 4x4 normalized grids (4 channels: R, G, B, L)
void build_frame_grids(
    const std::vector<std::vector<float>>& rgb_frames,
    const std::vector<FrameTransform>& transforms,
    std::vector<std::array<std::vector<double>, 4>>& frame_grids,
    std::vector<int>& valid_grid_rows
) {
    const int n_frames = static_cast<int>(rgb_frames.size());
    frame_grids.resize(n_frames);
    valid_grid_rows.resize(n_frames, GH);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int f = 0; f < n_frames; ++f) {
        const float* src = rgb_frames[f].data();
        const int vrows = std::clamp(transforms[f].valid_rows, 0, ACTIVE_H);
        valid_grid_rows[f] = vrows / BLK;

        double med_ch[4] = {1.0, 1.0, 1.0, 1.0};
        for (int ch = 0; ch < 4; ++ch) {
            std::vector<double> samp;
            samp.reserve(60000);
            const int r_lim = std::max(ROI_MARGIN + 16, vrows - ROI_MARGIN);
            for (int r = ROI_MARGIN; r < r_lim; r += 16) {
                for (int c = ROI_MARGIN; c < ACTIVE_W - ROI_MARGIN; c += 16) {
                    const float* p = src + (static_cast<size_t>(r) * ACTIVE_W + c) * 3;
                    const double v = (ch < 3) ? static_cast<double>(p[ch])
                                              : (0.25 * p[0] + 0.50 * p[1] + 0.25 * p[2]);
                    samp.push_back(v);
                }
            }
            if (!samp.empty()) {
                const size_t mid = samp.size() / 2;
                std::nth_element(samp.begin(), samp.begin() + mid, samp.end());
                med_ch[ch] = std::max(1.0, samp[mid]);
            }
        }

        for (int ch = 0; ch < 4; ++ch) {
            frame_grids[f][ch].assign(GPIX, 1.0);
        }

        double blk_buf[4][BLK * BLK];
        for (int gr = 0; gr < valid_grid_rows[f]; ++gr) {
            const int r0 = gr * BLK;
            const int r1 = std::min(r0 + BLK, vrows);
            for (int gc = 0; gc < GW; ++gc) {
                const int c0 = gc * BLK;
                const int c1 = std::min(c0 + BLK, ACTIVE_W);
                int cnt = 0;
                for (int r = r0; r < r1; ++r) {
                    const float* row_p = src + static_cast<size_t>(r) * ACTIVE_W * 3;
                    for (int c = c0; c < c1; ++c) {
                        const float* p = row_p + c * 3;
                        blk_buf[0][cnt] = p[0];
                        blk_buf[1][cnt] = p[1];
                        blk_buf[2][cnt] = p[2];
                        blk_buf[3][cnt] = 0.25 * p[0] + 0.50 * p[1] + 0.25 * p[2];
                        ++cnt;
                    }
                }
                const size_t gidx = static_cast<size_t>(gr) * GW + gc;
                for (int ch = 0; ch < 4; ++ch) {
                    // 40th percentile suppresses compact star cores inside 4x4 cell
                    frame_grids[f][ch][gidx] = percentile_small_d(blk_buf[ch], cnt, 0.40) / med_ch[ch];
                }
            }
        }
    }
}

// Compute sensor-space temporal lower-clipped mean across all frames for a single channel
std::vector<double> compute_sensor_space_median(
    const std::vector<std::array<std::vector<double>, 4>>& frame_grids,
    const std::vector<int>& valid_grid_rows,
    int ch
) {
    const int n_frames = static_cast<int>(frame_grids.size());
    std::vector<double> out(GPIX, 1.0);

    #pragma omp parallel for schedule(static)
    for (int gr = 0; gr < GH; ++gr) {
        double vals[64];
        double devs[64];
        const size_t roff = static_cast<size_t>(gr) * GW;
        for (int gc = 0; gc < GW; ++gc) {
            int cnt = 0;
            for (int f = 0; f < n_frames && cnt < 64; ++f) {
                if (gr < valid_grid_rows[f]) {
                    vals[cnt++] = frame_grids[f][ch][roff + gc];
                }
            }
            if (cnt == 0) {
                out[roff + gc] = 1.0;
                continue;
            }
            double tmp[64];
            for (int i = 0; i < cnt; ++i) tmp[i] = vals[i];
            const double med = median_small_d(tmp, cnt);
            for (int i = 0; i < cnt; ++i) devs[i] = std::abs(vals[i] - med);
            const double mad = std::max(1e-4, median_small_d(devs, cnt) * 1.4826);

            // Asymmetric sigma clip: reject drifting stars (+1.5*mad), keep dust-mote signal (-3.0*mad)
            const double lo = med - 3.0 * mad;
            const double hi = med + 1.5 * mad;
            double sum = 0.0;
            int kept = 0;
            for (int i = 0; i < cnt; ++i) {
                if (vals[i] >= lo && vals[i] <= hi) {
                    sum += vals[i];
                    ++kept;
                }
            }
            out[roff + gc] = (kept > 0) ? (sum / kept) : med;
        }
    }
    return out;
}

// Extract mid-frequency dust-mote map V_motes from a vignetting-detrended sensor-space grid
// using 2-pass dip-masked continuum normalization.
std::vector<double> extract_mote_field_from_grid(
    const std::vector<double>& sensor_grid_detrended,
    const FlatFieldConfig& cfg
) {
    const double sig_lo = cfg.mote_sigma_lo_px / static_cast<double>(BLK);
    const double sig_hi = cfg.mote_sigma_hi_px / static_cast<double>(BLK);

    std::vector<uint8_t> all_ones(GPIX, 1);
    const std::vector<double> l_sm = gaussian_blur_masked(sensor_grid_detrended, all_ones, sig_lo);
    const std::vector<double> c0 = gaussian_blur_masked(l_sm, all_ones, sig_hi);

    // 2-pass dip-masking: mask absorptive dust-mote dips (< 0.995) and bright compact sources (> 1.008)
    // so deep motes do not depress their own continuum and create bright halos.
    std::vector<double> c_cur = c0;
    std::vector<uint8_t> cont_mask(GPIX, 1);
    for (int mp = 0; mp < 2; ++mp) {
        std::fill(cont_mask.begin(), cont_mask.end(), 1);
        for (size_t i = 0; i < GPIX; ++i) {
            const double rat = l_sm[i] / std::max(1e-4, c_cur[i]);
            if (rat < 0.995 || rat > 1.008) {
                cont_mask[i] = 0;
            }
        }
        c_cur = gaussian_blur_masked(l_sm, cont_mask, sig_hi);
    }
    const std::vector<double>& c1 = c_cur;

    // Normalize so median over unmasked quiet sky is exactly 1.0000
    std::vector<double> quiet_rats;
    quiet_rats.reserve(GPIX / 4);
    for (size_t i = 0; i < GPIX; i += 4) {
        if (cont_mask[i]) {
            quiet_rats.push_back(l_sm[i] / std::max(1e-4, c1[i]));
        }
    }
    double norm_med = 1.0;
    if (!quiet_rats.empty()) {
        const size_t mid = quiet_rats.size() / 2;
        std::nth_element(quiet_rats.begin(), quiet_rats.begin() + mid, quiet_rats.end());
        norm_med = std::max(0.5, quiet_rats[mid]);
    }

    std::vector<double> v_motes(GPIX, 1.0);
    const double w_sig2 = cfg.mote_wiener_sigma * cfg.mote_wiener_sigma;
    for (size_t i = 0; i < GPIX; ++i) {
        double rat = (l_sm[i] / std::max(1e-4, c1[i])) / norm_med;
        if (w_sig2 > 0.0) {
            const double d = rat - 1.0;
            const double d2 = d * d;
            // Wiener shrinkage: shrinks tiny <0.3% noise ripples toward 1.0 while preserving 1-10% dust motes
            rat = 1.0 + d * (d2 / (d2 + w_sig2));
        }
        v_motes[i] = std::clamp(rat, cfg.mote_min_val, cfg.mote_max_val);
    }
    return v_motes;
}

// Iterative alternating sky-model division to isolate true sensor-locked dust motes
// while identically cancelling stars, M57, and M31's core & dust lanes.
std::vector<double> refine_motes_iterative_sky(
    const std::vector<std::array<std::vector<double>, 4>>& frame_grids,
    const std::vector<int>& valid_grid_rows,
    const std::vector<FrameTransform>& transforms,
    const std::vector<double>& v_init,
    const FlatFieldConfig& cfg,
    int ch
) {
    const int n_frames = static_cast<int>(frame_grids.size());
    std::vector<double> v_motes = v_init;

    const double cx_full = (ACTIVE_W - 1) * 0.5;
    const double cy_full = (ACTIVE_H - 1) * 0.5;
    const double inv_blk = 1.0 / static_cast<double>(BLK);

    // Precompute inverse transforms (sensor -> reference sky)
    struct InvTr {
        double i00, i01, i10, i11, tx, ty;
        bool valid;
    };
    std::vector<InvTr> inv_tr(n_frames);
    for (int f = 0; f < n_frames; ++f) {
        const auto& t = transforms[f];
        const double det = t.m00 * t.m11 - t.m01 * t.m10;
        if (t.integrated && std::abs(det) > 1e-9) {
            inv_tr[f] = {t.m11 / det, -t.m01 / det, -t.m10 / det, t.m00 / det, t.tx, t.ty, true};
        } else {
            inv_tr[f] = {1, 0, 0, 1, 0, 0, false};
        }
    }

    const double sig_lo = cfg.mote_sigma_lo_px / static_cast<double>(BLK);
    const double sig_hi = cfg.mote_sigma_hi_px / static_cast<double>(BLK);
    std::vector<uint8_t> all_ones(GPIX, 1);

    for (int pass = 0; pass < cfg.iter_passes; ++pass) {
        // 1. Build sky-aligned median stack S_sky on 4x4 grid using current V_motes
        std::vector<double> s_sky(GPIX, 1.0);
        #pragma omp parallel for schedule(static)
        for (int gr = 0; gr < GH; ++gr) {
            const double ry = (gr + 0.5) * BLK - cy_full;
            const size_t roff = static_cast<size_t>(gr) * GW;
            double vals[64];
            for (int gc = 0; gc < GW; ++gc) {
                const double rx = (gc + 0.5) * BLK - cx_full;
                int cnt = 0;
                for (int f = 0; f < n_frames && cnt < 64; ++f) {
                    if (!inv_tr[f].valid) continue;
                    const auto& t = transforms[f];
                    const double fx = cx_full + (t.m00 * rx + t.m01 * ry + t.tx);
                    const double fy = cy_full + (t.m10 * rx + t.m11 * ry + t.ty);
                    const double gx = fx * inv_blk - 0.5;
                    const double gy = fy * inv_blk - 0.5;
                    if (gx < 1.0 || gx >= GW - 2.0 || gy < 1.0 || gy >= valid_grid_rows[f] - 2.0) continue;
                    const double raw_v = sample_grid_bilinear(frame_grids[f][ch], gx, gy);
                    const double flat_v = sample_grid_bilinear(v_motes, gx, gy);
                    vals[cnt++] = raw_v / std::max(0.2, flat_v);
                }
                s_sky[roff + gc] = (cnt > 0) ? median_small_d(vals, cnt) : 1.0;
            }
        }

        // 2. Compute sensor-space temporal median of (I_f / V_motes) / S_sky(T_f^-1(x))
        std::vector<double> delta_v(GPIX, 1.0);
        #pragma omp parallel for schedule(static)
        for (int gr = 0; gr < GH; ++gr) {
            const double fy = (gr + 0.5) * BLK - cy_full;
            const size_t roff = static_cast<size_t>(gr) * GW;
            double rats[64];
            for (int gc = 0; gc < GW; ++gc) {
                const double fx = (gc + 0.5) * BLK - cx_full;
                const double cur_v = std::max(0.2, v_motes[roff + gc]);
                int cnt = 0;
                for (int f = 0; f < n_frames && cnt < 64; ++f) {
                    if (!inv_tr[f].valid || gr >= valid_grid_rows[f]) continue;
                    const auto& it = inv_tr[f];
                    const double dfx = fx - it.tx;
                    const double dfy = fy - it.ty;
                    const double rx = cx_full + (it.i00 * dfx + it.i01 * dfy);
                    const double ry = cy_full + (it.i10 * dfx + it.i11 * dfy);
                    const double gx = rx * inv_blk - 0.5;
                    const double gy = ry * inv_blk - 0.5;
                    if (gx < 1.0 || gx >= GW - 2.0 || gy < 1.0 || gy >= GH - 2.0) continue;
                    const double sky_v = sample_grid_bilinear(s_sky, gx, gy);
                    if (sky_v <= 0.1) continue;
                    const double meas = frame_grids[f][ch][roff + gc] / cur_v;
                    rats[cnt++] = meas / sky_v;
                }
                delta_v[roff + gc] = (cnt >= 5) ? std::clamp(median_small_d(rats, cnt), 0.85, 1.15) : 1.0;
            }
        }

        // 3. Band-pass delta_v so only mid-frequency sensor-locked dust motes update
        const std::vector<double> dv_sm = gaussian_blur_masked(delta_v, all_ones, sig_lo);
        const std::vector<double> dv_cont = gaussian_blur_masked(dv_sm, all_ones, sig_hi);
        for (size_t i = 0; i < GPIX; ++i) {
            const double upd = dv_sm[i] / std::max(0.5, dv_cont[i]);
            v_motes[i] = std::clamp(v_motes[i] * upd, cfg.mote_min_val, cfg.mote_max_val);
        }
    }

    return v_motes;
}

// Solve small NxN linear system via Gaussian elimination with partial pivoting
bool solve_linear_system(int n, std::vector<double>& A, std::vector<double>& b, std::vector<double>& x) {
    x.assign(n, 0.0);
    for (int col = 0; col < n; ++col) {
        int piv = col;
        double best = std::abs(A[col * n + col]);
        for (int r = col + 1; r < n; ++r) {
            const double v = std::abs(A[r * n + col]);
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        if (best < 1e-12) return false;
        if (piv != col) {
            for (int c = col; c < n; ++c) std::swap(A[col * n + c], A[piv * n + c]);
            std::swap(b[col], b[piv]);
        }
        const double inv = 1.0 / A[col * n + col];
        for (int c = col; c < n; ++c) A[col * n + c] *= inv;
        b[col] *= inv;
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            const double f = A[r * n + col];
            for (int c = col; c < n; ++c) A[r * n + c] -= f * A[col * n + c];
            b[r] -= f * b[col];
        }
    }
    for (int i = 0; i < n; ++i) x[i] = b[i];
    return true;
}

// Fit smooth radial optical vignetting V_vig(x, y) = norm(1 + c2*r^2 + c4*r^4 + c6*r^6)
// around fitted optical axis (x0, y0) using lower-envelope radial initialization +
// asymmetric IRLS (heavily downweights positive celestial emission like M31!).
std::vector<double> fit_radial_vignetting(
    const std::vector<double>& sensor_grid_mote_corr,
    double& out_x0_px,
    double& out_y0_px
) {
    const double cx0 = (ACTIVE_W - 1) * 0.5;
    const double cy0 = (ACTIVE_H - 1) * 0.5;
    const double r_max = std::hypot(ACTIVE_W * 0.5, ACTIVE_H * 0.5);

    // Step 1: Initialize pure radial profile from lower-25th percentile in 20 radial annuli
    // (immune to M31 extended disk along major axis!).
    constexpr int NBINS = 20;
    std::vector<std::vector<double>> r_bins(NBINS);
    for (int gr = 8; gr < GH - 8; gr += 2) {
        const double py = (gr + 0.5) * BLK;
        for (int gc = 8; gc < GW - 8; gc += 2) {
            const double px = (gc + 0.5) * BLK;
            const double rn = std::hypot(px - cx0, py - cy0) / r_max;
            const int b = std::clamp(static_cast<int>(rn * NBINS), 0, NBINS - 1);
            const double val = sensor_grid_mote_corr[static_cast<size_t>(gr) * GW + gc];
            if (std::isfinite(val) && val > 0.1) {
                r_bins[b].push_back(val);
            }
        }
    }

    // Fit initial 4-param radial polynomial (c0 + c2*r^2 + c4*r^4 + c6*r^6) on bin 25th percentiles
    std::vector<double> p_rad = {1.0, -0.30, 0.0, 0.0};
    {
        std::vector<double> A(16, 0.0), rhs(4, 0.0);
        for (int b = 0; b < NBINS; ++b) {
            if (r_bins[b].size() < 20) continue;
            const double r_mid = (b + 0.5) / static_cast<double>(NBINS);
            const double r2 = r_mid * r_mid;
            // Use 25th percentile to track true sky floor even when M31 occupies part of inner bins
            const size_t qidx = static_cast<size_t>(0.25 * r_bins[b].size());
            std::nth_element(r_bins[b].begin(), r_bins[b].begin() + qidx, r_bins[b].end());
            const double y = r_bins[b][qidx];
            const double u[4] = {1.0, r2, r2 * r2, r2 * r2 * r2};
            for (int i = 0; i < 4; ++i) {
                rhs[i] += u[i] * y;
                for (int j = 0; j < 4; ++j) A[i * 4 + j] += u[i] * u[j];
            }
        }
        // Small ridge on c6 for stability
        A[3 * 4 + 3] += 1e-3;
        std::vector<double> sol;
        if (solve_linear_system(4, A, rhs, sol)) {
            p_rad = sol;
        }
    }

    // Step 2: Build clean quiet-sky mask using initial lower-envelope radial fit
    // (excludes M31 extended disk & bright stars where y - pred0 > +0.055),
    // then run SYMMETRIC Huber IRLS on quiet sky so noisy corners have ZERO downward bias!
    std::vector<double> p6 = {p_rad[0], 0.0, 0.0, p_rad[1], p_rad[2], p_rad[3]};
    for (int pass = 0; pass < 6; ++pass) {
        std::vector<double> A(36, 0.0), rhs(6, 0.0);
        for (int gr = 8; gr < GH - 8; gr += 2) {
            const double ny = ((gr + 0.5) * BLK - cy0) / r_max;
            const size_t roff = static_cast<size_t>(gr) * GW;
            for (int gc = 8; gc < GW - 8; gc += 2) {
                const double nx = ((gc + 0.5) * BLK - cx0) / r_max;
                const double r2 = nx * nx + ny * ny;
                const double y = sensor_grid_mote_corr[roff + gc];
                if (!std::isfinite(y) || y <= 0.1) continue;

                const double u[6] = {1.0, nx, ny, r2, r2 * r2, r2 * r2 * r2};
                double pred = 0.0;
                for (int i = 0; i < 6; ++i) pred += p6[i] * u[i];
                const double resid = y - pred;

                // Hard-exclude bright extended galaxy emission (M31 core/disk) above +0.055,
                // then use strictly SYMMETRIC Huber weights (delta = 0.025) on quiet sky.
                if (resid > 0.055) continue;
                const double abs_r = std::abs(resid);
                const double w = (abs_r <= 0.025) ? 1.0 : (0.025 / abs_r);

                for (int i = 0; i < 6; ++i) {
                    rhs[i] += w * u[i] * y;
                    for (int j = 0; j < 6; ++j) A[i * 6 + j] += w * u[i] * u[j];
                }
            }
        }
        A[5 * 6 + 5] += 1e-2; // Gentle regularization on r^6
        std::vector<double> sol;
        if (solve_linear_system(6, A, rhs, sol)) {
            p6 = sol;
        }
    }

    // Extract optical center offset from linear terms: c2*(nx^2+ny^2) + cx*nx + cy*ny
    const double c2_eff = std::min(-0.05, p6[3]);
    const double dx_opt_norm = std::clamp(-p6[1] / (2.0 * c2_eff), -250.0 / r_max, 250.0 / r_max);
    const double dy_opt_norm = std::clamp(-p6[2] / (2.0 * c2_eff), -250.0 / r_max, 250.0 / r_max);
    out_x0_px = cx0 + dx_opt_norm * r_max;
    out_y0_px = cy0 + dy_opt_norm * r_max;

    // Evaluate smooth V_vig normalized to 1.0 at optical center (out_x0_px, out_y0_px)
    const double v_at_center = std::max(
        0.2,
        p6[0] + p6[1] * dx_opt_norm + p6[2] * dy_opt_norm +
        p6[3] * (dx_opt_norm * dx_opt_norm + dy_opt_norm * dy_opt_norm)
    );

    std::vector<double> v_vig(GPIX, 1.0);
    for (int gr = 0; gr < GH; ++gr) {
        const double ny = ((gr + 0.5) * BLK - cy0) / r_max;
        const size_t roff = static_cast<size_t>(gr) * GW;
        for (int gc = 0; gc < GW; ++gc) {
            const double nx = ((gc + 0.5) * BLK - cx0) / r_max;
            const double r2 = nx * nx + ny * ny;
            // Clamp linear shift contribution to the bounded optical-axis decenter
            const double lin_x = -2.0 * c2_eff * dx_opt_norm * nx;
            const double lin_y = -2.0 * c2_eff * dy_opt_norm * ny;
            const double val = (p6[0] + lin_x + lin_y + p6[3] * r2 + p6[4] * r2 * r2 + p6[5] * r2 * r2 * r2) / v_at_center;
            v_vig[roff + gc] = std::clamp(val, 0.45, 1.05);
        }
    }
    return v_vig;
}

} // namespace

FlatMode parse_flat_mode(const std::string& s) {
    if (s.empty() || s == "none") return FlatMode::None;
    if (s == "motes-self") return FlatMode::MotesSelf;
    if (s == "motes-iter") return FlatMode::MotesIter;
    if (s == "full-self") return FlatMode::FullSelf;
    if (s == "full-iter") return FlatMode::FullIter;
    throw std::runtime_error("Unknown --flat-mode: " + s + " (expected none|motes-self|motes-iter|full-self|full-iter)");
}

FlatChromaMode parse_flat_chroma_mode(const std::string& s) {
    if (s.empty() || s == "coupled") return FlatChromaMode::Coupled;
    if (s == "per-channel") return FlatChromaMode::PerChannel;
    throw std::runtime_error("Unknown --flat-chroma: " + s + " (expected coupled|per-channel)");
}

std::vector<float> estimate_synthetic_flat(
    const std::vector<std::vector<float>>& rgb_frames,
    const std::vector<FrameTransform>& transforms,
    const FlatFieldConfig& cfg,
    FlatFieldStats& stats_out
) {
    stats_out = FlatFieldStats{};
    if (cfg.mode == FlatMode::None) {
        return std::vector<float>(ACTIVE_PIXELS * 3, 1.0f);
    }

    const bool use_iter = (cfg.mode == FlatMode::MotesIter || cfg.mode == FlatMode::FullIter);
    const bool use_vig = (cfg.mode == FlatMode::FullSelf || cfg.mode == FlatMode::FullIter);

    std::vector<std::array<std::vector<double>, 4>> frame_grids;
    std::vector<int> valid_grid_rows;
    build_frame_grids(rgb_frames, transforms, frame_grids, valid_grid_rows);

    // Pass 1: Fit initial smooth radial vignetting V_vig0 on each channel so dust-mote
    // band-pass extraction has zero concave-downward vignetting curvature or border bias.
    std::array<std::vector<double>, 3> v_motes_ch;
    std::array<std::vector<double>, 3> v_vig_ch;

    if (cfg.chroma == FlatChromaMode::Coupled) {
        // High-SNR luminance channel (ch = 3) for mid-frequency dust motes
        const std::vector<double> sens_l = compute_sensor_space_median(frame_grids, valid_grid_rows, 3);
        double xl0 = ACTIVE_W * 0.5, yl0 = ACTIVE_H * 0.5;
        const std::vector<double> vig_l0 = fit_radial_vignetting(sens_l, xl0, yl0);
        std::vector<double> sens_l_detrend(GPIX, 1.0);
        for (size_t i = 0; i < GPIX; ++i) {
            sens_l_detrend[i] = sens_l[i] / std::max(0.25, vig_l0[i]);
        }

        std::vector<double> vm_l = extract_mote_field_from_grid(sens_l_detrend, cfg);
        if (use_iter) {
            // Detrend per-frame luminance grids by vig_l0 for exact moving-sky cancellation
            std::vector<std::array<std::vector<double>, 4>> fg_det = frame_grids;
            for (auto& fg : fg_det) {
                for (size_t i = 0; i < GPIX; ++i) {
                    fg[3][i] /= std::max(0.25, vig_l0[i]);
                }
            }
            vm_l = refine_motes_iterative_sky(fg_det, valid_grid_rows, transforms, vm_l, cfg, 3);
        }
        for (int ch = 0; ch < 3; ++ch) {
            v_motes_ch[ch] = vm_l;
        }
    } else {
        for (int ch = 0; ch < 3; ++ch) {
            const std::vector<double> sens_c = compute_sensor_space_median(frame_grids, valid_grid_rows, ch);
            double xc0 = ACTIVE_W * 0.5, yc0 = ACTIVE_H * 0.5;
            const std::vector<double> vig_c0 = fit_radial_vignetting(sens_c, xc0, yc0);
            std::vector<double> sens_c_detrend(GPIX, 1.0);
            for (size_t i = 0; i < GPIX; ++i) {
                sens_c_detrend[i] = sens_c[i] / std::max(0.25, vig_c0[i]);
            }

            v_motes_ch[ch] = extract_mote_field_from_grid(sens_c_detrend, cfg);
            if (use_iter) {
                std::vector<std::array<std::vector<double>, 4>> fg_det = frame_grids;
                for (auto& fg : fg_det) {
                    for (size_t i = 0; i < GPIX; ++i) {
                        fg[ch][i] /= std::max(0.25, vig_c0[i]);
                    }
                }
                v_motes_ch[ch] = refine_motes_iterative_sky(
                    fg_det, valid_grid_rows, transforms, v_motes_ch[ch], cfg, ch
                );
            }
        }
    }

    double x0_acc = 0.0, y0_acc = 0.0;
    if (use_vig) {
        if (cfg.chroma == FlatChromaMode::Coupled) {
            // Luminance-anchored radial vignetting + smooth 2-param symmetric chromatic ratio correction
            // (eliminates asymmetric shot-noise downward bias in low-ADU Red channel at corners!)
            const std::vector<double> sens_l = compute_sensor_space_median(frame_grids, valid_grid_rows, 3);
            std::vector<double> corr_l(GPIX, 1.0);
            for (size_t i = 0; i < GPIX; ++i) {
                corr_l[i] = sens_l[i] / std::max(0.2, v_motes_ch[1][i]);
            }
            double x0 = ACTIVE_W * 0.5, y0 = ACTIVE_H * 0.5;
            const std::vector<double> v_vig_l = fit_radial_vignetting(corr_l, x0, y0);
            stats_out.vig_x0_px = x0;
            stats_out.vig_y0_px = y0;

            const double cx0 = (ACTIVE_W - 1) * 0.5;
            const double cy0 = (ACTIVE_H - 1) * 0.5;
            const double r_max = std::hypot(ACTIVE_W * 0.5, ACTIVE_H * 0.5);

            for (int ch = 0; ch < 3; ++ch) {
                const std::vector<double> sens_c = compute_sensor_space_median(frame_grids, valid_grid_rows, ch);
                // Fit smooth chromatic ratio q_c(r) = a0 + a2*r^2 to (sens_c / sens_l) via symmetric Huber IRLS
                double a0 = 1.0, a2 = 0.0;
                for (int pass = 0; pass < 4; ++pass) {
                    double A00 = 0.0, A01 = 0.0, A11 = 0.0, b0 = 0.0, b1 = 0.0;
                    for (int gr = 8; gr < GH - 8; gr += 2) {
                        const double ny = ((gr + 0.5) * BLK - cy0) / r_max;
                        const size_t roff = static_cast<size_t>(gr) * GW;
                        for (int gc = 8; gc < GW - 8; gc += 2) {
                            const double nx = ((gc + 0.5) * BLK - cx0) / r_max;
                            const double r2 = nx * nx + ny * ny;
                            const double sl = sens_l[roff + gc];
                            const double sc = sens_c[roff + gc];
                            if (sl <= 0.25 || sc <= 0.1) continue;
                            // Skip bright galaxy core where color differs from sky
                            if (sl / std::max(0.25, v_vig_l[roff + gc]) > 1.08) continue;
                            const double q = sc / sl;
                            const double resid = q - (a0 + a2 * r2);
                            const double w = (std::abs(resid) <= 0.03) ? 1.0 : (0.03 / std::abs(resid));
                            A00 += w;
                            A01 += w * r2;
                            A11 += w * r2 * r2;
                            b0 += w * q;
                            b1 += w * r2 * q;
                        }
                    }
                    const double det = A00 * A11 - A01 * A01;
                    if (std::abs(det) > 1e-12) {
                        a0 = (A11 * b0 - A01 * b1) / det;
                        a2 = (-A01 * b0 + A00 * b1) / det;
                    }
                }
                const double beta = std::clamp(a2 / std::max(0.2, a0), -0.08, 0.08);
                v_vig_ch[ch].resize(GPIX);
                for (int gr = 0; gr < GH; ++gr) {
                    const double ny = ((gr + 0.5) * BLK - cy0) / r_max;
                    const size_t roff = static_cast<size_t>(gr) * GW;
                    for (int gc = 0; gc < GW; ++gc) {
                        const double nx = ((gc + 0.5) * BLK - cx0) / r_max;
                        const double r2 = nx * nx + ny * ny;
                        v_vig_ch[ch][roff + gc] = std::clamp(v_vig_l[roff + gc] * (1.0 + beta * r2), 0.45, 1.05);
                    }
                }
            }
        } else {
            for (int ch = 0; ch < 3; ++ch) {
                const std::vector<double> sens_c = compute_sensor_space_median(frame_grids, valid_grid_rows, ch);
                std::vector<double> corr_c(GPIX, 1.0);
                for (size_t i = 0; i < GPIX; ++i) {
                    corr_c[i] = sens_c[i] / std::max(0.2, v_motes_ch[ch][i]);
                }
                double x0 = ACTIVE_W * 0.5, y0 = ACTIVE_H * 0.5;
                v_vig_ch[ch] = fit_radial_vignetting(corr_c, x0, y0);
                x0_acc += x0;
                y0_acc += y0;
            }
            stats_out.vig_x0_px = x0_acc / 3.0;
            stats_out.vig_y0_px = y0_acc / 3.0;
        }
    } else {
        for (int ch = 0; ch < 3; ++ch) {
            v_vig_ch[ch].assign(GPIX, 1.0);
        }
    }

    // Combine V(x) = V_vig(x) * V_motes(x) on 4x4 grid, then bilinearly upsample to full resolution
    std::array<std::vector<double>, 3> v_total_grid;
    for (int ch = 0; ch < 3; ++ch) {
        v_total_grid[ch].resize(GPIX);
        for (size_t i = 0; i < GPIX; ++i) {
            v_total_grid[ch][i] = std::clamp(v_vig_ch[ch][i] * v_motes_ch[ch][i], 0.45, 1.05);
        }
    }

    // Mote telemetry from green channel
    double m_min = 1.0;
    double m_sq = 0.0;
    for (size_t i = 0; i < GPIX; ++i) {
        const double vm = v_motes_ch[1][i];
        if (vm < m_min) m_min = vm;
        const double d = vm - 1.0;
        m_sq += d * d;
    }
    stats_out.mote_min_val = m_min;
    stats_out.mote_rms_pct = std::sqrt(m_sq / static_cast<double>(GPIX)) * 100.0;

    std::vector<float> flat_rgb(ACTIVE_PIXELS * 3, 1.0f);
    const double inv_blk = 1.0 / static_cast<double>(BLK);

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < ACTIVE_H; ++r) {
        const double gy = (r + 0.5) * inv_blk - 0.5;
        float* dst_row = flat_rgb.data() + static_cast<size_t>(r) * ACTIVE_W * 3;
        for (int c = 0; c < ACTIVE_W; ++c) {
            const double gx = (c + 0.5) * inv_blk - 0.5;
            dst_row[3 * c + 0] = static_cast<float>(sample_grid_bilinear(v_total_grid[0], gx, gy));
            dst_row[3 * c + 1] = static_cast<float>(sample_grid_bilinear(v_total_grid[1], gx, gy));
            dst_row[3 * c + 2] = static_cast<float>(sample_grid_bilinear(v_total_grid[2], gx, gy));
        }
    }

    const FlatFieldStats s2 = summarize_flat_field(flat_rgb);
    stats_out.v_min = s2.v_min;
    stats_out.v_max = s2.v_max;
    stats_out.v_median = s2.v_median;
    stats_out.vig_center_to_corner_drop_pct = s2.vig_center_to_corner_drop_pct;
    return flat_rgb;
}

void apply_flat_field_inplace(
    std::vector<std::vector<float>>& rgb_frames,
    const std::vector<float>& flat_rgb
) {
    if (flat_rgb.size() != ACTIVE_PIXELS * 3) {
        throw std::runtime_error("Flat field buffer size mismatch in apply_flat_field_inplace");
    }
    const int n_frames = static_cast<int>(rgb_frames.size());
    #pragma omp parallel for schedule(dynamic, 1)
    for (int f = 0; f < n_frames; ++f) {
        float* p = rgb_frames[f].data();
        const float* v = flat_rgb.data();
        for (size_t i = 0; i < ACTIVE_PIXELS * 3; ++i) {
            const float fv = (v[i] > 0.25f) ? v[i] : 0.25f;
            p[i] /= fv;
        }
    }
}

FlatFieldStats summarize_flat_field(const std::vector<float>& flat_rgb) {
    FlatFieldStats s{};
    if (flat_rgb.size() != ACTIVE_PIXELS * 3) return s;

    double vmin = 1e9, vmax = -1e9;
    std::vector<double> samp;
    samp.reserve(60000);
    for (int r = ROI_MARGIN; r < ACTIVE_H - ROI_MARGIN; r += 16) {
        for (int c = ROI_MARGIN; c < ACTIVE_W - ROI_MARGIN; c += 16) {
            const float* p = flat_rgb.data() + (static_cast<size_t>(r) * ACTIVE_W + c) * 3;
            const double vl = 0.25 * p[0] + 0.50 * p[1] + 0.25 * p[2];
            if (vl < vmin) vmin = vl;
            if (vl > vmax) vmax = vl;
            samp.push_back(vl);
        }
    }
    s.v_min = vmin;
    s.v_max = vmax;
    if (!samp.empty()) {
        const size_t mid = samp.size() / 2;
        std::nth_element(samp.begin(), samp.begin() + mid, samp.end());
        s.v_median = samp[mid];
    }

    // Center vs corner average
    auto patch_mean = [&](int r0, int c0) -> double {
        double acc = 0.0;
        int cnt = 0;
        for (int r = r0; r < r0 + 64; ++r) {
            for (int c = c0; c < c0 + 64; ++c) {
                const float* p = flat_rgb.data() + (static_cast<size_t>(r) * ACTIVE_W + c) * 3;
                acc += 0.25 * p[0] + 0.50 * p[1] + 0.25 * p[2];
                ++cnt;
            }
        }
        return acc / std::max(1, cnt);
    };
    const double vc = patch_mean(ACTIVE_H / 2 - 32, ACTIVE_W / 2 - 32);
    const double vcor = 0.25 * (
        patch_mean(ROI_MARGIN, ROI_MARGIN) +
        patch_mean(ROI_MARGIN, ACTIVE_W - ROI_MARGIN - 64) +
        patch_mean(ACTIVE_H - ROI_MARGIN - 64, ROI_MARGIN) +
        patch_mean(ACTIVE_H - ROI_MARGIN - 64, ACTIVE_W - ROI_MARGIN - 64)
    );
    s.vig_center_to_corner_drop_pct = (1.0 - vcor / std::max(1e-6, vc)) * 100.0;
    return s;
}

} // namespace astro

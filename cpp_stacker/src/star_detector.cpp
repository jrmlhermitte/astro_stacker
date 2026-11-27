#include "star_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace astro {

namespace {

double compute_median(std::vector<double>& vals) {
    if (vals.empty()) return 0.0;
    const size_t n = vals.size();
    const size_t mid = n / 2;
    std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
    const double hi = vals[mid];
    if ((n & 1) == 1) return hi;
    std::nth_element(vals.begin(), vals.begin() + (mid - 1), vals.begin() + mid);
    return 0.5 * (vals[mid - 1] + hi);
}

// Internal 64x64 iterative sigma-clipped SEP background mesh STRICTLY for star detection
// (NEVER subtracted from integrated output RGB frames — Guardrail 4!).
void estimate_detection_bg_mesh_64(
    const std::vector<float>& lum,
    std::vector<float>& bg_map_out,
    double& sky_med_out,
    double& sky_rms_out
) {
    constexpr int CELL = 64;
    const int bh = (ACTIVE_H + CELL - 1) / CELL;
    const int bw = (ACTIVE_W + CELL - 1) / CELL;

    std::vector<double> cell_med(static_cast<size_t>(bh) * bw, 0.0);
    std::vector<double> cell_mad(static_cast<size_t>(bh) * bw, 0.0);

    std::vector<double> buf;
    std::vector<double> dev;
    buf.reserve(CELL * CELL);
    dev.reserve(CELL * CELL);

    for (int by = 0; by < bh; ++by) {
        const int r0 = by * CELL;
        const int r1 = std::min(ACTIVE_H, r0 + CELL);
        for (int bx = 0; bx < bw; ++bx) {
            const int c0 = bx * CELL;
            const int c1 = std::min(ACTIVE_W, c0 + CELL);
            buf.clear();
            for (int r = r0; r < r1; ++r) {
                const float* row = lum.data() + static_cast<size_t>(r) * ACTIVE_W;
                for (int c = c0; c < c1; ++c) {
                    buf.push_back(static_cast<double>(row[c]));
                }
            }
            const double m1 = compute_median(buf);
            dev.resize(buf.size());
            for (size_t i = 0; i < buf.size(); ++i) dev[i] = std::abs(buf[i] - m1);
            const double s1 = compute_median(dev) * 1.4826;

            const double lo = m1 - 3.0 * s1;
            const double hi = m1 + 3.0 * s1;
            std::vector<double> clipped;
            clipped.reserve(buf.size());
            for (double v : buf) {
                if (v >= lo && v <= hi) clipped.push_back(v);
            }
            if (clipped.size() >= 16) {
                const double m2 = compute_median(clipped);
                dev.resize(clipped.size());
                for (size_t i = 0; i < clipped.size(); ++i) dev[i] = std::abs(clipped[i] - m2);
                cell_med[static_cast<size_t>(by) * bw + bx] = m2;
                cell_mad[static_cast<size_t>(by) * bw + bx] = compute_median(dev) * 1.4826;
            } else {
                cell_med[static_cast<size_t>(by) * bw + bx] = m1;
                cell_mad[static_cast<size_t>(by) * bw + bx] = s1;
            }
        }
    }

    // Quietest interior sky cells for scalar sky_median and sky_rms
    std::vector<double> int_mads;
    std::vector<double> int_meds;
    for (int by = 1; by < bh - 1; ++by) {
        for (int bx = 1; bx < bw - 1; ++bx) {
            int_mads.push_back(cell_mad[static_cast<size_t>(by) * bw + bx]);
            int_meds.push_back(cell_med[static_cast<size_t>(by) * bw + bx]);
        }
    }
    sky_med_out = compute_median(int_meds);
    sky_rms_out = std::max(1e-6, compute_median(int_mads));

    // 3x3 median filter across cell medians to suppress bright extended stars
    std::vector<double> med_3x3(static_cast<size_t>(bh) * bw, 0.0);
    std::vector<double> nb9;
    nb9.reserve(9);
    for (int by = 0; by < bh; ++by) {
        for (int bx = 0; bx < bw; ++bx) {
            nb9.clear();
            for (int dy = -1; dy <= 1; ++dy) {
                const int yy = std::clamp(by + dy, 0, bh - 1);
                for (int dx = -1; dx <= 1; ++dx) {
                    const int xx = std::clamp(bx + dx, 0, bw - 1);
                    nb9.push_back(cell_med[static_cast<size_t>(yy) * bw + xx]);
                }
            }
            med_3x3[static_cast<size_t>(by) * bw + bx] = compute_median(nb9);
        }
    }

    // Bilinear interpolation to full sensor resolution (strictly for detection residual!)
    bg_map_out.resize(ACTIVE_PIXELS);
    for (int r = 0; r < ACTIVE_H; ++r) {
        const double gy = std::clamp((static_cast<double>(r) - (CELL * 0.5 - 0.5)) / CELL, 0.0, bh - 1.000001);
        const int iy = static_cast<int>(gy);
        const double fy = gy - iy;
        float* dst_row = bg_map_out.data() + static_cast<size_t>(r) * ACTIVE_W;
        for (int c = 0; c < ACTIVE_W; ++c) {
            const double gx = std::clamp((static_cast<double>(c) - (CELL * 0.5 - 0.5)) / CELL, 0.0, bw - 1.000001);
            const int ix = static_cast<int>(gx);
            const double fx = gx - ix;
            const double v00 = med_3x3[static_cast<size_t>(iy) * bw + ix];
            const double v01 = med_3x3[static_cast<size_t>(iy) * bw + (ix + 1)];
            const double v10 = med_3x3[static_cast<size_t>(iy + 1) * bw + ix];
            const double v11 = med_3x3[static_cast<size_t>(iy + 1) * bw + (ix + 1)];
            const double val = (1.0 - fy) * ((1.0 - fx) * v00 + fx * v01) +
                               fy * ((1.0 - fx) * v10 + fx * v11);
            dst_row[c] = static_cast<float>(val);
        }
    }
}

bool solve_7x7_symmetric(double A[7][7], const double b[7], double x[7]) {
    double M[7][8];
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) M[i][j] = A[i][j];
        M[i][7] = b[i];
    }
    for (int col = 0; col < 7; ++col) {
        int piv = col;
        double max_abs = std::abs(M[col][col]);
        for (int r = col + 1; r < 7; ++r) {
            if (std::abs(M[r][col]) > max_abs) {
                max_abs = std::abs(M[r][col]);
                piv = r;
            }
        }
        if (max_abs < 1e-14) return false;
        if (piv != col) {
            for (int j = col; j < 8; ++j) std::swap(M[col][j], M[piv][j]);
        }
        const double inv_diag = 1.0 / M[col][col];
        for (int j = col; j < 8; ++j) M[col][j] *= inv_diag;
        for (int r = 0; r < 7; ++r) {
            if (r == col) continue;
            const double factor = M[r][col];
            if (factor == 0.0) continue;
            for (int j = col; j < 8; ++j) M[r][j] -= factor * M[col][j];
        }
    }
    for (int i = 0; i < 7; ++i) x[i] = M[i][7];
    return true;
}

} // namespace

FrameQuality analyze_frame_stars(
    const std::vector<float>& lum,
    int frame_index,
    const std::string& filename,
    int valid_rows
) {
    FrameQuality fq;
    fq.frame_index = frame_index;
    fq.filename = filename;
    fq.valid_rows = std::clamp(valid_rows, 0, ACTIVE_H);
    const int y_max_roi = std::max(ROI_MARGIN, fq.valid_rows - ROI_MARGIN);

    std::vector<float> bg_map;
    double sky_med = 0.0;
    double sky_rms = 1.0;
    estimate_detection_bg_mesh_64(lum, bg_map, sky_med, sky_rms);
    fq.sky_median = sky_med;
    fq.sky_rms = sky_rms;

    std::vector<float> lum_res(ACTIVE_PIXELS);
    for (size_t i = 0; i < ACTIVE_PIXELS; ++i) {
        lum_res[i] = lum[i] - bg_map[i];
    }

    const float thresh = static_cast<float>(5.0 * sky_rms);
    std::vector<uint8_t> visited(ACTIVE_PIXELS, 0);

    struct CompCandidate {
        float peak_above_bg;
        int py;
        int px;
        int area;
    };
    std::vector<CompCandidate> cands;
    std::vector<int> qy;
    std::vector<int> qx;
    qy.reserve(2048);
    qx.reserve(2048);

    for (int r0 = ROI_MARGIN; r0 < y_max_roi; ++r0) {
        const size_t row_off = static_cast<size_t>(r0) * ACTIVE_W;
        for (int c0 = ROI_MARGIN; c0 < ACTIVE_W - ROI_MARGIN; ++c0) {
            const size_t idx0 = row_off + c0;
            if (visited[idx0] || lum_res[idx0] <= thresh) continue;

            qy.clear();
            qx.clear();
            qy.push_back(r0);
            qx.push_back(c0);
            visited[idx0] = 1;

            int py = r0;
            int px = c0;
            float pval = lum_res[idx0];
            size_t head = 0;

            while (head < qy.size()) {
                const int cy = qy[head];
                const int cx = qx[head];
                ++head;
                const float v = lum_res[static_cast<size_t>(cy) * ACTIVE_W + cx];
                if (v > pval) {
                    pval = v;
                    py = cy;
                    px = cx;
                }
                for (int ny = cy - 1; ny <= cy + 1; ++ny) {
                    if (ny < ROI_MARGIN || ny >= y_max_roi) continue;
                    for (int nx = cx - 1; nx <= cx + 1; ++nx) {
                        if (nx < ROI_MARGIN || nx >= ACTIVE_W - ROI_MARGIN) continue;
                        const size_t nidx = static_cast<size_t>(ny) * ACTIVE_W + nx;
                        if (!visited[nidx] && lum_res[nidx] > thresh) {
                            visited[nidx] = 1;
                            qy.push_back(ny);
                            qx.push_back(nx);
                        }
                    }
                }
            }

            const int area = static_cast<int>(qy.size());
            if (area < 5 || area > 2500) continue;
            if (pval >= 10500.0f) continue;
            if (py < ROI_MARGIN || py >= y_max_roi ||
                px < ROI_MARGIN || px >= ACTIVE_W - ROI_MARGIN) {
                continue;
            }
            cands.push_back({pval, py, px, area});
        }
    }

    std::sort(cands.begin(), cands.end(), [](const CompCandidate& a, const CompCandidate& b) {
        return a.peak_above_bg > b.peak_above_bg;
    });
    if (cands.size() > 600) cands.resize(600);

    const double cx_img = (ACTIVE_W - 1) * 0.5;
    const double cy_img = (ACTIVE_H - 1) * 0.5;
    const double corner_radius = 0.65 * std::hypot(ACTIVE_W * 0.5, ACTIVE_H * 0.5);
    constexpr int R_HALF = 12;
    constexpr int PATCH_N = (2 * R_HALF + 1) * (2 * R_HALF + 1);
    const double log_smin = std::log(0.4);
    const double log_smax = std::log(16.0);

    std::vector<double> cut(PATCH_N);
    std::vector<double> cut_sorted(PATCH_N);

    for (const auto& cand : cands) {
        const int py = cand.py;
        const int px = cand.px;
        const double pval = static_cast<double>(cand.peak_above_bg);

        // Reject MHC 5x5 single-pixel warm impulses
        double sum_3x3 = 0.0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const double v = static_cast<double>(lum_res[static_cast<size_t>(py + dy) * ACTIVE_W + (px + dx)]);
                if (v > 0.0) sum_3x3 += v;
            }
        }
        const double sharpness_3x3 = pval / (sum_3x3 + 1e-12);
        const double ax2_mean = 0.25 * (
            static_cast<double>(lum_res[static_cast<size_t>(py - 2) * ACTIVE_W + px]) +
            static_cast<double>(lum_res[static_cast<size_t>(py + 2) * ACTIVE_W + px]) +
            static_cast<double>(lum_res[static_cast<size_t>(py) * ACTIVE_W + (px - 2)]) +
            static_cast<double>(lum_res[static_cast<size_t>(py) * ACTIVE_W + (px + 2)])
        );
        if (sharpness_3x3 > 0.36 || (ax2_mean / std::max(pval, 1e-12)) < -0.02) {
            continue;
        }

        int k = 0;
        for (int dy = -R_HALF; dy <= R_HALF; ++dy) {
            const float* row = lum_res.data() + static_cast<size_t>(py + dy) * ACTIVE_W;
            for (int dx = -R_HALF; dx <= R_HALF; ++dx) {
                cut[k++] = static_cast<double>(row[px + dx]);
            }
        }
        cut_sorted = cut;
        const size_t p15_idx = static_cast<size_t>(0.15 * (PATCH_N - 1));
        std::nth_element(cut_sorted.begin(), cut_sorted.begin() + p15_idx, cut_sorted.end());
        const double b0 = cut_sorted[p15_idx];
        const double amp0 = std::max(cut[R_HALF * (2 * R_HALF + 1) + R_HALF] - b0, sky_rms);

        double params[7] = {b0, amp0, 0.0, 0.0, std::log(3.5), std::log(3.5), 0.0};
        const double lam = 1e-2;
        bool fit_ok = true;

        for (int iter = 0; iter < 16; ++iter) {
            const double b = params[0];
            const double amp = params[1];
            const double x0 = params[2];
            const double y0 = params[3];
            const double sx = std::exp(std::clamp(params[4], log_smin, log_smax));
            const double sy = std::exp(std::clamp(params[5], log_smin, log_smax));
            const double ct = std::cos(params[6]);
            const double st = std::sin(params[6]);
            const double inv_sx2 = 1.0 / (sx * sx);
            const double inv_sy2 = 1.0 / (sy * sy);

            double jtj[7][7] = {};
            double jte[7] = {};

            int idx = 0;
            for (int dy_i = -R_HALF; dy_i <= R_HALF; ++dy_i) {
                const double dy = static_cast<double>(dy_i) - y0;
                for (int dx_i = -R_HALF; dx_i <= R_HALF; ++dx_i) {
                    const double dx = static_cast<double>(dx_i) - x0;
                    const double u = ct * dx + st * dy;
                    const double v = -st * dx + ct * dy;
                    const double q_arg = std::min(50.0, 0.5 * (u * u * inv_sx2 + v * v * inv_sy2));
                    const double ee = std::exp(-q_arg);
                    const double ae = amp * ee;
                    const double err = cut[idx++] - (b + ae);

                    double J[7];
                    J[0] = 1.0;
                    J[1] = ee;
                    J[2] = -ae * (-ct * u * inv_sx2 + st * v * inv_sy2);
                    J[3] = -ae * (-st * u * inv_sx2 - ct * v * inv_sy2);
                    J[4] = ae * (u * u * inv_sx2);
                    J[5] = ae * (v * v * inv_sy2);
                    J[6] = -ae * (u * v * (inv_sx2 - inv_sy2));

                    for (int r = 0; r < 7; ++r) {
                        jte[r] += J[r] * err;
                        for (int c = r; c < 7; ++c) {
                            jtj[r][c] += J[r] * J[c];
                        }
                    }
                }
            }

            for (int r = 0; r < 7; ++r) {
                for (int c = 0; c < r; ++c) jtj[r][c] = jtj[c][r];
                jtj[r][r] += lam * std::max(jtj[r][r], 1e-12);
            }

            double dp[7];
            if (!solve_7x7_symmetric(jtj, jte, dp)) {
                fit_ok = false;
                break;
            }

            if (std::abs(params[2] + dp[2]) > 6.5 || std::abs(params[3] + dp[3]) > 6.5) {
                fit_ok = false;
                break;
            }
            for (int r = 0; r < 7; ++r) params[r] += dp[r];

            double step_norm = std::max(
                std::abs(dp[0]) / sky_rms,
                std::abs(dp[1]) / sky_rms
            );
            for (int r = 2; r < 7; ++r) step_norm = std::max(step_norm, std::abs(dp[r]));
            if (step_norm < 1e-5) break;
        }

        if (!fit_ok) continue;

        const double b_f = params[0];
        const double amp_f = params[1];
        const double x0_f = params[2];
        const double y0_f = params[3];
        const double sx_f = std::exp(std::clamp(params[4], log_smin, log_smax));
        const double sy_f = std::exp(std::clamp(params[5], log_smin, log_smax));
        const double s_maj = std::max(sx_f, sy_f);
        const double s_min = std::min(sx_f, sy_f);
        const double fwhm_geom = 2.354820045 * std::sqrt(s_maj * s_min);
        const double fwhm_maj = 2.354820045 * s_maj;

        if (fwhm_geom < 2.0 || amp_f < 3.5 * sky_rms || fwhm_geom > 22.0 || fwhm_maj > 28.0) {
            continue;
        }

        const double ct_f = std::cos(params[6]);
        const double st_f = std::sin(params[6]);
        double chi2_sum = 0.0;
        int idx = 0;
        for (int dy_i = -R_HALF; dy_i <= R_HALF; ++dy_i) {
            const double dy = static_cast<double>(dy_i) - y0_f;
            for (int dx_i = -R_HALF; dx_i <= R_HALF; ++dx_i) {
                const double dx = static_cast<double>(dx_i) - x0_f;
                const double u = ct_f * dx + st_f * dy;
                const double v = -st_f * dx + ct_f * dy;
                const double q = std::min(50.0, 0.5 * ((u / sx_f) * (u / sx_f) + (v / sy_f) * (v / sy_f)));
                const double mod_above = amp_f * std::exp(-q);
                const double resid = cut[idx++] - (b_f + mod_above);
                const double p_var = sky_rms * sky_rms +
                                     (0.08 * std::max(0.0, mod_above)) * (0.08 * std::max(0.0, mod_above)) +
                                     (0.03 * std::max(0.0, amp_f)) * (0.03 * std::max(0.0, amp_f));
                chi2_sum += (resid * resid) / p_var;
            }
        }
        if (chi2_sum / static_cast<double>(PATCH_N - 7) > 5.0) continue;

        const double xc = static_cast<double>(px) + x0_f;
        const double yc = static_cast<double>(py) + y0_f;
        const double ecc = std::sqrt(std::max(0.0, 1.0 - (s_min / s_maj) * (s_min / s_maj)));
        const bool is_corner = std::hypot(xc - cx_img, yc - cy_img) > corner_radius;
        const double flux_gauss = 6.283185307179586 * amp_f * s_maj * s_min;

        fq.stars.push_back({
            xc,
            yc,
            flux_gauss,
            amp_f,
            fwhm_geom,
            ecc,
            is_corner
        });
    }

    fq.star_count = static_cast<int>(fq.stars.size());
    if (fq.star_count > 0) {
        std::vector<double> fwhms;
        std::vector<double> corner_fwhms;
        std::vector<double> eccs;
        fwhms.reserve(fq.star_count);
        eccs.reserve(fq.star_count);
        for (const auto& s : fq.stars) {
            fwhms.push_back(s.fwhm_px);
            eccs.push_back(s.eccentricity);
            if (s.is_corner) corner_fwhms.push_back(s.fwhm_px);
        }
        fq.median_fwhm_px = compute_median(fwhms);
        fq.median_eccentricity = compute_median(eccs);
        fq.corner_fwhm_px = corner_fwhms.empty() ? fq.median_fwhm_px : compute_median(corner_fwhms);
    }

    return fq;
}

int select_sharpest_reference_frame(const std::vector<FrameQuality>& qualities) {
    if (qualities.empty()) return -1;

    std::vector<double> star_counts;
    std::vector<double> eccs;
    for (const auto& q : qualities) {
        star_counts.push_back(static_cast<double>(q.star_count));
        eccs.push_back(q.median_eccentricity);
    }
    const int min_healthy_stars = static_cast<int>(std::floor(0.75 * compute_median(star_counts)));
    const double med_ecc = compute_median(eccs);

    int best_idx = -1;
    auto is_better = [&](int a, int b) {
        if (std::abs(qualities[a].median_fwhm_px - qualities[b].median_fwhm_px) > 1e-6) {
            return qualities[a].median_fwhm_px < qualities[b].median_fwhm_px;
        }
        if (std::abs(qualities[a].corner_fwhm_px - qualities[b].corner_fwhm_px) > 1e-6) {
            return qualities[a].corner_fwhm_px < qualities[b].corner_fwhm_px;
        }
        return qualities[a].median_eccentricity < qualities[b].median_eccentricity;
    };

    for (int i = 0; i < static_cast<int>(qualities.size()); ++i) {
        if (qualities[i].valid_rows < ACTIVE_H) continue; // Reference must always be 100% full-frame
        if (qualities[i].star_count >= min_healthy_stars && qualities[i].median_eccentricity <= med_ecc + 1e-9) {
            if (best_idx < 0 || is_better(i, best_idx)) {
                best_idx = i;
            }
        }
    }
    if (best_idx < 0) {
        for (int i = 0; i < static_cast<int>(qualities.size()); ++i) {
            if (best_idx < 0 || is_better(i, best_idx)) {
                best_idx = i;
            }
        }
    }
    return best_idx;
}

} // namespace astro

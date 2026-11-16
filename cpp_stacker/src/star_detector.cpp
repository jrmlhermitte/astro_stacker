#include "star_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

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

} // namespace

FrameQuality analyze_frame_stars(
    const std::vector<float>& lum,
    int frame_index,
    const std::string& filename
) {
    FrameQuality fq;
    fq.frame_index = frame_index;
    fq.filename = filename;

    // Iter 0 Baseline: Global scalar sky median and MAD on strided interior grid
    std::vector<double> samples;
    samples.reserve(((ACTIVE_H - 2 * ROI_MARGIN) / 8) * ((ACTIVE_W - 2 * ROI_MARGIN) / 8));
    for (int r = ROI_MARGIN; r < ACTIVE_H - ROI_MARGIN; r += 8) {
        const float* row = lum.data() + static_cast<size_t>(r) * ACTIVE_W;
        for (int c = ROI_MARGIN; c < ACTIVE_W - ROI_MARGIN; c += 8) {
            samples.push_back(static_cast<double>(row[c]));
        }
    }
    const double sky_med = compute_median(samples);
    std::vector<double> abs_dev(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        abs_dev[i] = std::abs(samples[i] - sky_med);
    }
    const double sky_rms = std::max(1e-6, compute_median(abs_dev) * 1.4826);
    fq.sky_median = sky_med;
    fq.sky_rms = sky_rms;

    const float thresh = static_cast<float>(sky_med + 6.0 * sky_rms);
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

    for (int r0 = ROI_MARGIN; r0 < ACTIVE_H - ROI_MARGIN; ++r0) {
        const size_t row_off = static_cast<size_t>(r0) * ACTIVE_W;
        for (int c0 = ROI_MARGIN; c0 < ACTIVE_W - ROI_MARGIN; ++c0) {
            const size_t idx0 = row_off + c0;
            if (visited[idx0] || lum[idx0] <= thresh) continue;

            qy.clear();
            qx.clear();
            qy.push_back(r0);
            qx.push_back(c0);
            visited[idx0] = 1;

            int py = r0;
            int px = c0;
            float pval = lum[idx0];
            size_t head = 0;

            while (head < qy.size()) {
                const int cy = qy[head];
                const int cx = qx[head];
                ++head;
                const float v = lum[static_cast<size_t>(cy) * ACTIVE_W + cx];
                if (v > pval) {
                    pval = v;
                    py = cy;
                    px = cx;
                }
                for (int ny = cy - 1; ny <= cy + 1; ++ny) {
                    if (ny < ROI_MARGIN || ny >= ACTIVE_H - ROI_MARGIN) continue;
                    for (int nx = cx - 1; nx <= cx + 1; ++nx) {
                        if (nx < ROI_MARGIN || nx >= ACTIVE_W - ROI_MARGIN) continue;
                        const size_t nidx = static_cast<size_t>(ny) * ACTIVE_W + nx;
                        if (!visited[nidx] && lum[nidx] > thresh) {
                            visited[nidx] = 1;
                            qy.push_back(ny);
                            qx.push_back(nx);
                        }
                    }
                }
            }

            const int area = static_cast<int>(qy.size());
            const float peak_above = pval - static_cast<float>(sky_med);
            if (area < 5 || area > 2000) continue;
            if (peak_above >= 10500.0f) continue;
            if (py < ROI_MARGIN || py >= ACTIVE_H - ROI_MARGIN ||
                px < ROI_MARGIN || px >= ACTIVE_W - ROI_MARGIN) {
                continue;
            }
            cands.push_back({peak_above, py, px, area});
        }
    }

    std::sort(cands.begin(), cands.end(), [](const CompCandidate& a, const CompCandidate& b) {
        return a.peak_above_bg > b.peak_above_bg;
    });
    if (cands.size() > 500) cands.resize(500);

    const double cx_img = (ACTIVE_W - 1) * 0.5;
    const double cy_img = (ACTIVE_H - 1) * 0.5;
    const double corner_radius = 0.65 * std::hypot(ACTIVE_W * 0.5, ACTIVE_H * 0.5);

    std::vector<double> patch_vals;
    patch_vals.reserve(21 * 21);

    for (const auto& cand : cands) {
        const int py = cand.py;
        const int px = cand.px;
        const double p_res = static_cast<double>(lum[static_cast<size_t>(py) * ACTIVE_W + px]) - sky_med;

        // Filter single-site MHC 5x5 warm impulses before moment computation
        double sum_3x3 = 0.0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const double v = static_cast<double>(lum[static_cast<size_t>(py + dy) * ACTIVE_W + (px + dx)]) - sky_med;
                if (v > 0.0) sum_3x3 += v;
            }
        }
        const double sharpness_3x3 = p_res / (sum_3x3 + 1e-12);
        const double ax2_mean = 0.25 * (
            (static_cast<double>(lum[static_cast<size_t>(py - 2) * ACTIVE_W + px]) - sky_med) +
            (static_cast<double>(lum[static_cast<size_t>(py + 2) * ACTIVE_W + px]) - sky_med) +
            (static_cast<double>(lum[static_cast<size_t>(py) * ACTIVE_W + (px - 2)]) - sky_med) +
            (static_cast<double>(lum[static_cast<size_t>(py) * ACTIVE_W + (px + 2)]) - sky_med)
        );
        const double ring_r2_ratio = ax2_mean / std::max(p_res, 1e-12);
        if (sharpness_3x3 > 0.36 || ring_r2_ratio < -0.02) {
            continue;
        }

        // Local outer sky annulus (r in [16, 22], outside PSF wings) + scale-invariant COM & moments
        patch_vals.clear();
        for (int dy = -22; dy <= 22; ++dy) {
            for (int dx = -22; dx <= 22; ++dx) {
                const int r2 = dx * dx + dy * dy;
                if (r2 >= 256 && r2 <= 484) {
                    patch_vals.push_back(static_cast<double>(lum[static_cast<size_t>(py + dy) * ACTIVE_W + (px + dx)]));
                }
            }
        }
        const double local_bg = compute_median(patch_vals);
        const double peak_local = static_cast<double>(lum[static_cast<size_t>(py) * ACTIVE_W + px]) - local_bg;
        if (peak_local < 5.0 * sky_rms) continue;
        // Scale-invariant 15%-of-peak gate eliminates faint-star isophotal truncation bias
        const double frac_gate = std::max(2.0 * sky_rms, 0.15 * peak_local);

        double w_sum = 0.0;
        double wx_sum = 0.0;
        double wy_sum = 0.0;
        for (int dy = -14; dy <= 14; ++dy) {
            for (int dx = -14; dx <= 14; ++dx) {
                if (dx * dx + dy * dy > 196) continue;
                const double val = static_cast<double>(lum[static_cast<size_t>(py + dy) * ACTIVE_W + (px + dx)]) - local_bg;
                if (val > frac_gate) {
                    w_sum += val;
                    wx_sum += val * dx;
                    wy_sum += val * dy;
                }
            }
        }
        if (w_sum <= 0.0) continue;
        const double dx0 = wx_sum / w_sum;
        const double dy0 = wy_sum / w_sum;
        if (std::abs(dx0) > 5.0 || std::abs(dy0) > 5.0) continue;

        double m_xx = 0.0;
        double m_yy = 0.0;
        double m_xy = 0.0;
        for (int dy = -14; dy <= 14; ++dy) {
            for (int dx = -14; dx <= 14; ++dx) {
                if (dx * dx + dy * dy > 196) continue;
                const double val = static_cast<double>(lum[static_cast<size_t>(py + dy) * ACTIVE_W + (px + dx)]) - local_bg;
                if (val > frac_gate) {
                    const double ux = dx - dx0;
                    const double uy = dy - dy0;
                    m_xx += val * ux * ux;
                    m_yy += val * uy * uy;
                    m_xy += val * ux * uy;
                }
            }
        }
        // Analytic correction 1 / 0.66521 = 1.50328 for 15%-amplitude Gaussian truncation
        const double trunc_corr = 1.50328 / w_sum;
        m_xx *= trunc_corr;
        m_yy *= trunc_corr;
        m_xy *= trunc_corr;

        const double trace = m_xx + m_yy;
        const double det_disc = std::sqrt(std::max(0.0, 0.25 * (m_xx - m_yy) * (m_xx - m_yy) + m_xy * m_xy));
        const double var_maj = std::max(0.16, 0.5 * trace + det_disc);
        const double var_min = std::max(0.16, 0.5 * trace - det_disc);
        const double fwhm_geom = 2.354820045 * std::pow(var_maj * var_min, 0.25);
        if (fwhm_geom < 2.2 || fwhm_geom > 22.0) continue;

        const double ecc = std::sqrt(std::max(0.0, 1.0 - var_min / var_maj));
        const double xc = px + dx0;
        const double yc = py + dy0;
        const bool is_corner = std::hypot(xc - cx_img, yc - cy_img) > corner_radius;

        fq.stars.push_back({
            xc,
            yc,
            w_sum,
            static_cast<double>(cand.peak_above_bg),
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

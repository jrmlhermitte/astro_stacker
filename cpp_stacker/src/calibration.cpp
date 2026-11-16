#include "calibration.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace astro {

namespace {

float median_small(float* arr, int n) {
    const int mid = n / 2;
    std::nth_element(arr, arr + mid, arr + n);
    const float hi = arr[mid];
    if ((n & 1) == 1) return hi;
    float lo = arr[0];
    for (int i = 1; i < mid; ++i) {
        if (arr[i] > lo) lo = arr[i];
    }
    return 0.5f * (lo + hi);
}

// Smooth per-row left optical-black strip medians with a 33-row running median to cancel
// Canon CMOS horizontal row-readout bias drift without injecting 58-column sample variance.
std::vector<float> smooth_ob_row_pedestal(const RawSensorFrame& frame) {
    std::vector<float> out(ACTIVE_H, static_cast<float>(frame.meta.ob_left_median));
    if (static_cast<int>(frame.ob_row_medians.size()) != ACTIVE_H) {
        return out;
    }
    float win[33];
    for (int r = 0; r < ACTIVE_H; ++r) {
        int cnt = 0;
        for (int dr = -16; dr <= 16; ++dr) {
            const int rr = std::clamp(r + dr, 0, ACTIVE_H - 1);
            win[cnt++] = frame.ob_row_medians[rr];
        }
        out[r] = median_small(win, cnt);
    }
    return out;
}

} // namespace

std::vector<float> build_master_dark(const std::vector<RawSensorFrame>& clean_darks) {
    if (clean_darks.empty()) {
        throw std::runtime_error("No clean dark frames available to build master dark");
    }

    const int n_darks = static_cast<int>(clean_darks.size());
    std::vector<std::vector<float>> ob_rows(n_darks);
    for (int i = 0; i < n_darks; ++i) {
        ob_rows[i] = smooth_ob_row_pedestal(clean_darks[i]);
    }

    std::vector<float> master_dark(ACTIVE_PIXELS, 0.0f);

    // Temporal Winsorized Sigma-Clipped mean (kappa = 3.0) across the clean dark frames
    // to reject single-dark cosmic-ray hits while preserving 100% Gaussian read-noise efficiency.
    #pragma omp parallel
    {
        std::vector<float> vals(n_darks);
        std::vector<float> work(n_darks);
        std::vector<float> devs(n_darks);

        #pragma omp for schedule(static)
        for (int r = 0; r < ACTIVE_H; ++r) {
            const size_t row_off = static_cast<size_t>(r) * ACTIVE_W;
            for (int c = 0; c < ACTIVE_W; ++c) {
                const size_t idx = row_off + c;
                for (int i = 0; i < n_darks; ++i) {
                    vals[i] = static_cast<float>(clean_darks[i].active_cfa[idx]) - ob_rows[i][r];
                }

                work = vals;
                float loc = median_small(work.data(), n_darks);
                for (int i = 0; i < n_darks; ++i) {
                    devs[i] = std::abs(vals[i] - loc);
                }
                float sigma = std::max(1.0f, median_small(devs.data(), n_darks) * 1.4826f);

                work = vals;
                for (int iter = 0; iter < 3; ++iter) {
                    const float lo = loc - 3.0f * sigma;
                    const float hi = loc + 3.0f * sigma;
                    double sum = 0.0;
                    for (int i = 0; i < n_darks; ++i) {
                        work[i] = std::clamp(work[i], lo, hi);
                        sum += static_cast<double>(work[i]);
                    }
                    loc = static_cast<float>(sum / n_darks);
                    double var_sum = 0.0;
                    for (int i = 0; i < n_darks; ++i) {
                        const double d = static_cast<double>(work[i]) - loc;
                        var_sum += d * d;
                    }
                    sigma = std::max(1.0f, static_cast<float>(std::sqrt(var_sum / std::max(1, n_darks - 1)) * 1.030));
                }

                double final_sum = 0.0;
                for (int i = 0; i < n_darks; ++i) {
                    final_sum += static_cast<double>(work[i]);
                }
                master_dark[idx] = static_cast<float>(final_sum / n_darks);
            }
        }
    }

    return master_dark;
}

void calibrate_light_cfa(
    const RawSensorFrame& light,
    const std::vector<float>& master_dark,
    std::vector<float>& cfa_out,
    bool use_omp
) {
    const std::vector<float> ob_row = smooth_ob_row_pedestal(light);

    std::vector<float> light_bias_sub(ACTIVE_PIXELS);
    #pragma omp parallel for schedule(static) if(use_omp)
    for (int r = 0; r < ACTIVE_H; ++r) {
        const size_t row_off = static_cast<size_t>(r) * ACTIVE_W;
        const float bias_r = ob_row[r];
        for (int c = 0; c < ACTIVE_W; ++c) {
            light_bias_sub[row_off + c] = static_cast<float>(light.active_cfa[row_off + c]) - bias_r;
        }
    }

    // Fit per-frame thermal dark scaling factor alpha on known hot dark sites (250..9000 ADU)
    std::vector<double> ratios;
    ratios.reserve(256);
    for (int r = 4; r < ACTIVE_H - 4; ++r) {
        const size_t row_off = static_cast<size_t>(r) * ACTIVE_W;
        for (int c = 4; c < ACTIVE_W - 4; ++c) {
            const size_t idx = row_off + c;
            const float d0 = master_dark[idx];
            if (d0 <= 250.0f || d0 >= 9000.0f) continue;

            float d_nb[4] = {
                master_dark[(r - 2) * ACTIVE_W + c],
                master_dark[(r + 2) * ACTIVE_W + c],
                master_dark[r * ACTIVE_W + (c - 2)],
                master_dark[r * ACTIVE_W + (c + 2)]
            };
            const float d_bg = median_small(d_nb, 4);
            const float d_exc = d0 - d_bg;
            if (d_exc <= 220.0f) continue;

            float l_nb[4] = {
                light_bias_sub[(r - 2) * ACTIVE_W + c],
                light_bias_sub[(r + 2) * ACTIVE_W + c],
                light_bias_sub[r * ACTIVE_W + (c - 2)],
                light_bias_sub[r * ACTIVE_W + (c + 2)]
            };
            const float l_bg = median_small(l_nb, 4);
            const float l_0 = light_bias_sub[idx];
            if (l_0 >= 12500.0f) continue;
            const float l_exc = l_0 - l_bg;
            const double rat = static_cast<double>(l_exc) / static_cast<double>(d_exc);
            if (rat > 0.2 && rat < 6.0) {
                ratios.push_back(rat);
            }
        }
    }

    float alpha_thermal = 1.0f;
    if (ratios.size() >= 12) {
        const size_t mid = ratios.size() / 2;
        std::nth_element(ratios.begin(), ratios.begin() + mid, ratios.end());
        alpha_thermal = static_cast<float>(std::clamp(ratios[mid], 0.65, 4.5));
    }

    cfa_out.resize(ACTIVE_PIXELS);
    #pragma omp parallel for schedule(static) if(use_omp)
    for (size_t idx = 0; idx < ACTIVE_PIXELS; ++idx) {
        const float d = master_dark[idx];
        const float d_scaled = (d > 25.0f) ? (25.0f + alpha_thermal * (d - 25.0f)) : d;
        cfa_out[idx] = light_bias_sub[idx] - d_scaled;
    }
}

} // namespace astro

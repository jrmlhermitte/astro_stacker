#include "integrator.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace astro {

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

    std::vector<float> stack_out(ACTIVE_PIXELS * 3, 0.0f);
    const double inv_count = 1.0 / static_cast<double>(active_count);

    // Iter 0 Baseline: Integer shift-and-add unweighted mean
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < ACTIVE_H; ++r) {
        float* dst_row = stack_out.data() + static_cast<size_t>(r) * ACTIVE_W * 3;
        for (int c = 0; c < ACTIVE_W; ++c) {
            double acc_r = 0.0;
            double acc_g = 0.0;
            double acc_b = 0.0;

            for (int f = 0; f < n_frames; ++f) {
                const auto& tr = transforms[f];
                if (!tr.integrated || tr.weight <= 0.0) continue;

                const int shift_x = static_cast<int>(std::lround(tr.tx));
                const int shift_y = static_cast<int>(std::lround(tr.ty));
                const int sr = std::clamp(r + shift_y, 0, ACTIVE_H - 1);
                const int sc = std::clamp(c + shift_x, 0, ACTIVE_W - 1);
                const float* src_px = rgb_frames[f].data() + (static_cast<size_t>(sr) * ACTIVE_W + sc) * 3;

                acc_r += static_cast<double>(src_px[0]);
                acc_g += static_cast<double>(src_px[1]);
                acc_b += static_cast<double>(src_px[2]);
            }

            dst_row[3 * c + 0] = static_cast<float>(acc_r * inv_count);
            dst_row[3 * c + 1] = static_cast<float>(acc_g * inv_count);
            dst_row[3 * c + 2] = static_cast<float>(acc_b * inv_count);
        }
    }

    return stack_out;
}

} // namespace astro

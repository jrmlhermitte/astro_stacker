#include "registration.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace astro {

namespace {

std::vector<DetectedStar> pick_isolated_bright_stars(
    const std::vector<DetectedStar>& stars,
    size_t max_n = 40,
    double min_sep = 28.0
) {
    std::vector<DetectedStar> sorted = stars;
    std::sort(sorted.begin(), sorted.end(), [](const DetectedStar& a, const DetectedStar& b) {
        return a.flux > b.flux;
    });
    std::vector<DetectedStar> out;
    out.reserve(max_n);
    for (const auto& s : sorted) {
        bool too_close = false;
        for (const auto& o : out) {
            if (std::hypot(s.x - o.x, s.y - o.y) < min_sep) {
                too_close = true;
                break;
            }
        }
        if (!too_close) {
            out.push_back(s);
            if (out.size() >= max_n) break;
        }
    }
    return out;
}

} // namespace

std::vector<FrameTransform> register_all_frames(
    const std::vector<FrameQuality>& qualities,
    int ref_index
) {
    const int n_frames = static_cast<int>(qualities.size());
    std::vector<FrameTransform> transforms(n_frames);
    const auto ref_stars = pick_isolated_bright_stars(qualities[ref_index].stars, 40, 28.0);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < n_frames; ++i) {
        FrameTransform ft;
        ft.filename = qualities[i].filename;
        ft.integrated = true;
        ft.weight = 1.0;
        ft.is_reference = (i == ref_index);

        if (i == ref_index) {
            ft.inlier_matches = static_cast<int>(ref_stars.size());
            transforms[i] = ft;
            continue;
        }

        const auto tgt_stars = pick_isolated_bright_stars(qualities[i].stars, 45, 24.0);
        struct ShiftVote {
            double dx;
            double dy;
        };
        std::vector<ShiftVote> votes;
        votes.reserve(ref_stars.size() * tgt_stars.size());

        for (const auto& rs : ref_stars) {
            for (const auto& ts : tgt_stars) {
                const double dx = ts.x - rs.x;
                const double dy = ts.y - rs.y;
                if (std::abs(dx) <= 64.0 && std::abs(dy) <= 64.0) {
                    votes.push_back({dx, dy});
                }
            }
        }

        int best_count = -1;
        double best_dx = 0.0;
        double best_dy = 0.0;
        for (const auto& v : votes) {
            int cnt = 0;
            double sdx = 0.0;
            double sdy = 0.0;
            for (const auto& u : votes) {
                if (std::hypot(u.dx - v.dx, u.dy - v.dy) <= 2.5) {
                    ++cnt;
                    sdx += u.dx;
                    sdy += u.dy;
                }
            }
            if (cnt > best_count) {
                best_count = cnt;
                best_dx = sdx / std::max(1, cnt);
                best_dy = sdy / std::max(1, cnt);
            }
        }

        // Refine translation on unique nearest star matches within 3.0 px
        double sum_dx = 0.0;
        double sum_dy = 0.0;
        double sum_r2 = 0.0;
        int n_inliers = 0;
        for (const auto& rs : ref_stars) {
            double best_d = std::numeric_limits<double>::infinity();
            int best_j = -1;
            for (int j = 0; j < static_cast<int>(tgt_stars.size()); ++j) {
                const double d = std::hypot(
                    (tgt_stars[j].x - best_dx) - rs.x,
                    (tgt_stars[j].y - best_dy) - rs.y
                );
                if (d < best_d) {
                    best_d = d;
                    best_j = j;
                }
            }
            if (best_j >= 0 && best_d <= 3.0) {
                const double dx = tgt_stars[best_j].x - rs.x;
                const double dy = tgt_stars[best_j].y - rs.y;
                sum_dx += dx;
                sum_dy += dy;
                sum_r2 += best_d * best_d;
                ++n_inliers;
            }
        }

        if (n_inliers >= 4) {
            best_dx = sum_dx / n_inliers;
            best_dy = sum_dy / n_inliers;
        }

        // Iter 0 Baseline: Integer pixel shift-and-add (rounding sub-pixel shift to nearest integer)
        const int int_dx = static_cast<int>(std::lround(best_dx));
        const int int_dy = static_cast<int>(std::lround(best_dy));

        ft.dx = static_cast<double>(int_dx);
        ft.dy = static_cast<double>(int_dy);
        ft.dtheta_deg = 0.0;
        ft.scale = 1.0;
        ft.m00 = 1.0; ft.m01 = 0.0;
        ft.m10 = 0.0; ft.m11 = 1.0;
        ft.tx = ft.dx;
        ft.ty = ft.dy;
        ft.inlier_matches = n_inliers;
        ft.residual_rms_px = (n_inliers > 0) ? std::sqrt(sum_r2 / n_inliers) : 0.0;
        transforms[i] = ft;
    }

    return transforms;
}

} // namespace astro

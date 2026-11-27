#include "registration.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace astro {

namespace {

constexpr double CX_CENTER = (ACTIVE_W - 1) * 0.5;
constexpr double CY_CENTER = (ACTIVE_H - 1) * 0.5;
constexpr double RAD_TO_DEG = 180.0 / 3.14159265358979323846;

std::vector<DetectedStar> pick_isolated_bright_stars(
    const std::vector<DetectedStar>& stars,
    size_t max_n,
    double min_sep
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

struct TriangleAsterism {
    double u;          // L1 / L2 in (0, 1]
    double v;          // L0 / L2 in (0, u]
    double l2;         // longest side in pixels
    int chirality;     // +1 or -1 signed orientation
    std::array<int, 3> idx; // star indices ordered opposite (L0, L1, L2)
};

std::vector<TriangleAsterism> build_triangles(const std::vector<DetectedStar>& stars, size_t max_stars) {
    const int n = static_cast<int>(std::min(stars.size(), max_stars));
    std::vector<TriangleAsterism> tris;
    if (n < 3) return tris;
    tris.reserve(static_cast<size_t>(n) * (n - 1) * (n - 2) / 6);

    struct SideOpp {
        double len;
        int opp_idx;
    };

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double d_ij = std::hypot(stars[i].x - stars[j].x, stars[i].y - stars[j].y);
            for (int k = j + 1; k < n; ++k) {
                const double d_jk = std::hypot(stars[j].x - stars[k].x, stars[j].y - stars[k].y);
                const double d_ki = std::hypot(stars[k].x - stars[i].x, stars[k].y - stars[i].y);

                std::array<SideOpp, 3> s = {{
                    {d_jk, i},
                    {d_ki, j},
                    {d_ij, k}
                }};
                if (s[0].len > s[1].len) std::swap(s[0], s[1]);
                if (s[1].len > s[2].len) std::swap(s[1], s[2]);
                if (s[0].len > s[1].len) std::swap(s[0], s[1]);

                const double l0 = s[0].len;
                const double l1 = s[1].len;
                const double l2 = s[2].len;
                if (l2 < 140.0) continue;
                const double u = l1 / l2;
                const double v = l0 / l2;
                if (v < 0.12) continue;
                // Avoid near-isosceles vertex-swap ambiguity
                if ((l1 - l0) < 0.015 * l2 || (l2 - l1) < 0.015 * l2) continue;

                const auto& p0 = stars[s[0].opp_idx];
                const auto& p1 = stars[s[1].opp_idx];
                const auto& p2 = stars[s[2].opp_idx];
                const double cross = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
                if (std::abs(cross) < 50.0) continue;

                tris.push_back({
                    u,
                    v,
                    l2,
                    (cross > 0.0) ? 1 : -1,
                    {s[0].opp_idx, s[1].opp_idx, s[2].opp_idx}
                });
            }
        }
    }
    return tris;
}

struct Similarity4Dof {
    double a = 1.0;  // s * cos(theta)
    double b = 0.0;  // s * sin(theta)
    double tx = 0.0;
    double ty = 0.0;
    bool valid = false;
};

// Exact weighted linear Procrustes solver for:
// fx = a * rx - b * ry + tx
// fy = b * rx + a * ry + ty
Similarity4Dof fit_weighted_similarity(
    const std::vector<double>& rx,
    const std::vector<double>& ry,
    const std::vector<double>& fx,
    const std::vector<double>& fy,
    const std::vector<double>& w
) {
    Similarity4Dof res;
    const size_t n = rx.size();
    if (n < 2) return res;

    double w_sum = 0.0;
    double m_rx = 0.0, m_ry = 0.0, m_fx = 0.0, m_fy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        w_sum += w[i];
        m_rx += w[i] * rx[i];
        m_ry += w[i] * ry[i];
        m_fx += w[i] * fx[i];
        m_fy += w[i] * fy[i];
    }
    if (w_sum <= 1e-12) return res;
    m_rx /= w_sum; m_ry /= w_sum;
    m_fx /= w_sum; m_fy /= w_sum;

    double denom = 0.0;
    double num_a = 0.0;
    double num_b = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double drx = rx[i] - m_rx;
        const double dry = ry[i] - m_ry;
        const double dfx = fx[i] - m_fx;
        const double dfy = fy[i] - m_fy;
        denom += w[i] * (drx * drx + dry * dry);
        num_a += w[i] * (drx * dfx + dry * dfy);
        num_b += w[i] * (drx * dfy - dry * dfx);
    }
    if (denom <= 1e-6) return res;

    res.a = num_a / denom;
    res.b = num_b / denom;
    const double scale = std::hypot(res.a, res.b);
    if (scale < 0.995 || scale > 1.005) return res;

    res.tx = m_fx - (res.a * m_rx - res.b * m_ry);
    res.ty = m_fy - (res.b * m_rx + res.a * m_ry);
    res.valid = true;
    return res;
}

} // namespace

std::vector<FrameTransform> register_all_frames(
    const std::vector<FrameQuality>& qualities,
    int ref_index
) {
    const int n_frames = static_cast<int>(qualities.size());
    std::vector<FrameTransform> transforms(n_frames);

    // Full-field isolated star pool (up to 70 stars) for high-precision corner-anchored IRLS,
    // and top 32 brightest stars for rapid triangle asterism geometric hashing.
    const auto ref_pool = pick_isolated_bright_stars(qualities[ref_index].stars, 70, 24.0);
    const auto ref_tris = build_triangles(ref_pool, 32);

    // Spatial hash grid over 2D invariant space (u, v) in [0, 1]^2
    constexpr int GRID_N = 64;
    std::vector<std::vector<int>> hash_grid(GRID_N * GRID_N);
    for (int idx = 0; idx < static_cast<int>(ref_tris.size()); ++idx) {
        const int gu = std::clamp(static_cast<int>(ref_tris[idx].u * GRID_N), 0, GRID_N - 1);
        const int gv = std::clamp(static_cast<int>(ref_tris[idx].v * GRID_N), 0, GRID_N - 1);
        hash_grid[gu * GRID_N + gv].push_back(idx);
    }

    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < n_frames; ++i) {
        FrameTransform ft;
        ft.filename = qualities[i].filename;
        ft.integrated = true;
        ft.weight = 1.0;
        ft.valid_rows = qualities[i].valid_rows;
        ft.is_reference = (i == ref_index);

        if (i == ref_index) {
            ft.flux_scale = 1.0;
            ft.inlier_matches = static_cast<int>(ref_pool.size());
            transforms[i] = ft;
            continue;
        }

        const auto tgt_pool = pick_isolated_bright_stars(qualities[i].stars, 80, 22.0);
        const auto tgt_tris = build_triangles(tgt_pool, 34);

        // 1. Generate candidate 4-DOF Similarity hypotheses from matched chiral triangle asterisms
        std::vector<Similarity4Dof> hypotheses;
        hypotheses.reserve(512);

        std::vector<double> tri_rx(3), tri_ry(3), tri_fx(3), tri_fy(3), tri_w(3, 1.0);

        for (const auto& tt : tgt_tris) {
            const int gu0 = std::clamp(static_cast<int>(tt.u * GRID_N), 0, GRID_N - 1);
            const int gv0 = std::clamp(static_cast<int>(tt.v * GRID_N), 0, GRID_N - 1);

            for (int du = -1; du <= 1; ++du) {
                const int gu = gu0 + du;
                if (gu < 0 || gu >= GRID_N) continue;
                for (int dv = -1; dv <= 1; ++dv) {
                    const int gv = gv0 + dv;
                    if (gv < 0 || gv >= GRID_N) continue;

                    for (int r_idx : hash_grid[gu * GRID_N + gv]) {
                        const auto& rt = ref_tris[r_idx];
                        if (rt.chirality != tt.chirality) continue;
                        if (std::abs(tt.u - rt.u) > 0.0035 || std::abs(tt.v - rt.v) > 0.0035) continue;
                        const double scale_est = tt.l2 / rt.l2;
                        if (scale_est < 0.995 || scale_est > 1.005) continue;

                        for (int k = 0; k < 3; ++k) {
                            const auto& rs = ref_pool[rt.idx[k]];
                            const auto& ts = tgt_pool[tt.idx[k]];
                            tri_rx[k] = rs.x - CX_CENTER;
                            tri_ry[k] = rs.y - CY_CENTER;
                            tri_fx[k] = ts.x - CX_CENTER;
                            tri_fy[k] = ts.y - CY_CENTER;
                        }
                        const Similarity4Dof hyp = fit_weighted_similarity(tri_rx, tri_ry, tri_fx, tri_fy, tri_w);
                        if (hyp.valid && std::abs(hyp.tx) <= 120.0 && std::abs(hyp.ty) <= 120.0) {
                            hypotheses.push_back(hyp);
                        }
                    }
                }
            }
            if (hypotheses.size() >= 600) break;
        }

        // Fallback translation seed if sparse field yielded zero triangles
        if (hypotheses.empty()) {
            for (size_t r = 0; r < std::min<size_t>(25, ref_pool.size()); ++r) {
                for (size_t t = 0; t < std::min<size_t>(25, tgt_pool.size()); ++t) {
                    const double dx = tgt_pool[t].x - ref_pool[r].x;
                    const double dy = tgt_pool[t].y - ref_pool[r].y;
                    if (std::abs(dx) <= 80.0 && std::abs(dy) <= 80.0) {
                        hypotheses.push_back({1.0, 0.0, dx, dy, true});
                    }
                }
            }
        }

        // 2. RANSAC consensus scoring across full-field isolated star pool (inlier radius = 2.2 px)
        Similarity4Dof best_model{1.0, 0.0, 0.0, 0.0, true};
        int best_inliers = -1;
        double best_cost = std::numeric_limits<double>::infinity();

        for (const auto& hyp : hypotheses) {
            int inl = 0;
            double cost = 0.0;
            for (const auto& rs : ref_pool) {
                const double rx = rs.x - CX_CENTER;
                const double ry = rs.y - CY_CENTER;
                const double pred_x = CX_CENTER + (hyp.a * rx - hyp.b * ry + hyp.tx);
                const double pred_y = CY_CENTER + (hyp.b * rx + hyp.a * ry + hyp.ty);

                double min_d2 = std::numeric_limits<double>::infinity();
                for (const auto& ts : tgt_pool) {
                    const double dx = ts.x - pred_x;
                    const double dy = ts.y - pred_y;
                    const double d2 = dx * dx + dy * dy;
                    if (d2 < min_d2) min_d2 = d2;
                }
                if (min_d2 <= 2.2 * 2.2) {
                    ++inl;
                    cost += min_d2;
                } else {
                    cost += 2.2 * 2.2;
                }
            }
            if (inl > best_inliers || (inl == best_inliers && cost < best_cost)) {
                best_inliers = inl;
                best_cost = cost;
                best_model = hyp;
            }
        }

        // 3. 5-pass Huber-weighted IRLS refinement over full-field star matches
        int final_inliers = 0;
        double final_r2_sum = 0.0;
        std::vector<double> in_rx, in_ry, in_fx, in_fy, in_w;
        in_rx.reserve(ref_pool.size());
        in_ry.reserve(ref_pool.size());
        in_fx.reserve(ref_pool.size());
        in_fy.reserve(ref_pool.size());
        in_w.reserve(ref_pool.size());

        for (int pass = 0; pass < 5; ++pass) {
            const double match_rad = (pass == 0) ? 2.8 : 2.0;
            const double huber_delta = 0.45;
            in_rx.clear(); in_ry.clear();
            in_fx.clear(); in_fy.clear();
            in_w.clear();
            double r2_acc = 0.0;

            for (const auto& rs : ref_pool) {
                const double rx = rs.x - CX_CENTER;
                const double ry = rs.y - CY_CENTER;
                const double pred_x = CX_CENTER + (best_model.a * rx - best_model.b * ry + best_model.tx);
                const double pred_y = CY_CENTER + (best_model.b * rx + best_model.a * ry + best_model.ty);

                double min_d = std::numeric_limits<double>::infinity();
                int best_j = -1;
                for (int j = 0; j < static_cast<int>(tgt_pool.size()); ++j) {
                    const double d = std::hypot(tgt_pool[j].x - pred_x, tgt_pool[j].y - pred_y);
                    if (d < min_d) {
                        min_d = d;
                        best_j = j;
                    }
                }
                if (best_j >= 0 && min_d <= match_rad) {
                    const double w_huber = (min_d <= huber_delta) ? 1.0 : (huber_delta / min_d);
                    // Weight slightly higher for higher-SNR stars while capping so bright stars don't monopolize
                    const double snr_w = std::clamp(std::sqrt(rs.peak), 1.0, 6.0);
                    in_rx.push_back(rx);
                    in_ry.push_back(ry);
                    in_fx.push_back(tgt_pool[best_j].x - CX_CENTER);
                    in_fy.push_back(tgt_pool[best_j].y - CY_CENTER);
                    in_w.push_back(w_huber * snr_w);
                    r2_acc += min_d * min_d;
                }
            }

            if (in_rx.size() >= 4) {
                const Similarity4Dof refined = fit_weighted_similarity(in_rx, in_ry, in_fx, in_fy, in_w);
                if (refined.valid) {
                    best_model = refined;
                    final_inliers = static_cast<int>(in_rx.size());
                    final_r2_sum = r2_acc;
                }
            }
        }

        // Compute relative photometric flux ratio F_i = median(flux_tgt / flux_ref) on final inliers
        std::vector<double> flux_ratios;
        flux_ratios.reserve(ref_pool.size());
        for (const auto& rs : ref_pool) {
            if (rs.flux <= 0.0) continue;
            const double rx = rs.x - CX_CENTER;
            const double ry = rs.y - CY_CENTER;
            const double pred_x = CX_CENTER + (best_model.a * rx - best_model.b * ry + best_model.tx);
            const double pred_y = CY_CENTER + (best_model.b * rx + best_model.a * ry + best_model.ty);

            double min_d = std::numeric_limits<double>::infinity();
            int best_j = -1;
            for (int j = 0; j < static_cast<int>(tgt_pool.size()); ++j) {
                const double d = std::hypot(tgt_pool[j].x - pred_x, tgt_pool[j].y - pred_y);
                if (d < min_d) {
                    min_d = d;
                    best_j = j;
                }
            }
            if (best_j >= 0 && min_d <= 2.2 && tgt_pool[best_j].flux > 0.0) {
                const double rat = tgt_pool[best_j].flux / rs.flux;
                if (rat > 0.3 && rat < 3.0) {
                    flux_ratios.push_back(rat);
                }
            }
        }
        double flux_scale = 1.0;
        if (flux_ratios.size() >= 6) {
            const size_t mid = flux_ratios.size() / 2;
            std::nth_element(flux_ratios.begin(), flux_ratios.begin() + mid, flux_ratios.end());
            flux_scale = std::clamp(flux_ratios[mid], 0.5, 2.0);
        }

        const double scale = std::hypot(best_model.a, best_model.b);
        const double dtheta_rad = std::atan2(best_model.b, best_model.a);

        ft.dx = best_model.tx;
        ft.dy = best_model.ty;
        ft.dtheta_deg = dtheta_rad * RAD_TO_DEG;
        ft.scale = scale;
        ft.m00 = best_model.a;
        ft.m01 = -best_model.b;
        ft.m10 = best_model.b;
        ft.m11 = best_model.a;
        ft.tx = best_model.tx;
        ft.ty = best_model.ty;
        ft.flux_scale = flux_scale;
        ft.inlier_matches = final_inliers;
        ft.residual_rms_px = (final_inliers > 0) ? std::sqrt(final_r2_sum / final_inliers) : 0.0;
        transforms[i] = ft;
    }

    // Compute optimal PSF/SNR weights: w_i proportional to F_i^2 / (sigma_bg_i^2 * FWHM_eff_i^3 * (1 + 0.5*ecc_i^2))
    // normalized so mean(w_i) = 1.0 and clamped to [0.35, 1.85] to maintain high Kish effective sample size.
    std::vector<double> raw_w(n_frames, 1.0);
    double med_w_buf_sum = 0.0;
    for (int i = 0; i < n_frames; ++i) {
        const double f_i = transforms[i].flux_scale;
        const double sig_i = std::max(1.0, qualities[i].sky_rms);
        const double fwhm_eff_i = std::max(
            2.5,
            0.65 * qualities[i].median_fwhm_px + 0.35 * qualities[i].corner_fwhm_px
        );
        const double ecc_i = std::clamp(qualities[i].median_eccentricity, 0.0, 0.99);
        raw_w[i] = (f_i * f_i) / (sig_i * sig_i * std::pow(fwhm_eff_i, 3.0) * (1.0 + 0.5 * ecc_i * ecc_i));
        med_w_buf_sum += raw_w[i];
    }
    const double mean_raw = (n_frames > 0) ? (med_w_buf_sum / n_frames) : 1.0;
    double norm_sum = 0.0;
    for (int i = 0; i < n_frames; ++i) {
        transforms[i].weight = std::clamp(raw_w[i] / std::max(1e-30, mean_raw), 0.35, 1.85);
        norm_sum += transforms[i].weight;
    }
    if (norm_sum > 0.0) {
        const double scale_w = static_cast<double>(n_frames) / norm_sum;
        for (int i = 0; i < n_frames; ++i) {
            transforms[i].weight *= scale_w;
        }
    }

    return transforms;
}

} // namespace astro

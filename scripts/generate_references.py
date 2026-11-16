#!/usr/bin/env python3
"""Phase 2 Task 1: Reference Baseline Generator & Astrometric Drift Profiler.

Calibrates and linearly demosaics all clean .CR2 frames in unrotated physical
sensor coordinates (2856 x 4290 x 3, float32) using exact Malvar-He-Cutler (2004)
5x5 linear kernels via an inline C++20 + OpenMP + LibRaw (-lraw_r) shared shim.

Outputs:
  - logs/ref_ring.npy             (2856, 4290, 3) float32 winning reference for ring_set
  - logs/ref_andromeda.npy        (2856, 4290, 3) float32 winning reference for andromeda_set
  - logs/reference_selection.json Full sharpness, warm-pixel, and drift telemetry
"""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np


ACTIVE_TOP = 18
ACTIVE_LEFT = 62
ACTIVE_H = 2856
ACTIVE_W = 4290
RAW_H_EXPECTED = 2874
RAW_W_EXPECTED = 4352
OB_COL_START = 2
OB_COL_END = 60  # raw_image[18:2874, 2:60] -> 2856 x 58 optical black strip


CPP_SOURCE = r"""
#include <libraw/libraw.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" {

struct FrameIntegrityMeta {
    int32_t rc;
    int32_t data_error_count;
    int32_t first_data_error_offset;
    uint32_t process_warnings;
    uint16_t raw_width;
    uint16_t raw_height;
    uint16_t top_margin;
    uint16_t left_margin;
    uint16_t width;
    uint16_t height;
    int32_t flip;
    float iso_speed;
    float shutter;
    float camera_temp_c;
    int64_t timestamp;
    uint16_t active_min;
    uint16_t active_max;
    double active_p1;
    double ob_left_median;
    int32_t cfa_is_rggb;
    int32_t is_clean;
    char error_msg[128];
};

struct CandidateObj {
    int32_t area;
    int32_t r_peak;
    int32_t c_peak;
    float peak_l_above_black;
    float peak_l_above_bg;
    float peak_rgb_max_3x3;
    float comp_flux_l;
    float sharpness_3x3;
    float ring_r2_ratio;
    float raw_cfa_impulse_ratio;
    // 2D Elliptical Gaussian fit results on 25x25 cutout
    int32_t fit_ok;
    double x_centroid;     // column in [0, W-1]
    double y_centroid;     // row in [0, H-1]
    double bg_local;
    double amp;
    double fwhm_px;        // geometric mean 2.35482 * sqrt(sigma_maj * sigma_min)
    double fwhm_major_px;
    double fwhm_minor_px;
    double eccentricity;
    double theta_rad;
    double r2_core;
    double chi2_red;
    double flux_gauss;     // 2 * pi * amp * sigma_maj * sigma_min
};

static void data_err_cb(void* user_data, const char* /*file*/, const int offset) {
    if (!user_data) return;
    auto* m = static_cast<FrameIntegrityMeta*>(user_data);
    if (m->data_error_count == 0) {
        m->first_data_error_offset = offset;
    }
    m->data_error_count += 1;
}

static double exact_u16_quantile_from_hist(const uint32_t* hist, size_t n_total, double q) {
    if (n_total == 0) return 0.0;
    const double pos = q * static_cast<double>(n_total - 1);
    const size_t idx_lo = static_cast<size_t>(std::floor(pos));
    const size_t idx_hi = static_cast<size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(idx_lo);

    size_t cum = 0;
    int val_lo = -1;
    int val_hi = -1;
    for (int v = 0; v < 65536; ++v) {
        cum += hist[v];
        if (val_lo < 0 && cum > idx_lo) val_lo = v;
        if (val_hi < 0 && cum > idx_hi) {
            val_hi = v;
            break;
        }
    }
    if (val_lo < 0) val_lo = 65535;
    if (val_hi < 0) val_hi = val_lo;
    return static_cast<double>(val_lo) + frac * static_cast<double>(val_hi - val_lo);
}

static int unpack_and_check_raw(
    const char* path,
    FrameIntegrityMeta* meta,
    std::vector<uint16_t>* raw_out
) {
    std::memset(meta, 0, sizeof(FrameIntegrityMeta));
    meta->first_data_error_offset = -1;

    LibRaw lr;
    lr.set_dataerror_handler(data_err_cb, meta);

    int rc = lr.open_file(path);
    if (rc != LIBRAW_SUCCESS) {
        meta->rc = rc;
        std::strncpy(meta->error_msg, libraw_strerror(rc), sizeof(meta->error_msg) - 1);
        return rc;
    }
    rc = lr.unpack();
    meta->process_warnings = lr.imgdata.process_warnings;
    if (rc != LIBRAW_SUCCESS) {
        meta->rc = rc;
        std::strncpy(meta->error_msg, libraw_strerror(rc), sizeof(meta->error_msg) - 1);
        return rc;
    }
    if (!lr.imgdata.rawdata.raw_image) {
        meta->rc = -2;
        std::strncpy(meta->error_msg, "raw_image is null", sizeof(meta->error_msg) - 1);
        return -2;
    }

    const auto& sz = lr.imgdata.sizes;
    const auto& oth = lr.imgdata.other;
    const auto& cmn = lr.imgdata.makernotes.common;
    const auto& id = lr.imgdata.idata;

    meta->raw_width = sz.raw_width;
    meta->raw_height = sz.raw_height;
    meta->top_margin = sz.top_margin;
    meta->left_margin = sz.left_margin;
    meta->width = sz.width;
    meta->height = sz.height;
    meta->flip = sz.flip;
    meta->iso_speed = oth.iso_speed;
    meta->shutter = oth.shutter;
    meta->camera_temp_c = cmn.CameraTemperature;
    meta->timestamp = static_cast<int64_t>(oth.timestamp);

    if (sz.raw_width != 4352 || sz.raw_height != 2874 ||
        sz.top_margin != 18 || sz.left_margin != 62 ||
        sz.width != 4290 || sz.height != 2856) {
        meta->rc = -4;
        std::strncpy(meta->error_msg, "Unexpected sensor geometry", sizeof(meta->error_msg) - 1);
        return -4;
    }

    // LibRaw::FC(r, c) inspects unrotated raw filter pattern (ignoring sizes.flip)
    const char c00 = id.cdesc[lr.FC(0, 0)];
    const char c01 = id.cdesc[lr.FC(0, 1)];
    const char c10 = id.cdesc[lr.FC(1, 0)];
    const char c11 = id.cdesc[lr.FC(1, 1)];
    meta->cfa_is_rggb = (c00 == 'R' && c01 == 'G' && c10 == 'G' && c11 == 'B') ? 1 : 0;

    const uint16_t* raw = lr.imgdata.rawdata.raw_image;
    uint32_t hist_act[65536];
    uint32_t hist_ob[65536];
    std::memset(hist_act, 0, sizeof(hist_act));
    std::memset(hist_ob, 0, sizeof(hist_ob));

    uint16_t vmin = 65535;
    uint16_t vmax = 0;
    for (int r = 18; r < 18 + 2856; ++r) {
        const uint16_t* row = raw + static_cast<size_t>(r) * 4352;
        for (int c = 2; c < 60; ++c) {
            hist_ob[row[c]] += 1;
        }
        for (int c = 62; c < 62 + 4290; ++c) {
            const uint16_t v = row[c];
            hist_act[v] += 1;
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
    }

    meta->active_min = vmin;
    meta->active_max = vmax;
    meta->active_p1 = exact_u16_quantile_from_hist(hist_act, 2856ULL * 4290ULL, 0.01);
    meta->ob_left_median = exact_u16_quantile_from_hist(hist_ob, 2856ULL * 58ULL, 0.50);

    const bool corrupted = (meta->data_error_count > 0) ||
                           (meta->active_max > 16383) ||
                           (meta->active_p1 < 1850.0) ||
                           (meta->cfa_is_rggb == 0);
    meta->is_clean = corrupted ? 0 : 1;
    meta->rc = 0;

    if (raw_out) {
        raw_out->assign(raw, raw + 4352ULL * 2874ULL);
    }
    return 0;
}

static inline int reflect101(int idx, int n) {
    if (idx < 0) return -idx;
    if (idx >= n) return 2 * (n - 1) - idx;
    return idx;
}

static void mhc_demosaic_rggb_f32_impl(const float* cfa, int H, int W, float* rgb_out, bool use_omp) {
    auto at_border = [&](int r, int c) -> float {
        const int rr = reflect101(r, H);
        const int cc = reflect101(c, W);
        return cfa[static_cast<size_t>(rr) * W + cc];
    };

    #pragma omp parallel for schedule(static) if(use_omp)
    for (int r = 0; r < H; ++r) {
        const bool interior_r = (r >= 2 && r < H - 2);
        const int r_even = ((r & 1) == 0);
        float* out_row = rgb_out + static_cast<size_t>(r) * W * 3;

        for (int c = 0; c < W; ++c) {
            float c0, n1, s1, w1, e1, n2, s2, w2, e2, nw, ne, sw, se;
            if (interior_r && c >= 2 && c < W - 2) {
                const float* p = cfa + static_cast<size_t>(r) * W + c;
                c0 = p[0];
                n1 = p[-W];       s1 = p[W];
                w1 = p[-1];       e1 = p[1];
                n2 = p[-2 * W];   s2 = p[2 * W];
                w2 = p[-2];       e2 = p[2];
                nw = p[-W - 1];   ne = p[-W + 1];
                sw = p[W - 1];    se = p[W + 1];
            } else {
                c0 = at_border(r, c);
                n1 = at_border(r - 1, c); s1 = at_border(r + 1, c);
                w1 = at_border(r, c - 1); e1 = at_border(r, c + 1);
                n2 = at_border(r - 2, c); s2 = at_border(r + 2, c);
                w2 = at_border(r, c - 2); e2 = at_border(r, c + 2);
                nw = at_border(r - 1, c - 1); ne = at_border(r - 1, c + 1);
                sw = at_border(r + 1, c - 1); se = at_border(r + 1, c + 1);
            }

            const float h1 = w1 + e1;
            const float v1 = n1 + s1;
            const float h2 = w2 + e2;
            const float v2 = n2 + s2;
            const float ax1 = h1 + v1;
            const float ax2 = h2 + v2;
            const float dg1 = nw + ne + sw + se;

            const int c_even = ((c & 1) == 0);
            float R, G, B;
            if (r_even) {
                if (c_even) {
                    // R site (even r, even c)
                    R = c0;
                    G = 0.125f * (4.0f * c0 + 2.0f * ax1 - ax2);
                    B = 0.125f * (6.0f * c0 + 2.0f * dg1 - 1.5f * ax2);
                } else {
                    // G1 site (even r, odd c, R-row B-col)
                    G = c0;
                    R = 0.125f * (5.0f * c0 + 4.0f * h1 - dg1 - h2 + 0.5f * v2);
                    B = 0.125f * (5.0f * c0 + 4.0f * v1 - dg1 + 0.5f * h2 - v2);
                }
            } else {
                if (c_even) {
                    // G2 site (odd r, even c, B-row R-col): transpose of G1
                    G = c0;
                    R = 0.125f * (5.0f * c0 + 4.0f * v1 - dg1 + 0.5f * h2 - v2);
                    B = 0.125f * (5.0f * c0 + 4.0f * h1 - dg1 - h2 + 0.5f * v2);
                } else {
                    // B site (odd r, odd c): swap R and B roles from R-site
                    B = c0;
                    G = 0.125f * (4.0f * c0 + 2.0f * ax1 - ax2);
                    R = 0.125f * (6.0f * c0 + 2.0f * dg1 - 1.5f * ax2);
                }
            }

            out_row[3 * c + 0] = R;
            out_row[3 * c + 1] = G;
            out_row[3 * c + 2] = B;
        }
    }
}

void mhc_demosaic_rggb_f32(const float* cfa, int H, int W, float* rgb_out) {
    mhc_demosaic_rggb_f32_impl(cfa, H, W, rgb_out, true);
}

int build_master_dark(
    int n_files,
    const char** paths,
    FrameIntegrityMeta* metas_out,
    float* master_dark_out // 2856 * 4290
) {
    const size_t N_ACT = 2856ULL * 4290ULL;
    std::vector<double> accum(N_ACT, 0.0);
    int clean_count = 0;

    #pragma omp parallel num_threads(8)
    {
        std::vector<double> local_accum(N_ACT, 0.0);
        int local_clean = 0;

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < n_files; ++i) {
            std::vector<uint16_t> raw;
            unpack_and_check_raw(paths[i], &metas_out[i], &raw);
            if (metas_out[i].is_clean && !raw.empty()) {
                const float ob = static_cast<float>(metas_out[i].ob_left_median);
                for (int r = 0; r < 2856; ++r) {
                    const uint16_t* src = raw.data() + static_cast<size_t>(r + 18) * 4352 + 62;
                    double* dst = local_accum.data() + static_cast<size_t>(r) * 4290;
                    for (int c = 0; c < 4290; ++c) {
                        dst[c] += static_cast<double>(static_cast<float>(src[c]) - ob);
                    }
                }
                local_clean += 1;
            }
        }

        #pragma omp critical
        {
            if (local_clean > 0) {
                for (size_t k = 0; k < N_ACT; ++k) {
                    accum[k] += local_accum[k];
                }
                clean_count += local_clean;
            }
        }
    }

    if (clean_count > 0) {
        const double inv_n = 1.0 / static_cast<double>(clean_count);
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < N_ACT; ++k) {
            master_dark_out[k] = static_cast<float>(accum[k] * inv_n);
        }
    }
    return clean_count;
}

static float median_in_place(std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    const size_t n = v.size();
    const size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    const float hi = v[mid];
    if ((n & 1) == 1) return hi;
    const float lo = *std::max_element(v.begin(), v.begin() + mid);
    return 0.5f * (lo + hi);
}

static bool solve_7x7(double A[7][7], double b[7], double x[7]) {
    double M[7][8];
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) M[i][j] = A[i][j];
        M[i][7] = b[i];
    }
    for (int col = 0; col < 7; ++col) {
        int piv = col;
        double max_abs = std::fabs(M[col][col]);
        for (int r = col + 1; r < 7; ++r) {
            const double v = std::fabs(M[r][col]);
            if (v > max_abs) { max_abs = v; piv = r; }
        }
        if (max_abs < 1e-14) return false;
        if (piv != col) {
            for (int j = col; j < 8; ++j) std::swap(M[col][j], M[piv][j]);
        }
        const double inv_diag = 1.0 / M[col][col];
        for (int r = col + 1; r < 7; ++r) {
            const double factor = M[r][col] * inv_diag;
            for (int j = col; j < 8; ++j) {
                M[r][j] -= factor * M[col][j];
            }
        }
    }
    for (int i = 6; i >= 0; --i) {
        double s = M[i][7];
        for (int j = i + 1; j < 7; ++j) s -= M[i][j] * x[j];
        x[i] = s / M[i][i];
    }
    return true;
}

// Fit 7-parameter 2D elliptical Gaussian on a 25x25 cutout centered at (r_p, c_p)
// so stars with FWHM ~ 7-11 px reach asymptotic background cleanly without wing clipping.
static void fit_2d_elliptical_gaussian_25x25(
    const float* L,
    int H,
    int W,
    int r_p,
    int c_p,
    double global_bg_rms,
    CandidateObj* out
) {
    constexpr int R_WIN = 12;
    constexpr int SIDE = 2 * R_WIN + 1; // 25
    constexpr int N_PIX = SIDE * SIDE;  // 625

    double z[N_PIX];
    std::vector<float> border_vals;
    border_vals.reserve(184);

    for (int dy = -R_WIN; dy <= R_WIN; ++dy) {
        for (int dx = -R_WIN; dx <= R_WIN; ++dx) {
            const float val = L[static_cast<size_t>(r_p + dy) * W + (c_p + dx)];
            z[(dy + R_WIN) * SIDE + (dx + R_WIN)] = static_cast<double>(val);
            if (std::abs(dx) >= R_WIN - 1 || std::abs(dy) >= R_WIN - 1) {
                border_vals.push_back(val);
            }
        }
    }

    double b0 = static_cast<double>(median_in_place(border_vals));
    double amp = std::max(5.0, z[R_WIN * SIDE + R_WIN] - b0);

    // Gaussian-windowed moments within radius 7 for initialization
    double sum_w = 0.0, mx = 0.0, my = 0.0;
    for (int dy = -7; dy <= 7; ++dy) {
        for (int dx = -7; dx <= 7; ++dx) {
            if (dx * dx + dy * dy > 49) continue;
            const double val = std::max(0.0, z[(dy + R_WIN) * SIDE + (dx + R_WIN)] - b0);
            const double gw = std::exp(-0.5 * (dx * dx + dy * dy) / (3.5 * 3.5));
            const double w = val * gw;
            sum_w += w;
            mx += w * dx;
            my += w * dy;
        }
    }
    double x0 = (sum_w > 1e-6) ? std::clamp(mx / sum_w, -3.5, 3.5) : 0.0;
    double y0 = (sum_w > 1e-6) ? std::clamp(my / sum_w, -3.5, 3.5) : 0.0;

    double cxx = 2.0, cyy = 2.0, cxy = 0.0;
    if (sum_w > 1e-6) {
        double sxx = 0.0, syy = 0.0, sxy = 0.0;
        for (int dy = -7; dy <= 7; ++dy) {
            for (int dx = -7; dx <= 7; ++dx) {
                if (dx * dx + dy * dy > 49) continue;
                const double val = std::max(0.0, z[(dy + R_WIN) * SIDE + (dx + R_WIN)] - b0);
                const double gw = std::exp(-0.5 * (dx * dx + dy * dy) / (4.5 * 4.5));
                const double w = val * gw;
                const double ux = dx - x0;
                const double uy = dy - y0;
                sxx += w * ux * ux;
                syy += w * uy * uy;
                sxy += w * ux * uy;
            }
        }
        cxx = std::clamp(sxx / sum_w, 0.16, 36.0);
        cyy = std::clamp(syy / sum_w, 0.16, 36.0);
        cxy = sxy / sum_w;
    }

    const double tr = cxx + cyy;
    const double det_part = std::hypot(cxx - cyy, 2.0 * cxy);
    const double lam1 = std::clamp(0.5 * (tr + det_part), 0.12, 49.0);
    const double lam2 = std::clamp(0.5 * (tr - det_part), 0.12, 49.0);
    double theta = 0.5 * std::atan2(2.0 * cxy, cxx - cyy);

    double p[7] = {
        b0,
        amp,
        x0,
        y0,
        0.5 * std::log(lam1),
        0.5 * std::log(lam2),
        theta
    };

    auto eval_cost = [&](const double q[7]) -> double {
        const double qb0 = q[0];
        const double qA = q[1];
        const double qx0 = q[2];
        const double qy0 = q[3];
        const double s1 = std::exp(q[4]);
        const double s2 = std::exp(q[5]);
        const double ct = std::cos(q[6]);
        const double st = std::sin(q[6]);
        const double inv_s1_2 = 1.0 / (s1 * s1);
        const double inv_s2_2 = 1.0 / (s2 * s2);

        double sse = 0.0;
        for (int dy = -R_WIN; dy <= R_WIN; ++dy) {
            const double y = dy - qy0;
            for (int dx = -R_WIN; dx <= R_WIN; ++dx) {
                const double x = dx - qx0;
                const double u = x * ct + y * st;
                const double v = -x * st + y * ct;
                const double ex = -0.5 * (u * u * inv_s1_2 + v * v * inv_s2_2);
                const double g = (ex > -60.0) ? std::exp(ex) : 0.0;
                const double r = z[(dy + R_WIN) * SIDE + (dx + R_WIN)] - (qb0 + qA * g);
                sse += r * r;
            }
        }
        return sse;
    };

    double cur_sse = eval_cost(p);
    double lambda = 1e-3;

    for (int iter = 0; iter < 40; ++iter) {
        double JtJ[7][7] = {{0.0}};
        double Jtr[7] = {0.0};

        const double qb0 = p[0];
        const double qA = p[1];
        const double qx0 = p[2];
        const double qy0 = p[3];
        const double s1 = std::exp(p[4]);
        const double s2 = std::exp(p[5]);
        const double ct = std::cos(p[6]);
        const double st = std::sin(p[6]);
        const double inv_s1_2 = 1.0 / (s1 * s1);
        const double inv_s2_2 = 1.0 / (s2 * s2);

        for (int dy = -R_WIN; dy <= R_WIN; ++dy) {
            const double y = dy - qy0;
            for (int dx = -R_WIN; dx <= R_WIN; ++dx) {
                const double x = dx - qx0;
                const double u = x * ct + y * st;
                const double v = -x * st + y * ct;
                const double u2_s1 = u * u * inv_s1_2;
                const double v2_s2 = v * v * inv_s2_2;
                const double ex = -0.5 * (u2_s1 + v2_s2);
                const double g = (ex > -60.0) ? std::exp(ex) : 0.0;
                const double ag = qA * g;
                const double res = z[(dy + R_WIN) * SIDE + (dx + R_WIN)] - (qb0 + ag);

                double J[7];
                J[0] = 1.0;
                J[1] = g;
                J[2] = ag * (u * ct * inv_s1_2 - v * st * inv_s2_2);
                J[3] = ag * (u * st * inv_s1_2 + v * ct * inv_s2_2);
                J[4] = ag * u2_s1;
                J[5] = ag * v2_s2;
                J[6] = ag * (-u * v * (inv_s1_2 - inv_s2_2));

                for (int a = 0; a < 7; ++a) {
                    Jtr[a] += J[a] * res;
                    for (int b = a; b < 7; ++b) {
                        JtJ[a][b] += J[a] * J[b];
                    }
                }
            }
        }
        for (int a = 0; a < 7; ++a) {
            for (int b = 0; b < a; ++b) JtJ[a][b] = JtJ[b][a];
        }

        bool accepted = false;
        for (int sub = 0; sub < 6; ++sub) {
            double M[7][7];
            for (int a = 0; a < 7; ++a) {
                for (int b = 0; b < 7; ++b) M[a][b] = JtJ[a][b];
                M[a][a] += lambda * (JtJ[a][a] + 1e-6);
            }
            double dp[7] = {0.0};
            if (!solve_7x7(M, Jtr, dp)) {
                lambda *= 10.0;
                continue;
            }

            double cand[7];
            for (int a = 0; a < 7; ++a) cand[a] = p[a] + dp[a];
            cand[1] = std::max(1e-3, cand[1]);
            cand[2] = std::clamp(cand[2], -5.0, 5.0);
            cand[3] = std::clamp(cand[3], -5.0, 5.0);
            cand[4] = std::clamp(cand[4], -2.3, 3.2);
            cand[5] = std::clamp(cand[5], -2.3, 3.2);

            const double cand_sse = eval_cost(cand);
            if (cand_sse < cur_sse) {
                const double rel_imp = (cur_sse - cand_sse) / (cur_sse + 1e-12);
                for (int a = 0; a < 7; ++a) p[a] = cand[a];
                cur_sse = cand_sse;
                lambda = std::max(1e-7, lambda * 0.3);
                accepted = true;
                if (rel_imp < 1e-6) iter = 40;
                break;
            } else {
                lambda *= 5.0;
            }
        }
        if (!accepted && lambda > 1e8) break;
    }

    double s1 = std::exp(p[4]);
    double s2 = std::exp(p[5]);
    double th = p[6];
    if (s2 > s1) {
        std::swap(s1, s2);
        th += 1.5707963267948966;
    }
    th = std::remainder(th, 3.14159265358979323846);

    // Evaluate R^2 and Poisson+sky reduced chi-squared on central 15x15 core (225 px, dof = 218)
    constexpr int R_CORE = 7;
    double core_sum = 0.0;
    int core_n = 0;
    double core_vals[225];
    double core_pred[225];
    const double ct = std::cos(p[6]);
    const double st = std::sin(p[6]);
    const double inv_s1_2 = 1.0 / std::exp(2.0 * p[4]);
    const double inv_s2_2 = 1.0 / std::exp(2.0 * p[5]);
    const double bg_var = std::max(1.0, global_bg_rms * global_bg_rms);

    double chi2_sum = 0.0;
    for (int dy = -R_CORE; dy <= R_CORE; ++dy) {
        const double y = dy - p[3];
        for (int dx = -R_CORE; dx <= R_CORE; ++dx) {
            const double x = dx - p[2];
            const double u = x * ct + y * st;
            const double v = -x * st + y * ct;
            const double ex = -0.5 * (u * u * inv_s1_2 + v * v * inv_s2_2);
            const double g = (ex > -60.0) ? std::exp(ex) : 0.0;
            const double zv = z[(dy + R_WIN) * SIDE + (dx + R_WIN)];
            const double sig_above_bg = p[1] * g;
            const double pv = p[0] + sig_above_bg;
            core_vals[core_n] = zv;
            core_pred[core_n] = pv;
            core_sum += zv;
            core_n += 1;

            const double err = zv - pv;
            // Canon EOS 1100D @ ISO 1600: ~3.5 ADU/e- shot noise scale on demosaiced luminance
            const double pix_var = bg_var + 2.0 * std::max(0.0, sig_above_bg);
            chi2_sum += (err * err) / pix_var;
        }
    }
    const double core_mean = core_sum / core_n;
    double ss_res = 0.0, ss_tot = 0.0;
    for (int k = 0; k < core_n; ++k) {
        const double er = core_vals[k] - core_pred[k];
        const double et = core_vals[k] - core_mean;
        ss_res += er * er;
        ss_tot += et * et;
    }
    const double r2 = (ss_tot > 1e-9) ? (1.0 - ss_res / ss_tot) : 0.0;
    const double chi2_red = chi2_sum / static_cast<double>(core_n - 7);

    constexpr double SIGMA_TO_FWHM = 2.3548200450309493;
    out->fit_ok = (std::isfinite(s1) && std::isfinite(s2) && p[1] > 0.0 &&
                   std::fabs(p[2]) <= 4.0 && std::fabs(p[3]) <= 4.0) ? 1 : 0;
    out->bg_local = p[0];
    out->amp = p[1];
    out->x_centroid = static_cast<double>(c_p) + p[2];
    out->y_centroid = static_cast<double>(r_p) + p[3];
    out->fwhm_major_px = SIGMA_TO_FWHM * s1;
    out->fwhm_minor_px = SIGMA_TO_FWHM * s2;
    out->fwhm_px = SIGMA_TO_FWHM * std::sqrt(s1 * s2);
    out->eccentricity = std::sqrt(std::max(0.0, 1.0 - (s2 * s2) / (s1 * s1)));
    out->theta_rad = th;
    out->r2_core = r2;
    out->chi2_red = chi2_red;
    out->flux_gauss = 6.283185307179586 * p[1] * s1 * s2;
}

static int analyze_single_light_internal(
    const char* path,
    const float* master_dark,
    FrameIntegrityMeta* meta_out,
    double* bg_rms_adu_out,
    double* bg_median_adu_out,
    CandidateObj* cands_out,
    int max_cands,
    int* total_5sigma_minarea5_out,
    int* total_saturated_rejected_out,
    float* rgb_out,
    bool inner_omp
) {
    std::vector<uint16_t> raw;
    unpack_and_check_raw(path, meta_out, &raw);
    if (!meta_out->is_clean || raw.empty()) {
        *bg_rms_adu_out = 0.0;
        *bg_median_adu_out = 0.0;
        *total_5sigma_minarea5_out = 0;
        *total_saturated_rejected_out = 0;
        return 0;
    }

    constexpr int H = 2856;
    constexpr int W = 4290;
    const size_t N_ACT = static_cast<size_t>(H) * W;

    // 1. Calibrate in float32 WITHOUT zero-clipping!
    std::vector<float> cfa_cal(N_ACT);
    const float ob = static_cast<float>(meta_out->ob_left_median);
    #pragma omp parallel for schedule(static) if(inner_omp)
    for (int r = 0; r < H; ++r) {
        const uint16_t* src = raw.data() + static_cast<size_t>(r + 18) * 4352 + 62;
        const float* dk = master_dark + static_cast<size_t>(r) * W;
        float* dst = cfa_cal.data() + static_cast<size_t>(r) * W;
        for (int c = 0; c < W; ++c) {
            dst[c] = (static_cast<float>(src[c]) - ob) - dk[c];
        }
    }
    raw.clear();
    raw.shrink_to_fit();

    // 2. Exact Malvar-He-Cutler 5x5 linear demosaicing
    std::vector<float> rgb_local;
    float* rgb_ptr = rgb_out;
    if (!rgb_ptr) {
        rgb_local.resize(N_ACT * 3ULL);
        rgb_ptr = rgb_local.data();
    }
    mhc_demosaic_rggb_f32_impl(cfa_cal.data(), H, W, rgb_ptr, inner_omp);

    // 3. Bayer-weighted luminance L = 0.25*R + 0.50*G + 0.25*B
    std::vector<float> L(N_ACT);
    #pragma omp parallel for schedule(static) if(inner_omp)
    for (size_t k = 0; k < N_ACT; ++k) {
        L[k] = 0.25f * rgb_ptr[3 * k + 0] + 0.50f * rgb_ptr[3 * k + 1] + 0.25f * rgb_ptr[3 * k + 2];
    }

    // 4. 64x64 clipped median + MAD background mesh
    constexpr int CELL = 64;
    const int ny = (H + CELL - 1) / CELL; // 45
    const int nx = (W + CELL - 1) / CELL; // 68
    std::vector<float> cell_bg(ny * nx, 0.0f);
    std::vector<float> cell_rms(ny * nx, 0.0f);

    #pragma omp parallel for collapse(2) schedule(static) if(inner_omp)
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            const int r0 = iy * CELL;
            const int r1 = std::min(H, r0 + CELL);
            const int c0 = ix * CELL;
            const int c1 = std::min(W, c0 + CELL);

            std::vector<float> work;
            work.reserve((r1 - r0) * (c1 - c0));
            for (int r = r0; r < r1; ++r) {
                const float* row = L.data() + static_cast<size_t>(r) * W;
                for (int c = c0; c < c1; ++c) work.push_back(row[c]);
            }

            std::vector<float> dev;
            dev.reserve(work.size());
            float med = 0.0f;
            float rms = 0.0f;

            for (int clip_it = 0; clip_it < 3; ++clip_it) {
                std::vector<float> tmp = work;
                med = median_in_place(tmp);
                dev.resize(work.size());
                for (size_t k = 0; k < work.size(); ++k) dev[k] = std::fabs(work[k] - med);
                rms = 1.4826f * median_in_place(dev);
                if (rms < 1e-5f || clip_it == 2) break;

                const float hi_cut = 3.0f * rms;
                std::vector<float> next_work;
                next_work.reserve(work.size());
                for (float v : work) {
                    if (std::fabs(v - med) <= hi_cut) next_work.push_back(v);
                }
                if (next_work.size() < 64) break;
                work.swap(next_work);
            }
            cell_bg[iy * nx + ix] = med;
            cell_rms[iy * nx + ix] = rms;
        }
    }

    std::vector<float> cell_bg_sm(ny * nx);
    std::vector<float> cell_rms_sm(ny * nx);
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            std::vector<float> nb_bg, nb_rms;
            nb_bg.reserve(9);
            nb_rms.reserve(9);
            for (int dy = -1; dy <= 1; ++dy) {
                const int yy = std::clamp(iy + dy, 0, ny - 1);
                for (int dx = -1; dx <= 1; ++dx) {
                    const int xx = std::clamp(ix + dx, 0, nx - 1);
                    nb_bg.push_back(cell_bg[yy * nx + xx]);
                    nb_rms.push_back(cell_rms[yy * nx + xx]);
                }
            }
            cell_bg_sm[iy * nx + ix] = median_in_place(nb_bg);
            cell_rms_sm[iy * nx + ix] = median_in_place(nb_rms);
        }
    }

    std::vector<float> all_rms = cell_rms;
    const float global_bg_rms = median_in_place(all_rms);
    std::vector<float> all_bg = cell_bg;
    const float global_bg_med = median_in_place(all_bg);
    *bg_rms_adu_out = static_cast<double>(global_bg_rms);
    *bg_median_adu_out = static_cast<double>(global_bg_med);

    std::vector<float> L_sub(N_ACT);
    std::vector<uint8_t> det_mask(N_ACT, 0);

    #pragma omp parallel for schedule(static) if(inner_omp)
    for (int r = 0; r < H; ++r) {
        const float fy = std::clamp((static_cast<float>(r) - 31.5f) / 64.0f, 0.0f, static_cast<float>(ny - 1));
        const int iy0 = std::min(static_cast<int>(std::floor(fy)), ny - 1);
        const int iy1 = std::min(iy0 + 1, ny - 1);
        const float wy = fy - static_cast<float>(iy0);

        for (int c = 0; c < W; ++c) {
            const float fx = std::clamp((static_cast<float>(c) - 31.5f) / 64.0f, 0.0f, static_cast<float>(nx - 1));
            const int ix0 = std::min(static_cast<int>(std::floor(fx)), nx - 1);
            const int ix1 = std::min(ix0 + 1, nx - 1);
            const float wx = fx - static_cast<float>(ix0);

            const float b00 = cell_bg_sm[iy0 * nx + ix0];
            const float b01 = cell_bg_sm[iy0 * nx + ix1];
            const float b10 = cell_bg_sm[iy1 * nx + ix0];
            const float b11 = cell_bg_sm[iy1 * nx + ix1];
            const float bg = (1.0f - wy) * ((1.0f - wx) * b00 + wx * b01) +
                             wy * ((1.0f - wx) * b10 + wx * b11);

            const float s00 = cell_rms_sm[iy0 * nx + ix0];
            const float s01 = cell_rms_sm[iy0 * nx + ix1];
            const float s10 = cell_rms_sm[iy1 * nx + ix0];
            const float s11 = cell_rms_sm[iy1 * nx + ix1];
            const float sig_loc = std::max(
                global_bg_rms,
                (1.0f - wy) * ((1.0f - wx) * s00 + wx * s01) +
                wy * ((1.0f - wx) * s10 + wx * s11)
            );

            const size_t idx = static_cast<size_t>(r) * W + c;
            const float sub = L[idx] - bg;
            L_sub[idx] = sub;
            if (sub > 5.0f * sig_loc) {
                det_mask[idx] = 1;
            }
        }
    }

    // 5. 8-connected components (minarea >= 5)
    std::vector<uint8_t> visited(N_ACT, 0);
    std::vector<int32_t> queue;
    queue.reserve(4096);

    struct RawComp {
        int32_t area;
        int32_t r_p;
        int32_t c_p;
        float peak_sub;
        float flux_sub;
    };
    std::vector<RawComp> comps;
    comps.reserve(2048);

    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            const size_t start = static_cast<size_t>(r) * W + c;
            if (!det_mask[start] || visited[start]) continue;

            visited[start] = 1;
            queue.clear();
            queue.push_back(static_cast<int32_t>(start));

            int32_t best_r = r, best_c = c;
            float best_val = L_sub[start];
            double flux_sum = 0.0;

            size_t head = 0;
            while (head < queue.size()) {
                const int32_t cur = queue[head++];
                const int cr = cur / W;
                const int cc = cur % W;
                const float v = L_sub[cur];
                flux_sum += static_cast<double>(v);
                if (v > best_val) {
                    best_val = v;
                    best_r = cr;
                    best_c = cc;
                }
                for (int dr = -1; dr <= 1; ++dr) {
                    const int nr = cr + dr;
                    if (nr < 0 || nr >= H) continue;
                    for (int dc = -1; dc <= 1; ++dc) {
                        if (dr == 0 && dc == 0) continue;
                        const int nc = cc + dc;
                        if (nc < 0 || nc >= W) continue;
                        const size_t nidx = static_cast<size_t>(nr) * W + nc;
                        if (det_mask[nidx] && !visited[nidx]) {
                            visited[nidx] = 1;
                            queue.push_back(static_cast<int32_t>(nidx));
                        }
                    }
                }
            }

            if (static_cast<int32_t>(queue.size()) >= 5) {
                comps.push_back(RawComp{
                    static_cast<int32_t>(queue.size()),
                    best_r,
                    best_c,
                    best_val,
                    static_cast<float>(flux_sum)
                });
            }
        }
    }

    *total_5sigma_minarea5_out = static_cast<int>(comps.size());

    std::vector<RawComp> valid_comps;
    valid_comps.reserve(comps.size());
    int sat_rejected = 0;

    for (const auto& cp : comps) {
        if (cp.r_p < 12 || cp.r_p >= H - 12 || cp.c_p < 12 || cp.c_p >= W - 12) {
            continue;
        }
        float max_rgb = 0.0f;
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                const size_t k = static_cast<size_t>(cp.r_p + dr) * W + (cp.c_p + dc);
                max_rgb = std::max({max_rgb, rgb_ptr[3 * k + 0], rgb_ptr[3 * k + 1], rgb_ptr[3 * k + 2]});
            }
        }
        const float peak_l_blk = L[static_cast<size_t>(cp.r_p) * W + cp.c_p];
        if (peak_l_blk >= 11000.0f || max_rgb >= 11000.0f) {
            sat_rejected += 1;
            continue;
        }
        valid_comps.push_back(cp);
    }
    *total_saturated_rejected_out = sat_rejected;

    const int n_fit = std::min(static_cast<int>(valid_comps.size()), max_cands);
    #pragma omp parallel for schedule(dynamic, 8) if(inner_omp)
    for (int i = 0; i < n_fit; ++i) {
        const auto& cp = valid_comps[i];
        CandidateObj obj{};
        obj.area = cp.area;
        obj.r_peak = cp.r_p;
        obj.c_peak = cp.c_p;
        obj.peak_l_above_black = L[static_cast<size_t>(cp.r_p) * W + cp.c_p];
        obj.peak_l_above_bg = cp.peak_sub;
        obj.comp_flux_l = cp.flux_sub;

        float max_rgb = 0.0f;
        float sum_pos_3x3 = 0.0f;
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                const size_t k = static_cast<size_t>(cp.r_p + dr) * W + (cp.c_p + dc);
                max_rgb = std::max({max_rgb, rgb_ptr[3 * k + 0], rgb_ptr[3 * k + 1], rgb_ptr[3 * k + 2]});
                sum_pos_3x3 += std::max(0.0f, L_sub[k]);
            }
        }
        obj.peak_rgb_max_3x3 = max_rgb;
        obj.sharpness_3x3 = (sum_pos_3x3 > 1e-6f) ? (cp.peak_sub / sum_pos_3x3) : 1.0f;

        const float ax_r2 = 0.25f * (
            L_sub[static_cast<size_t>(cp.r_p - 2) * W + cp.c_p] +
            L_sub[static_cast<size_t>(cp.r_p + 2) * W + cp.c_p] +
            L_sub[static_cast<size_t>(cp.r_p) * W + (cp.c_p - 2)] +
            L_sub[static_cast<size_t>(cp.r_p) * W + (cp.c_p + 2)]
        );
        obj.ring_r2_ratio = (cp.peak_sub > 1e-6f) ? (ax_r2 / cp.peak_sub) : 0.0f;

        std::vector<float> cfa_ring;
        cfa_ring.reserve(16);
        for (int dr = -2; dr <= 2; ++dr) {
            for (int dc = -2; dc <= 2; ++dc) {
                if (std::abs(dr) == 2 || std::abs(dc) == 2) {
                    cfa_ring.push_back(cfa_cal[static_cast<size_t>(cp.r_p + dr) * W + (cp.c_p + dc)]);
                }
            }
        }
        const float cfa_bg = median_in_place(cfa_ring);
        float cfa_max_3x3 = 0.0f;
        float cfa_sum_5x5 = 0.0f;
        for (int dr = -2; dr <= 2; ++dr) {
            for (int dc = -2; dc <= 2; ++dc) {
                const float v = std::max(0.0f, cfa_cal[static_cast<size_t>(cp.r_p + dr) * W + (cp.c_p + dc)] - cfa_bg);
                cfa_sum_5x5 += v;
                if (std::abs(dr) <= 1 && std::abs(dc) <= 1) {
                    cfa_max_3x3 = std::max(cfa_max_3x3, v);
                }
            }
        }
        obj.raw_cfa_impulse_ratio = (cfa_sum_5x5 > 1e-6f) ? (cfa_max_3x3 / cfa_sum_5x5) : 0.0f;

        fit_2d_elliptical_gaussian_25x25(
            L.data(), H, W, cp.r_p, cp.c_p, static_cast<double>(global_bg_rms), &obj
        );
        cands_out[i] = obj;
    }

    return n_fit;
}

int analyze_light_frame(
    const char* path,
    const float* master_dark,
    FrameIntegrityMeta* meta_out,
    double* bg_rms_adu_out,
    double* bg_median_adu_out,
    CandidateObj* cands_out,
    int max_cands,
    int* total_5sigma_minarea5_out,
    int* total_saturated_rejected_out,
    float* rgb_out
) {
    return analyze_single_light_internal(
        path, master_dark, meta_out, bg_rms_adu_out, bg_median_adu_out,
        cands_out, max_cands, total_5sigma_minarea5_out, total_saturated_rejected_out,
        rgb_out, true
    );
}

void analyze_light_batch(
    int n_files,
    const char** paths,
    const float* master_dark,
    FrameIntegrityMeta* metas_out,
    double* bg_rms_out,
    double* bg_med_out,
    CandidateObj* cands_flat_out, // n_files * stride_cands
    int stride_cands,
    int* n_fit_out,
    int* tot_5sig_out,
    int* tot_sat_out
) {
    #pragma omp parallel for num_threads(8) schedule(dynamic, 1)
    for (int i = 0; i < n_files; ++i) {
        n_fit_out[i] = analyze_single_light_internal(
            paths[i],
            master_dark,
            &metas_out[i],
            &bg_rms_out[i],
            &bg_med_out[i],
            cands_flat_out + static_cast<size_t>(i) * stride_cands,
            stride_cands,
            &tot_5sig_out[i],
            &tot_sat_out[i],
            nullptr,
            false
        );
    }
}

const char* get_libraw_version_str() {
    return LibRaw::version();
}

} // extern "C"
"""


class FrameIntegrityMetaC(ctypes.Structure):
    _fields_ = [
        ("rc", ctypes.c_int32),
        ("data_error_count", ctypes.c_int32),
        ("first_data_error_offset", ctypes.c_int32),
        ("process_warnings", ctypes.c_uint32),
        ("raw_width", ctypes.c_uint16),
        ("raw_height", ctypes.c_uint16),
        ("top_margin", ctypes.c_uint16),
        ("left_margin", ctypes.c_uint16),
        ("width", ctypes.c_uint16),
        ("height", ctypes.c_uint16),
        ("flip", ctypes.c_int32),
        ("iso_speed", ctypes.c_float),
        ("shutter", ctypes.c_float),
        ("camera_temp_c", ctypes.c_float),
        ("timestamp", ctypes.c_int64),
        ("active_min", ctypes.c_uint16),
        ("active_max", ctypes.c_uint16),
        ("active_p1", ctypes.c_double),
        ("ob_left_median", ctypes.c_double),
        ("cfa_is_rggb", ctypes.c_int32),
        ("is_clean", ctypes.c_int32),
        ("error_msg", ctypes.c_char * 128),
    ]


class CandidateObjC(ctypes.Structure):
    _fields_ = [
        ("area", ctypes.c_int32),
        ("r_peak", ctypes.c_int32),
        ("c_peak", ctypes.c_int32),
        ("peak_l_above_black", ctypes.c_float),
        ("peak_l_above_bg", ctypes.c_float),
        ("peak_rgb_max_3x3", ctypes.c_float),
        ("comp_flux_l", ctypes.c_float),
        ("sharpness_3x3", ctypes.c_float),
        ("ring_r2_ratio", ctypes.c_float),
        ("raw_cfa_impulse_ratio", ctypes.c_float),
        ("fit_ok", ctypes.c_int32),
        ("x_centroid", ctypes.c_double),
        ("y_centroid", ctypes.c_double),
        ("bg_local", ctypes.c_double),
        ("amp", ctypes.c_double),
        ("fwhm_px", ctypes.c_double),
        ("fwhm_major_px", ctypes.c_double),
        ("fwhm_minor_px", ctypes.c_double),
        ("eccentricity", ctypes.c_double),
        ("theta_rad", ctypes.c_double),
        ("r2_core", ctypes.c_double),
        ("chi2_red", ctypes.c_double),
        ("flux_gauss", ctypes.c_double),
    ]


def is_strict_cr2(filename: str) -> bool:
    """Strictly open only .CR2 files (excluding .CR2.xmp, .JPG, .png, .npz, .xcf)."""
    return len(filename) > 4 and filename.upper().endswith(".CR2")


def list_strict_cr2_files(dir_path: Path) -> list[Path]:
    files = []
    for fname in sorted(os.listdir(dir_path)):
        p = dir_path / fname
        if p.is_file() and is_strict_cr2(fname):
            files.append(p)
    return files


class CppRefShim:
    def __init__(self, tmp_dir: str) -> None:
        so_path = os.path.join(tmp_dir, "libref_gen.so")
        cmd = [
            "g++",
            "-std=c++20",
            "-O3",
            "-march=native",
            "-fopenmp",
            "-shared",
            "-fPIC",
            "-x",
            "c++",
            "-",
            "-lraw_r",
            "-o",
            so_path,
        ]
        proc = subprocess.run(
            cmd,
            input=CPP_SOURCE,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to compile C++20 shim:\n{proc.stderr}")

        self.lib = ctypes.CDLL(so_path)

        self.lib.mhc_demosaic_rggb_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.mhc_demosaic_rggb_f32.restype = None

        self.lib.build_master_dark.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(FrameIntegrityMetaC),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.build_master_dark.restype = ctypes.c_int

        self.lib.analyze_light_frame.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(FrameIntegrityMetaC),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(CandidateObjC),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.analyze_light_frame.restype = ctypes.c_int

        self.lib.analyze_light_batch.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(FrameIntegrityMetaC),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(CandidateObjC),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.analyze_light_batch.restype = None

        self.lib.get_libraw_version_str.argtypes = []
        self.lib.get_libraw_version_str.restype = ctypes.c_char_p
        self.libraw_version = self.lib.get_libraw_version_str().decode("utf-8", errors="replace")

    def verify_mhc_unit_tests(self) -> dict[str, Any]:
        """Verify exact constant-color conservation (including 2-px borders) and impulse symmetry."""
        h, w = 32, 48
        cfa = np.empty((h, w), dtype=np.float32)
        cfa[0::2, 0::2] = 100.0  # R
        cfa[0::2, 1::2] = 200.0  # G1
        cfa[1::2, 0::2] = 200.0  # G2
        cfa[1::2, 1::2] = 150.0  # B

        rgb = np.empty((h, w, 3), dtype=np.float32)
        self.lib.mhc_demosaic_rggb_f32(
            cfa.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            h,
            w,
            rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

        err_r = float(np.max(np.abs(rgb[:, :, 0] - 100.0)))
        err_g = float(np.max(np.abs(rgb[:, :, 1] - 200.0)))
        err_b = float(np.max(np.abs(rgb[:, :, 2] - 150.0)))
        if max(err_r, err_g, err_b) > 0.0:
            raise AssertionError(
                f"MHC 5x5 constant-color test failed: max_err=({err_r}, {err_g}, {err_b})"
            )

        cfa_neg = np.empty((h, w), dtype=np.float32)
        cfa_neg[0::2, 0::2] = -25.5
        cfa_neg[0::2, 1::2] = -10.25
        cfa_neg[1::2, 0::2] = -10.25
        cfa_neg[1::2, 1::2] = -40.75
        rgb_neg = np.empty((h, w, 3), dtype=np.float32)
        self.lib.mhc_demosaic_rggb_f32(
            cfa_neg.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            h,
            w,
            rgb_neg.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        neg_err = float(
            max(
                np.max(np.abs(rgb_neg[:, :, 0] - (-25.5))),
                np.max(np.abs(rgb_neg[:, :, 1] - (-10.25))),
                np.max(np.abs(rgb_neg[:, :, 2] - (-40.75))),
            )
        )
        if neg_err > 0.0:
            raise AssertionError(f"MHC 5x5 negative constant-color test failed: {neg_err}")

        impulse_stats = {}
        for name, (ir, ic) in [
            ("R_site", (10, 10)),
            ("G1_site", (10, 11)),
            ("G2_site", (11, 10)),
            ("B_site", (11, 11)),
        ]:
            imp = np.zeros((24, 24), dtype=np.float32)
            imp[ir, ic] = 1000.0
            imp_rgb = np.empty((24, 24, 3), dtype=np.float32)
            self.lib.mhc_demosaic_rggb_f32(
                imp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                24,
                24,
                imp_rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            L_imp = 0.25 * imp_rgb[:, :, 0] + 0.50 * imp_rgb[:, :, 1] + 0.25 * imp_rgb[:, :, 2]
            patch3 = np.maximum(0.0, L_imp[ir - 1 : ir + 2, ic - 1 : ic + 2])
            ax2 = 0.25 * (L_imp[ir - 2, ic] + L_imp[ir + 2, ic] + L_imp[ir, ic - 2] + L_imp[ir, ic + 2])
            impulse_stats[name] = {
                "center_L_per_1000_adu": round(float(L_imp[ir, ic]), 4),
                "axial_1px_L_per_1000_adu": round(float(L_imp[ir - 1, ic]), 4),
                "diag_1px_L_per_1000_adu": round(float(L_imp[ir - 1, ic - 1]), 4),
                "axial_2px_ring_L_per_1000_adu": round(float(ax2), 4),
                "sharpness_3x3": round(float(L_imp[ir, ic] / np.sum(patch3)), 4),
                "pixels_gt_75_adu_at_1000_impulse": int(np.count_nonzero(L_imp > 75.0)),
                "pixels_gt_75_adu_at_1500_impulse": int(np.count_nonzero(L_imp * 1.5 > 75.0)),
            }

        return {
            "constant_color_conservation_passed": True,
            "constant_rgb_tested": [100.0, 200.0, 150.0],
            "max_abs_error_per_channel": [err_r, err_g, err_b],
            "negative_excursion_preservation_passed": True,
            "mhc_5x5_single_pixel_impulse_analytic_profile": impulse_stats,
        }


def meta_to_dict(fname: str, m: FrameIntegrityMetaC) -> dict[str, Any]:
    reasons = []
    if m.rc != 0:
        reasons.append(f"libraw_rc={m.rc} ({m.error_msg.decode('utf-8', 'replace')})")
    if m.data_error_count > 0:
        reasons.append(
            f"libraw_data_error_count={m.data_error_count} (first_offset={m.first_data_error_offset})"
        )
    if m.active_max > 16383:
        reasons.append(f"active_max={m.active_max} > 16383")
    if m.active_p1 < 1850.0:
        reasons.append(f"active_p1={m.active_p1:.2f} < 1850")
    if m.cfa_is_rggb == 0:
        reasons.append("cfa_not_rggb")

    temp_c = None if m.camera_temp_c <= -999.0 else round(float(m.camera_temp_c), 2)
    return {
        "filename": fname,
        "is_clean": bool(m.is_clean),
        "rejection_reasons": reasons,
        "timestamp_unix": int(m.timestamp),
        "exposure_s": round(float(m.shutter), 3),
        "iso": round(float(m.iso_speed), 1),
        "camera_temp_c": temp_c,
        "exif_flip_ignored": int(m.flip),
        "active_min_adu": int(m.active_min),
        "active_p1_adu": round(float(m.active_p1), 3),
        "active_max_adu": int(m.active_max),
        "left_optical_black_median_adu": round(float(m.ob_left_median), 3),
    }


def classify_candidates(cands: list[CandidateObjC], bg_rms_adu: float) -> dict[str, Any]:
    """Separate MHC warm-pixel rings, extended/blended non-PSF objects, and clean optical stars."""
    half_diag = math.hypot(ACTIVE_W / 2.0, ACTIVE_H / 2.0)
    corner_thresh_r = 0.65 * half_diag
    cx = (ACTIVE_W - 1) / 2.0
    cy = (ACTIVE_H - 1) / 2.0

    warm_pixels = []
    extended_or_bad = []
    clean_stars = []

    raw_hot_total = 0
    raw_hot_caught_by_l_filter = 0
    false_pos_warm_on_real_star = 0

    for c in cands:
        is_raw_impulse = c.raw_cfa_impulse_ratio > 0.55
        if is_raw_impulse:
            raw_hot_total += 1

        # Pure L-based warm-pixel ring filter (applicable directly to RGB stacks in evaluate.py):
        # MHC 5x5 single-pixel impulses have fwhm < 1.5 px, high central sharpness (>0.38),
        # or negative radius-2 MHC ringing (< -0.02 of peak).
        is_warm_l = (
            (not c.fit_ok)
            or (c.fwhm_px < 1.5)
            or (c.sharpness_3x3 > 0.38)
            or (c.ring_r2_ratio < -0.02)
        )
        if (not c.fit_ok) and c.sharpness_3x3 < 0.25 and c.area > 25:
            is_warm_l = False

        if is_warm_l:
            if is_raw_impulse:
                raw_hot_caught_by_l_filter += 1
            elif c.fit_ok and c.fwhm_px >= 2.0 and c.chi2_red <= 3.0 and c.amp >= 3.5 * bg_rms_adu:
                false_pos_warm_on_real_star += 1
            warm_pixels.append(
                {
                    "x": round(float(c.x_centroid if c.fit_ok else c.c_peak), 3),
                    "y": round(float(c.y_centroid if c.fit_ok else c.r_peak), 3),
                    "r_peak": int(c.r_peak),
                    "c_peak": int(c.c_peak),
                    "area": int(c.area),
                    "peak_l_above_bg": round(float(c.peak_l_above_bg), 2),
                    "fwhm_px": round(float(c.fwhm_px), 3) if c.fit_ok else None,
                    "sharpness_3x3": round(float(c.sharpness_3x3), 4),
                    "ring_r2_ratio": round(float(c.ring_r2_ratio), 4),
                    "raw_cfa_impulse_ratio": round(float(c.raw_cfa_impulse_ratio), 4),
                }
            )
            continue

        # Reject extended nebulae (e.g. M57 ring, M31 core), blended binaries, or noise-chased fits
        # using unbiased Poisson+sky reduced chi-squared rather than SNR-biased raw R^2!
        if (
            (not c.fit_ok)
            or c.amp < 3.5 * bg_rms_adu
            or c.fwhm_px > 20.0
            or c.fwhm_major_px > 26.0
            or c.chi2_red > 4.0
            or c.area > 2000
        ):
            extended_or_bad.append(
                {
                    "r_peak": int(c.r_peak),
                    "c_peak": int(c.c_peak),
                    "area": int(c.area),
                    "fwhm_px": round(float(c.fwhm_px), 3) if c.fit_ok else None,
                    "chi2_red": round(float(c.chi2_red), 4) if c.fit_ok else None,
                    "amp_snr": round(float(c.amp / max(1e-6, bg_rms_adu)), 2) if c.fit_ok else None,
                }
            )
            continue

        r_dist = math.hypot(c.x_centroid - cx, c.y_centroid - cy)
        clean_stars.append(
            {
                "x": float(c.x_centroid),
                "y": float(c.y_centroid),
                "r_peak": int(c.r_peak),
                "c_peak": int(c.c_peak),
                "area": int(c.area),
                "flux": float(c.flux_gauss),
                "comp_flux_l": float(c.comp_flux_l),
                "amp": float(c.amp),
                "peak_l_above_bg": float(c.peak_l_above_bg),
                "peak_l_above_black": float(c.peak_l_above_black),
                "fwhm_px": float(c.fwhm_px),
                "fwhm_major_px": float(c.fwhm_major_px),
                "fwhm_minor_px": float(c.fwhm_minor_px),
                "eccentricity": float(c.eccentricity),
                "theta_rad": float(c.theta_rad),
                "r2_core": float(c.r2_core),
                "chi2_red": float(c.chi2_red),
                "sharpness_3x3": float(c.sharpness_3x3),
                "ring_r2_ratio": float(c.ring_r2_ratio),
                "raw_cfa_impulse_ratio": float(c.raw_cfa_impulse_ratio),
                "is_corner": r_dist > corner_thresh_r,
            }
        )

    fwhms = [s["fwhm_px"] for s in clean_stars]
    corner_fwhms = [s["fwhm_px"] for s in clean_stars if s["is_corner"]]
    eccs = [s["eccentricity"] for s in clean_stars]

    return {
        "clean_stars": clean_stars,
        "warm_pixels": warm_pixels,
        "extended_or_bad": extended_or_bad,
        "raw_hot_total_in_minarea5": raw_hot_total,
        "raw_hot_caught_by_l_filter": raw_hot_caught_by_l_filter,
        "false_pos_warm_on_real_star": false_pos_warm_on_real_star,
        "star_count": len(clean_stars),
        "corner_star_count": len(corner_fwhms),
        "median_fwhm_px": round(float(np.median(fwhms)), 4) if fwhms else None,
        "corner_fwhm_px": round(float(np.median(corner_fwhms)), 4) if corner_fwhms else None,
        "median_eccentricity": round(float(np.median(eccs)), 4) if eccs else None,
    }


def select_top_isolated_stars(
    stars: list[dict[str, Any]], top_k: int = 40, min_sep_px: float = 14.0
) -> np.ndarray:
    """Return (M, 2) array of (x, y) for up to top_k brightest isolated non-saturated stars."""
    if not stars:
        return np.zeros((0, 2), dtype=np.float64)
    all_xy = np.array([[s["x"], s["y"]] for s in stars], dtype=np.float64)
    order = np.argsort([-s["flux"] for s in stars])
    chosen = []
    for idx in order:
        pt = all_xy[idx]
        dists = np.hypot(all_xy[:, 0] - pt[0], all_xy[:, 1] - pt[1])
        dists[idx] = np.inf
        if np.min(dists) >= min_sep_px:
            chosen.append(pt)
            if len(chosen) >= top_k:
                break
    return np.array(chosen, dtype=np.float64) if chosen else np.zeros((0, 2), dtype=np.float64)


def _build_triangle_invariants(
    pts: np.ndarray, max_tri_stars: int = 26
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Valdes et al. (1995) triangle invariants (b/a, c/a) + signed chirality."""
    sub = pts[: min(len(pts), max_tri_stars)]
    m = len(sub)
    if m < 3:
        return np.zeros((0, 2)), np.zeros((0,), dtype=np.int8), np.zeros((0, 3), dtype=np.int32)

    invs = []
    chirs = []
    verts = []
    for i in range(m):
        for j in range(i + 1, m):
            for k in range(j + 1, m):
                d01 = math.hypot(sub[i, 0] - sub[j, 0], sub[i, 1] - sub[j, 1])
                d12 = math.hypot(sub[j, 0] - sub[k, 0], sub[j, 1] - sub[k, 1])
                d20 = math.hypot(sub[k, 0] - sub[i, 0], sub[k, 1] - sub[i, 1])
                sides_opp = [(d12, i), (d20, j), (d01, k)]
                sides_opp.sort(key=lambda t: t[0])
                c_len, v_c = sides_opp[0]
                b_len, v_b = sides_opp[1]
                a_len, v_a = sides_opp[2]
                if a_len < 120.0 or c_len < 40.0:
                    continue
                pa, pb, pc = sub[v_a], sub[v_b], sub[v_c]
                cross = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
                chir = 1 if cross >= 0.0 else -1
                invs.append([b_len / a_len, c_len / a_len])
                chirs.append(chir)
                verts.append([v_a, v_b, v_c])

    if not invs:
        return np.zeros((0, 2)), np.zeros((0,), dtype=np.int8), np.zeros((0, 3), dtype=np.int32)
    return (
        np.array(invs, dtype=np.float64),
        np.array(chirs, dtype=np.int8),
        np.array(verts, dtype=np.int32),
    )


def _fit_rigid_centered(
    src: np.ndarray, dst: np.ndarray, cx: float, cy: float
) -> tuple[float, float, float, float]:
    """Fit 2D rigid + 4-DOF scale transform centered at (cx, cy):
    dst_c = R(dtheta) @ src_c + [dx, dy].
    """
    sc = src - np.array([cx, cy])
    dc = dst - np.array([cx, cy])
    mu_s = np.mean(sc, axis=0)
    mu_d = np.mean(dc, axis=0)
    xs = sc - mu_s
    xd = dc - mu_d

    c00 = float(np.sum(xs[:, 0] * xd[:, 0] + xs[:, 1] * xd[:, 1]))
    c10 = float(np.sum(xs[:, 0] * xd[:, 1] - xs[:, 1] * xd[:, 0]))
    dtheta = math.atan2(c10, c00)

    ct = math.cos(dtheta)
    st = math.sin(dtheta)
    rot_mu_s = np.array([ct * mu_s[0] - st * mu_s[1], st * mu_s[0] + ct * mu_s[1]])
    t_rigid = mu_d - rot_mu_s

    denom = float(np.sum(xs * xs))
    scale = math.hypot(c00, c10) / denom if denom > 1e-12 else 1.0
    return float(t_rigid[0]), float(t_rigid[1]), dtheta, scale


def match_and_measure_drift(ref_pts: np.ndarray, frm_pts: np.ndarray) -> dict[str, Any]:
    """Match top-40 bright isolated stars between reference frame and target frame via
    Valdes triangle invariants + RANSAC + iterative mutual-NN rigid Procrustes refinement.
    """
    cx = (ACTIVE_W - 1) / 2.0
    cy = (ACTIVE_H - 1) / 2.0

    if len(ref_pts) < 4 or len(frm_pts) < 4:
        raise RuntimeError(
            f"Insufficient isolated bright stars for matching ({len(ref_pts)}, {len(frm_pts)})"
        )

    inv_r, chi_r, ver_r = _build_triangle_invariants(ref_pts, max_tri_stars=26)
    inv_f, chi_f, ver_f = _build_triangle_invariants(frm_pts, max_tri_stars=26)

    votes = np.zeros((len(ref_pts), len(frm_pts)), dtype=np.int32)
    tol = 0.0035
    for sign in (-1, 1):
        idx_r = np.where(chi_r == sign)[0]
        idx_f = np.where(chi_f == sign)[0]
        if len(idx_r) == 0 or len(idx_f) == 0:
            continue
        ir = inv_r[idx_r]
        if_ = inv_f[idx_f]
        diff0 = np.abs(ir[:, None, 0] - if_[None, :, 0])
        diff1 = np.abs(ir[:, None, 1] - if_[None, :, 1])
        pairs_r, pairs_f = np.where((diff0 < tol) & (diff1 < tol))
        for pr, pf in zip(pairs_r, pairs_f):
            vr = ver_r[idx_r[pr]]
            vf = ver_f[idx_f[pf]]
            votes[vr[0], vf[0]] += 1
            votes[vr[1], vf[1]] += 1
            votes[vr[2], vf[2]] += 1

    flat_order = np.argsort(-votes.ravel())
    putative = []
    for k in flat_order[:80]:
        v = int(votes.ravel()[k])
        if v < 2:
            break
        ri, fi = divmod(int(k), len(frm_pts))
        putative.append((ri, fi))

    if len(putative) < 3:
        raise RuntimeError("Triangle voting found fewer than 3 putative star matches")

    rng = np.random.default_rng(42)
    best_inliers = -1
    best_params = (0.0, 0.0, 0.0, 1.0)

    n_put = len(putative)
    for _ in range(800):
        i1, i2 = rng.choice(n_put, size=2, replace=False)
        r1, f1 = putative[i1]
        r2, f2 = putative[i2]
        if r1 == r2 or f1 == f2:
            continue
        src2 = np.array([ref_pts[r1], ref_pts[r2]])
        dst2 = np.array([frm_pts[f1], frm_pts[f2]])
        if np.hypot(*(src2[1] - src2[0])) < 50.0:
            continue
        dx, dy, dth, _ = _fit_rigid_centered(src2, dst2, cx, cy)
        ct, st = math.cos(dth), math.sin(dth)
        rc = ref_pts - np.array([cx, cy])
        pred = np.column_stack(
            [ct * rc[:, 0] - st * rc[:, 1] + dx + cx, st * rc[:, 0] + ct * rc[:, 1] + dy + cy]
        )
        dmat = np.hypot(pred[:, None, 0] - frm_pts[None, :, 0], pred[:, None, 1] - frm_pts[None, :, 1])
        min_d = np.min(dmat, axis=1)
        n_in = int(np.count_nonzero(min_d < 2.0))
        if n_in > best_inliers:
            best_inliers = n_in
            best_params = (dx, dy, dth, 1.0)

    dx, dy, dth, scale = best_params
    matched_r = []
    matched_f = []
    for radius_gate in (2.5, 1.5):
        ct, st = math.cos(dth), math.sin(dth)
        rc = ref_pts - np.array([cx, cy])
        pred = np.column_stack(
            [ct * rc[:, 0] - st * rc[:, 1] + dx + cx, st * rc[:, 0] + ct * rc[:, 1] + dy + cy]
        )
        dmat = np.hypot(pred[:, None, 0] - frm_pts[None, :, 0], pred[:, None, 1] - frm_pts[None, :, 1])
        matched_r = []
        matched_f = []
        for ri in range(len(ref_pts)):
            fi = int(np.argmin(dmat[ri]))
            if dmat[ri, fi] < radius_gate and int(np.argmin(dmat[:, fi])) == ri:
                matched_r.append(ri)
                matched_f.append(fi)
        if len(matched_r) >= 3:
            dx, dy, dth, scale = _fit_rigid_centered(ref_pts[matched_r], frm_pts[matched_f], cx, cy)

    src_m = ref_pts[matched_r]
    dst_m = frm_pts[matched_f]
    ct, st = math.cos(dth), math.sin(dth)
    sc = src_m - np.array([cx, cy])
    pred_rigid = np.column_stack(
        [ct * sc[:, 0] - st * sc[:, 1] + dx + cx, st * sc[:, 0] + ct * sc[:, 1] + dy + cy]
    )
    res_rigid = np.hypot(dst_m[:, 0] - pred_rigid[:, 0], dst_m[:, 1] - pred_rigid[:, 1])
    rms_rigid = float(np.sqrt(np.mean(res_rigid**2)))

    rms_affine = rms_rigid
    if len(src_m) >= 6:
        A_des = np.column_stack([sc[:, 0], sc[:, 1], np.ones(len(sc))])
        sol_x, _, _, _ = np.linalg.lstsq(A_des, dst_m[:, 0] - cx, rcond=None)
        sol_y, _, _, _ = np.linalg.lstsq(A_des, dst_m[:, 1] - cy, rcond=None)
        px = A_des @ sol_x + cx
        py = A_des @ sol_y + cy
        rms_affine = float(np.sqrt(np.mean((dst_m[:, 0] - px) ** 2 + (dst_m[:, 1] - py) ** 2)))

    t_origin_x = dx + cx - (ct * cx - st * cy)
    t_origin_y = dy + cy - (st * cx + ct * cy)

    return {
        "dx_px": round(dx, 4),
        "dy_px": round(dy, 4),
        "drift_radius_px": round(math.hypot(dx, dy), 4),
        "dtheta_deg": round(math.degrees(dth), 5),
        "scale_4dof": round(scale, 7),
        "tx_origin_00_px": round(t_origin_x, 4),
        "ty_origin_00_px": round(t_origin_y, 4),
        "matched_stars_of_top40": len(matched_r),
        "ref_isolated_pool_count": len(ref_pts),
        "frame_isolated_pool_count": len(frm_pts),
        "rigid_rms_residual_px": round(rms_rigid, 4),
        "affine_rms_residual_px": round(rms_affine, 4),
        "max_inlier_residual_px": round(float(np.max(res_rigid)), 4),
    }


def select_best_reference_frame(
    clean_frame_entries: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select the single sharpest clean frame using an objective sharpness rank:
    lowest median_fwhm_px among frames with healthy star_count (>= 75% of dataset median)
    and low median_eccentricity (<= dataset median eccentricity, i.e. the rounder half of frames).
    """
    counts = [f["metrics"]["star_count"] for f in clean_frame_entries]
    eccs = [f["metrics"]["median_eccentricity"] for f in clean_frame_entries]

    med_count = float(np.median(counts))
    med_ecc = float(np.median(eccs))

    min_healthy_stars = max(25, int(math.floor(0.75 * med_count)))
    max_healthy_ecc = round(med_ecc, 4)

    eligible = []
    for f in clean_frame_entries:
        m = f["metrics"]
        ok_stars = m["star_count"] >= min_healthy_stars
        ok_ecc = m["median_eccentricity"] <= max_healthy_ecc
        f["selection_eligibility"] = {
            "is_eligible": bool(ok_stars and ok_ecc),
            "passes_star_count_gate": bool(ok_stars),
            "passes_eccentricity_gate": bool(ok_ecc),
        }
        if ok_stars and ok_ecc:
            eligible.append(f)

    if not eligible:
        eligible = list(clean_frame_entries)

    eligible_sorted = sorted(
        eligible,
        key=lambda f: (
            f["metrics"]["median_fwhm_px"],
            f["metrics"]["corner_fwhm_px"] if f["metrics"]["corner_fwhm_px"] is not None else 999.0,
            f["metrics"]["median_eccentricity"],
        ),
    )
    for rank_idx, f in enumerate(eligible_sorted, start=1):
        f["sharpness_rank_among_eligible"] = rank_idx

    winner = eligible_sorted[0]
    runner_up = eligible_sorted[1] if len(eligible_sorted) > 1 else None

    rationale = {
        "rule": (
            "Lowest median_fwhm_px (tie-broken by corner_fwhm_px, then median_eccentricity) "
            "among clean frames passing healthy star_count (>= floor(0.75 * dataset_median)) "
            "and low median_eccentricity (<= dataset_median_eccentricity, the rounder half of frames)."
        ),
        "dataset_median_star_count": round(med_count, 1),
        "min_healthy_star_count_gate": min_healthy_stars,
        "dataset_median_eccentricity": round(med_ecc, 4),
        "max_healthy_eccentricity_gate": max_healthy_ecc,
        "eligible_frame_count": len(eligible),
        "total_clean_frame_count": len(clean_frame_entries),
        "winning_filename": winner["filename"],
        "winning_metrics": winner["metrics"],
        "runner_up_filename": runner_up["filename"] if runner_up else None,
        "runner_up_metrics": runner_up["metrics"] if runner_up else None,
        "top_5_ranked_eligible": [
            {
                "rank": i + 1,
                "filename": f["filename"],
                "median_fwhm_px": f["metrics"]["median_fwhm_px"],
                "corner_fwhm_px": f["metrics"]["corner_fwhm_px"],
                "median_eccentricity": f["metrics"]["median_eccentricity"],
                "star_count": f["metrics"]["star_count"],
                "bg_rms_adu": f["metrics"]["bg_rms_adu"],
            }
            for i, f in enumerate(eligible_sorted[:5])
        ],
    }
    return winner, rationale


def check_warm_pixel_sensor_stationarity(
    frame_warm_lists: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Verify empirically whether rejected warm-pixel candidates remain locked at (0, 0)
    sensor coordinates across frames (confirming sensor origin vs sky origin).
    """
    if len(frame_warm_lists) < 2 or not frame_warm_lists[0]:
        return {"recurrent_sensor_locked_fraction": None, "sample_count": 0}

    coord_counts: Counter[tuple[int, int]] = Counter()
    for wlist in frame_warm_lists:
        seen_in_frame = {(int(w["r_peak"]), int(w["c_peak"])) for w in wlist}
        for rc in seen_in_frame:
            coord_counts[rc] += 1

    recurrent_sites = {rc for rc, cnt in coord_counts.items() if cnt >= 3}
    total_instances = sum(len(w) for w in frame_warm_lists)
    recurrent_instances = 0
    for wlist in frame_warm_lists:
        for w in wlist:
            rc = (int(w["r_peak"]), int(w["c_peak"]))
            if any(
                (rc[0] + dr, rc[1] + dc) in recurrent_sites
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
            ):
                recurrent_instances += 1

    return {
        "unique_sensor_sites_appearing_ge_3_frames": len(recurrent_sites),
        "total_warm_pixel_detections_across_clean_frames": total_instances,
        "detections_within_1px_of_recurrent_sensor_site": recurrent_instances,
        "sensor_locked_recurrence_fraction": (
            round(recurrent_instances / total_instances, 4) if total_instances > 0 else 0.0
        ),
    }


def process_light_dataset(
    set_name: str,
    set_dir: Path,
    shim: CppRefShim,
    master_dark: np.ndarray,
    out_npy_path: Path,
) -> dict[str, Any]:
    cr2_files = list_strict_cr2_files(set_dir)
    n_files = len(cr2_files)
    md_ptr = master_dark.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    stride_cands = 4096
    c_paths = (ctypes.c_char_p * n_files)(*[os.fsencode(str(p)) for p in cr2_files])
    metas_arr = (FrameIntegrityMetaC * n_files)()
    bg_rms_arr = (ctypes.c_double * n_files)()
    bg_med_arr = (ctypes.c_double * n_files)()
    n_fit_arr = (ctypes.c_int * n_files)()
    tot_5sig_arr = (ctypes.c_int * n_files)()
    tot_sat_arr = (ctypes.c_int * n_files)()
    cands_flat = (CandidateObjC * (n_files * stride_cands))()

    shim.lib.analyze_light_batch(
        n_files,
        c_paths,
        md_ptr,
        metas_arr,
        bg_rms_arr,
        bg_med_arr,
        cands_flat,
        stride_cands,
        n_fit_arr,
        tot_5sig_arr,
        tot_sat_arr,
    )

    clean_entries = []
    rejected_entries = []
    raw_frames_internal = []

    for i, fpath in enumerate(cr2_files):
        m_dict = meta_to_dict(fpath.name, metas_arr[i])
        if not m_dict["is_clean"]:
            rejected_entries.append(m_dict)
            continue

        n_fit = int(n_fit_arr[i])
        base_off = i * stride_cands
        cands_list = [cands_flat[base_off + k] for k in range(n_fit)]
        bg_rms_val = float(bg_rms_arr[i])
        cls = classify_candidates(cands_list, bg_rms_val)

        entry = {
            **m_dict,
            "filepath": fpath,
            "metrics": {
                "star_count": cls["star_count"],
                "corner_star_count": cls["corner_star_count"],
                "median_fwhm_px": cls["median_fwhm_px"],
                "corner_fwhm_px": cls["corner_fwhm_px"],
                "median_eccentricity": cls["median_eccentricity"],
                "bg_rms_adu": round(bg_rms_val, 4),
                "bg_median_adu": round(float(bg_med_arr[i]), 3),
            },
            "candidate_breakdown": {
                "total_5sigma_minarea5": int(tot_5sig_arr[i]),
                "saturated_rejected_ge_11000_adu": int(tot_sat_arr[i]),
                "warm_pixel_rings_rejected": len(cls["warm_pixels"]),
                "extended_or_blended_rejected": len(cls["extended_or_bad"]),
                "clean_stars_accepted": cls["star_count"],
                "raw_cfa_hot_impulses_in_minarea5": cls["raw_hot_total_in_minarea5"],
                "raw_cfa_hot_caught_by_pure_l_filter": cls["raw_hot_caught_by_l_filter"],
                "false_positive_warm_on_real_star": cls["false_pos_warm_on_real_star"],
            },
            "_clean_stars": cls["clean_stars"],
            "_warm_pixels": cls["warm_pixels"],
        }
        raw_frames_internal.append(entry)

    raw_frames_internal.sort(key=lambda e: (e["timestamp_unix"], e["filename"]))

    winner, selection_rationale = select_best_reference_frame(raw_frames_internal)

    # Generate and save winning (2856, 4290, 3) float32 reference array
    win_rgb = np.empty((ACTIVE_H, ACTIVE_W, 3), dtype=np.float32)
    win_meta_verify = FrameIntegrityMetaC()
    win_bg_rms = ctypes.c_double(0.0)
    win_bg_med = ctypes.c_double(0.0)
    win_5sig = ctypes.c_int(0)
    win_sat = ctypes.c_int(0)
    win_cands = (CandidateObjC * stride_cands)()

    shim.lib.analyze_light_frame(
        os.fsencode(str(winner["filepath"])),
        md_ptr,
        ctypes.byref(win_meta_verify),
        ctypes.byref(win_bg_rms),
        ctypes.byref(win_bg_med),
        win_cands,
        stride_cands,
        ctypes.byref(win_5sig),
        ctypes.byref(win_sat),
        win_rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    out_npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy_path, win_rgb)

    neg_count = int(np.count_nonzero(win_rgb < 0.0))
    win_array_stats = {
        "output_path": str(out_npy_path),
        "shape": list(win_rgb.shape),
        "dtype": str(win_rgb.dtype),
        "min_val": round(float(np.min(win_rgb)), 3),
        "p01_val": round(float(np.percentile(win_rgb, 1.0)), 3),
        "median_per_channel_rgb": [
            round(float(np.median(win_rgb[:, :, ch])), 3) for ch in range(3)
        ],
        "max_val": round(float(np.max(win_rgb)), 3),
        "negative_pixel_value_count": neg_count,
        "negative_excursions_preserved": neg_count > 0,
    }

    # Compute empirical frame-to-frame registration drift (dx, dy, dtheta_deg) vs winning reference
    ref_top40 = select_top_isolated_stars(winner["_clean_stars"], top_k=40, min_sep_px=14.0)
    drift_table = []
    t0_unix = raw_frames_internal[0]["timestamp_unix"] if raw_frames_internal else 0

    for idx, frm in enumerate(raw_frames_internal):
        frm_top40 = select_top_isolated_stars(frm["_clean_stars"], top_k=40, min_sep_px=14.0)
        if frm["filename"] == winner["filename"]:
            drift_info = {
                "dx_px": 0.0,
                "dy_px": 0.0,
                "drift_radius_px": 0.0,
                "dtheta_deg": 0.0,
                "scale_4dof": 1.0,
                "tx_origin_00_px": 0.0,
                "ty_origin_00_px": 0.0,
                "matched_stars_of_top40": len(ref_top40),
                "ref_isolated_pool_count": len(ref_top40),
                "frame_isolated_pool_count": len(frm_top40),
                "rigid_rms_residual_px": 0.0,
                "affine_rms_residual_px": 0.0,
                "max_inlier_residual_px": 0.0,
            }
        else:
            drift_info = match_and_measure_drift(ref_top40, frm_top40)

        drift_entry = {
            "chronological_index": idx,
            "filename": frm["filename"],
            "is_reference": frm["filename"] == winner["filename"],
            "elapsed_s_from_set_start": int(frm["timestamp_unix"] - t0_unix),
            "camera_temp_c": frm["camera_temp_c"],
            **drift_info,
        }
        drift_table.append(drift_entry)
        frm["drift_vs_reference"] = drift_info

    step_dists = []
    for i in range(len(drift_table) - 1):
        ddx = drift_table[i + 1]["dx_px"] - drift_table[i]["dx_px"]
        ddy = drift_table[i + 1]["dy_px"] - drift_table[i]["dy_px"]
        ddth = drift_table[i + 1]["dtheta_deg"] - drift_table[i]["dtheta_deg"]
        dist = math.hypot(ddx, ddy)
        drift_table[i + 1]["step_from_prev_dx_px"] = round(ddx, 4)
        drift_table[i + 1]["step_from_prev_dy_px"] = round(ddy, 4)
        drift_table[i + 1]["step_from_prev_dist_px"] = round(dist, 4)
        drift_table[i + 1]["step_from_prev_dtheta_deg"] = round(ddth, 5)
        step_dists.append(dist)
    if drift_table:
        drift_table[0]["step_from_prev_dx_px"] = 0.0
        drift_table[0]["step_from_prev_dy_px"] = 0.0
        drift_table[0]["step_from_prev_dist_px"] = 0.0
        drift_table[0]["step_from_prev_dtheta_deg"] = 0.0

    warm_fwhms = [
        w["fwhm_px"]
        for f in raw_frames_internal
        for w in f["_warm_pixels"]
        if w["fwhm_px"] is not None
    ]
    warm_sharps = [w["sharpness_3x3"] for f in raw_frames_internal for w in f["_warm_pixels"]]
    warm_rings = [w["ring_r2_ratio"] for f in raw_frames_internal for w in f["_warm_pixels"]]
    warm_cfa_ratios = [
        w["raw_cfa_impulse_ratio"] for f in raw_frames_internal for w in f["_warm_pixels"]
    ]

    star_fwhms_all = [s["fwhm_px"] for f in raw_frames_internal for s in f["_clean_stars"]]
    star_sharps_all = [s["sharpness_3x3"] for f in raw_frames_internal for s in f["_clean_stars"]]
    star_rings_all = [s["ring_r2_ratio"] for f in raw_frames_internal for s in f["_clean_stars"]]
    star_cfa_ratios = [
        s["raw_cfa_impulse_ratio"] for f in raw_frames_internal for s in f["_clean_stars"]
    ]

    stationarity = check_warm_pixel_sensor_stationarity(
        [f["_warm_pixels"] for f in raw_frames_internal]
    )
    warm_counts_per_frame = [
        f["candidate_breakdown"]["warm_pixel_rings_rejected"] for f in raw_frames_internal
    ]
    tot_raw_hot = sum(
        f["candidate_breakdown"]["raw_cfa_hot_impulses_in_minarea5"] for f in raw_frames_internal
    )
    tot_raw_caught = sum(
        f["candidate_breakdown"]["raw_cfa_hot_caught_by_pure_l_filter"] for f in raw_frames_internal
    )
    tot_fp = sum(
        f["candidate_breakdown"]["false_positive_warm_on_real_star"] for f in raw_frames_internal
    )

    mod2_phases = set()
    for d in drift_table:
        pr = int(round(d["dy_px"])) % 2
        pc = int(round(d["dx_px"])) % 2
        mod2_phases.add((pr, pc))

    dxs = [d["dx_px"] for d in drift_table]
    dys = [d["dy_px"] for d in drift_table]
    dths = [d["dtheta_deg"] for d in drift_table]
    radii_non_ref = [d["drift_radius_px"] for d in drift_table if not d["is_reference"]]
    non_ref_rms = [d["rigid_rms_residual_px"] for d in drift_table if not d["is_reference"]]
    non_ref_aff_rms = [d["affine_rms_residual_px"] for d in drift_table if not d["is_reference"]]

    half_diag = math.hypot(ACTIVE_W / 2.0, ACTIVE_H / 2.0)
    dtheta_span_deg = max(dths) - min(dths)
    max_abs_dtheta_deg = max(abs(x) for x in dths)
    corner_rot_arc_span_px = half_diag * math.radians(dtheta_span_deg)
    corner_rot_arc_vs_ref_max_px = half_diag * math.radians(max_abs_dtheta_deg)

    drift_summary = {
        "transform_convention": (
            "Centered at active sensor midpoint (cx=2144.5, cy=1427.5) in unrotated RGGB coordinates "
            "(x=col in [0,4289], y=row in [0,2855]): "
            "[x_frame - cx; y_frame - cy] = R(dtheta) @ [x_ref - cx; y_ref - cy] + [dx; dy]. "
            "To warp a stationary hot pixel at (x_hp, y_hp) in frame i into reference coordinates: "
            "[x_ref - cx; y_ref - cy] = R(-dtheta) @ ([x_hp - cx; y_hp - cy] - [dx; dy])."
        ),
        "reference_frame": winner["filename"],
        "dx_range_px": [round(min(dxs), 4), round(max(dxs), 4)],
        "dy_range_px": [round(min(dys), 4), round(max(dys), 4)],
        "total_dx_span_px": round(max(dxs) - min(dxs), 4),
        "total_dy_span_px": round(max(dys) - min(dys), 4),
        "total_bounding_box_diag_drift_px": round(
            math.hypot(max(dxs) - min(dxs), max(dys) - min(dys)), 4
        ),
        "cumulative_trajectory_path_length_px": round(float(np.sum(step_dists)), 4),
        "drift_radius_vs_ref_min_median_max_px": (
            [
                round(float(min(radii_non_ref)), 4),
                round(float(np.median(radii_non_ref)), 4),
                round(float(max(radii_non_ref)), 4),
            ]
            if radii_non_ref
            else [0.0, 0.0, 0.0]
        ),
        "consecutive_step_dist_min_median_max_px": (
            [
                round(float(min(step_dists)), 4),
                round(float(np.median(step_dists)), 4),
                round(float(max(step_dists)), 4),
            ]
            if step_dists
            else [0.0, 0.0, 0.0]
        ),
        "dtheta_range_deg": [round(min(dths), 5), round(max(dths), 5)],
        "total_dtheta_span_deg": round(dtheta_span_deg, 5),
        "max_abs_dtheta_vs_ref_deg": round(max_abs_dtheta_deg, 5),
        "corner_rotation_displacement_span_px": round(corner_rot_arc_span_px, 3),
        "corner_rotation_displacement_max_vs_ref_px": round(corner_rot_arc_vs_ref_max_px, 3),
        "integer_bayer_2x2_phases_covered": len(mod2_phases),
        "rigid_fit_rms_median_px": round(float(np.median(non_ref_rms)), 4) if non_ref_rms else 0.0,
        "rigid_fit_rms_max_px": round(float(max(non_ref_rms)), 4) if non_ref_rms else 0.0,
        "affine_fit_rms_median_px": round(float(np.median(non_ref_aff_rms)), 4)
        if non_ref_aff_rms
        else 0.0,
        "min_matched_stars_of_top40": min(d["matched_stars_of_top40"] for d in drift_table),
        "max_matched_stars_of_top40": max(d["matched_stars_of_top40"] for d in drift_table),
    }

    warm_pixel_diagnostics = {
        "do_warm_pixels_pass_minarea5_after_mhc_5x5": sum(warm_counts_per_frame) > 0,
        "empirical_finding_explanation": (
            "YES: Unscaled 18C master-dark subtraction on warmer light frames (19-34C) leaves positive "
            "single-CFA-site thermal impulses. Under Malvar-He-Cutler 5x5 demosaicing, a single warm CFA pixel "
            "of amplitude A blooms into a 5-pixel cross (for A > ~40*sigma_bg) or 9-pixel 3x3 patch (for A > ~80*sigma_bg) "
            "surrounded by a negative radius-2 ring (-0.109*A on R/B sites, -0.0625*A on G diagonals), easily passing minarea >= 5!"
        ),
        "warm_pixel_count_per_frame_min_median_max": [
            int(min(warm_counts_per_frame)),
            round(float(np.median(warm_counts_per_frame)), 1),
            int(max(warm_counts_per_frame)),
        ],
        "warm_pixel_count_first_frame_vs_last_frame_cooling": [
            int(warm_counts_per_frame[0]),
            int(warm_counts_per_frame[-1]),
        ],
        "raw_cfa_ground_truth_validation": {
            "raw_cfa_single_pixel_impulses_passing_minarea5_total": tot_raw_hot,
            "caught_by_pure_luminance_filter_total": tot_raw_caught,
            "pure_luminance_recall_on_raw_impulses": (
                round(tot_raw_caught / tot_raw_hot, 5) if tot_raw_hot > 0 else 1.0
            ),
            "false_positives_on_clean_stars": tot_fp,
        },
        "distribution_separation_warm_pixels_vs_clean_stars": {
            "warm_pixel_fwhm_p10_p50_p90_px": (
                [round(float(v), 3) for v in np.percentile(warm_fwhms, [10, 50, 90])]
                if warm_fwhms
                else None
            ),
            "clean_star_fwhm_p01_p50_p99_px": (
                [round(float(v), 3) for v in np.percentile(star_fwhms_all, [1, 50, 99])]
                if star_fwhms_all
                else None
            ),
            "warm_pixel_sharpness_3x3_p10_p50_p90": (
                [round(float(v), 4) for v in np.percentile(warm_sharps, [10, 50, 90])]
                if warm_sharps
                else None
            ),
            "clean_star_sharpness_3x3_p01_p50_p99": (
                [round(float(v), 4) for v in np.percentile(star_sharps_all, [1, 50, 99])]
                if star_sharps_all
                else None
            ),
            "warm_pixel_ring_r2_ratio_p10_p50_p90": (
                [round(float(v), 4) for v in np.percentile(warm_rings, [10, 50, 90])]
                if warm_rings
                else None
            ),
            "clean_star_ring_r2_ratio_p01_p50_p99": (
                [round(float(v), 4) for v in np.percentile(star_rings_all, [1, 50, 99])]
                if star_rings_all
                else None
            ),
            "warm_pixel_raw_cfa_impulse_ratio_median": (
                round(float(np.median(warm_cfa_ratios)), 4) if warm_cfa_ratios else None
            ),
            "clean_star_raw_cfa_impulse_ratio_median": (
                round(float(np.median(star_cfa_ratios)), 4) if star_cfa_ratios else None
            ),
        },
        "sensor_stationarity_verification": stationarity,
    }

    for f in raw_frames_internal:
        f.pop("filepath", None)
        f.pop("_clean_stars", None)
        f.pop("_warm_pixels", None)
        clean_entries.append(f)

    return {
        "set_name": set_name,
        "cr2_inspected_count": len(cr2_files),
        "clean_frame_count": len(clean_entries),
        "rejected_corrupted_count": len(rejected_entries),
        "rejected_corrupted_frames": rejected_entries,
        "reference_selection": selection_rationale,
        "saved_reference_array": win_array_stats,
        "warm_pixel_vs_star_diagnostics": warm_pixel_diagnostics,
        "drift_summary": drift_summary,
        "per_frame_drift_table": drift_table,
        "per_frame_sharpness_and_calibration": clean_entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate linear MHC-demosaiced reference baselines and empirical drift profiles."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("pictures/2020-11-28-deep-sky-ring-andromeda"),
        help="Dataset root containing dark_set, ring_set, andromeda_set.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("logs"),
        help="Output directory for ref_ring.npy, ref_andromeda.npy, and reference_selection.json.",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    workspace_root = Path.cwd()
    base_dir = (workspace_root / args.base_dir).resolve()
    out_dir = (workspace_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dark_dir = base_dir / "dark_set"
    ring_dir = base_dir / "ring_set"
    andro_dir = base_dir / "andromeda_set"
    for d in (dark_dir, ring_dir, andro_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Dataset directory missing: {d}")

    with tempfile.TemporaryDirectory(prefix="astro_ref_gen_") as tmp_dir:
        t_comp0 = time.perf_counter()
        shim = CppRefShim(tmp_dir)
        compile_s = round(time.perf_counter() - t_comp0, 3)

        # 1. Run mandatory MHC 5x5 constant-color conservation & negative excursion unit tests
        unit_test_report = shim.verify_mhc_unit_tests()

        # 2. Build float32 master_dark from clean dark_set frames
        t_dark0 = time.perf_counter()
        dark_files = list_strict_cr2_files(dark_dir)
        n_dark = len(dark_files)
        c_paths = (ctypes.c_char_p * n_dark)(*[os.fsencode(str(p)) for p in dark_files])
        dark_metas_c = (FrameIntegrityMetaC * n_dark)()
        master_dark = np.empty((ACTIVE_H, ACTIVE_W), dtype=np.float32)

        clean_dark_count = shim.lib.build_master_dark(
            n_dark,
            c_paths,
            dark_metas_c,
            master_dark.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        dark_s = round(time.perf_counter() - t_dark0, 3)

        dark_records = [meta_to_dict(dark_files[i].name, dark_metas_c[i]) for i in range(n_dark)]
        clean_dark_records = [r for r in dark_records if r["is_clean"]]
        rejected_dark_records = [r for r in dark_records if not r["is_clean"]]

        if clean_dark_count != 18:
            raise RuntimeError(f"Expected 18 clean darks, got {clean_dark_count}")

        master_dark_summary = {
            "cr2_inspected_count": n_dark,
            "clean_dark_count": clean_dark_count,
            "rejected_corrupted_count": len(rejected_dark_records),
            "rejected_corrupted_frames": rejected_dark_records,
            "left_optical_black_median_range_adu": [
                min(r["left_optical_black_median_adu"] for r in clean_dark_records),
                max(r["left_optical_black_median_adu"] for r in clean_dark_records),
            ],
            "master_dark_shape": list(master_dark.shape),
            "master_dark_dtype": str(master_dark.dtype),
            "master_dark_min_adu": round(float(np.min(master_dark)), 4),
            "master_dark_median_adu": round(float(np.median(master_dark)), 4),
            "master_dark_mean_adu": round(float(np.mean(master_dark, dtype=np.float64)), 4),
            "master_dark_std_adu": round(float(np.std(master_dark, dtype=np.float64)), 4),
            "master_dark_p99_9_adu": round(float(np.percentile(master_dark, 99.9)), 3),
            "master_dark_max_adu": round(float(np.max(master_dark)), 3),
            "hot_sites_gt_500_adu": int(np.count_nonzero(master_dark > 500.0)),
            "hot_sites_gt_2000_adu": int(np.count_nonzero(master_dark > 2000.0)),
            "build_runtime_s": dark_s,
        }

        # 3. Process ring_set (expect 18 clean, 3 rejected)
        t_ring0 = time.perf_counter()
        ring_report = process_light_dataset(
            set_name="ring_set",
            set_dir=ring_dir,
            shim=shim,
            master_dark=master_dark,
            out_npy_path=out_dir / "ref_ring.npy",
        )
        ring_report["runtime_s"] = round(time.perf_counter() - t_ring0, 3)
        if ring_report["clean_frame_count"] != 18:
            raise RuntimeError(
                f"Expected 18 clean frames in ring_set, got {ring_report['clean_frame_count']}"
            )

        # 4. Process andromeda_set (expect 21 clean, 0 rejected)
        t_andro0 = time.perf_counter()
        andro_report = process_light_dataset(
            set_name="andromeda_set",
            set_dir=andro_dir,
            shim=shim,
            master_dark=master_dark,
            out_npy_path=out_dir / "ref_andromeda.npy",
        )
        andro_report["runtime_s"] = round(time.perf_counter() - t_andro0, 3)
        if andro_report["clean_frame_count"] != 21:
            raise RuntimeError(
                f"Expected 21 clean frames in andromeda_set, got {andro_report['clean_frame_count']}"
            )

        total_s = round(time.perf_counter() - t_start, 3)
        full_json = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_runtime_s": total_s,
            "cpp_shim_compile_s": compile_s,
            "libraw_version": shim.libraw_version,
            "sensor_coordinate_system": {
                "active_slice_in_raw": "raw_image[18:2874, 62:4352]",
                "optical_black_slice_in_raw": "raw_image[18:2874, 2:60]",
                "shape_hwc": [ACTIVE_H, ACTIVE_W, 3],
                "cfa_2x2": [["R", "G1"], ["G2", "B"]],
                "exif_flip_handling": "Strictly ignored (physical unrotated sensor coordinates)",
                "zero_clipping": False,
                "demosaicing": "Exact Malvar-He-Cutler (2004) 5x5 linear filter with parity-preserving symmetric border reflection",
                "luminance_formula": "L = 0.25*R + 0.50*G + 0.25*B",
                "psf_fitting": "7-parameter 2D Elliptical Gaussian on 25x25 cutout via Levenberg-Marquardt",
                "corner_definition": "hypot(x - 2144.5, y - 1427.5) > 0.65 * hypot(2145.0, 1428.0) (> 1674.90 px)",
            },
            "mhc_unit_verification": unit_test_report,
            "dark_calibration": master_dark_summary,
            "datasets": {
                "ring_set": ring_report,
                "andromeda_set": andro_report,
            },
        }

        json_path = out_dir / "reference_selection.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(full_json, fh, indent=2)

    print(
        f"Completed in {total_s:.2f}s (compile {compile_s:.2f}s): "
        f"darks={clean_dark_count}/20 clean | "
        f"ring_set winner={ring_report['reference_selection']['winning_filename']} "
        f"(FWHM={ring_report['reference_selection']['winning_metrics']['median_fwhm_px']} px, "
        f"corner={ring_report['reference_selection']['winning_metrics']['corner_fwhm_px']} px, "
        f"ecc={ring_report['reference_selection']['winning_metrics']['median_eccentricity']}, "
        f"stars={ring_report['reference_selection']['winning_metrics']['star_count']}) | "
        f"andromeda_set winner={andro_report['reference_selection']['winning_filename']} "
        f"(FWHM={andro_report['reference_selection']['winning_metrics']['median_fwhm_px']} px, "
        f"corner={andro_report['reference_selection']['winning_metrics']['corner_fwhm_px']} px, "
        f"ecc={andro_report['reference_selection']['winning_metrics']['median_eccentricity']}, "
        f"stars={andro_report['reference_selection']['winning_metrics']['star_count']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

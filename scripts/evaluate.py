#!/usr/bin/env python3
"""Blind Photometric & Astrometric Referee for DSLR Astrophotography Stacking.

Self-contained single-file evaluator (depends ONLY on Python stdlib + numpy; zero
external local .py or .so helper dependencies).

Expected Input Array Contract:
  --stack <stack_output.npy> : float32 ndarray of shape (2856, 4290, 3) [R, G, B]
                               in unrotated physical active sensor coordinates
                               (raw_image[18:2874, 62:4352]).
  --ref   <ref.npy>          : float32 ndarray of shape (2856, 4290, 3) [R, G, B]
                               linear Malvar-He-Cutler 5x5 reference baseline.
  --transforms <transforms.json> : Per-frame registration transforms & weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


EXPECTED_SHAPE = (2856, 4290, 3)
ROI_MARGIN = 64  # Full-overlap interior ROI [64:2792, 64:4226] avoiding <=28 px drift borders
TARGET_STAR_FLUX_F0 = 1000.0
PHOT_TOP_N_STARS = 40
APERTURE_R_PX = 12.0
ANNULUS_RIN_PX = 16.0
ANNULUS_ROUT_PX = 22.0

# Dataset-specific physical constants locked from Phase 1/2 dataset forensics:
# Fingerprinted directly from --ref pixel content (immune to transforms.json spoofing).
DATASET_PROFILES = {
    "ring_set": {
        "min_integrated_frames": 18,
        # Single-exposure median FWHM across all 18 clean ring_set frames (9.37..14.34 px)
        "exposure_ensemble_fwhm_px": 10.9047,
    },
    "andromeda_set": {
        "min_integrated_frames": 20,
        # Single-exposure median FWHM across all 21 clean andromeda_set frames (7.54..9.64 px)
        "exposure_ensemble_fwhm_px": 8.2791,
    },
}


def identify_dataset_from_ref(ref_arr: np.ndarray) -> tuple[str, dict[str, Any]]:
    """Deterministically identify dataset from --ref pixels without trusting transforms.json."""
    sub = np.ascontiguousarray(ref_arr[::64, ::64, :], dtype=np.float32)
    digest = hashlib.sha256(sub.tobytes()).hexdigest()[:16]
    med_r = float(np.median(ref_arr[::16, ::16, 0]))
    med_g = float(np.median(ref_arr[::16, ::16, 1]))
    # ring_set ref (IMG_6074): med_R ~ 72.7, med_G ~ 206.0
    # andromeda_set ref (IMG_6044): med_R ~ 57.8, med_G ~ 162.8
    if 65.0 <= med_r <= 85.0 and 190.0 <= med_g <= 225.0:
        return "ring_set", {**DATASET_PROFILES["ring_set"], "ref_sub_sha256": digest}
    if 45.0 <= med_r <= 65.0 and 145.0 <= med_g <= 180.0:
        return "andromeda_set", {**DATASET_PROFILES["andromeda_set"], "ref_sub_sha256": digest}
    return "unknown_set", {
        "min_integrated_frames": 18,
        "exposure_ensemble_fwhm_px": None,
        "ref_sub_sha256": digest,
    }


def compute_bayer_luminance(rgb: np.ndarray) -> np.ndarray:
    """Canonical RGGB photon-weighted linear luminance: L = 0.25*R + 0.50*G + 0.25*B."""
    return (
        0.25 * rgb[:, :, 0].astype(np.float64)
        + 0.50 * rgb[:, :, 1].astype(np.float64)
        + 0.25 * rgb[:, :, 2].astype(np.float64)
    )


def estimate_detection_background_mesh(
    lum: np.ndarray, cell_size: int = 64
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Estimate smooth background mesh STRICTLY for point-source detection & sky stats.

    Returns:
      bg_map      : (H, W) smooth interpolated background map
      sky_median  : scalar median sky level across star-free/nebula-free cells
      sky_rms     : scalar pixel-to-pixel RMS across star-free/nebula-free cells
      sky_cell_ok : boolean mask (bh, bw) of quiet star-free/nebula-free sky cells
    """
    h, w = lum.shape
    bh = h // cell_size
    bw = w // cell_size
    cropped = lum[: bh * cell_size, : bw * cell_size]
    cells = cropped.reshape(bh, cell_size, bw, cell_size).transpose(0, 2, 1, 3)
    flat_cells = cells.reshape(bh, bw, cell_size * cell_size)

    # Iterative sigma-clipped median/MAD per 64x64 cell
    c_med = np.median(flat_cells, axis=2)
    c_mad = np.median(np.abs(flat_cells - c_med[:, :, None]), axis=2) * 1.4826
    clip_hi = c_med[:, :, None] + 3.0 * c_mad[:, :, None]
    clip_lo = c_med[:, :, None] - 3.0 * c_mad[:, :, None]
    clipped = np.where((flat_cells >= clip_lo) & (flat_cells <= clip_hi), flat_cells, np.nan)
    c_med2 = np.nanmedian(clipped, axis=2)
    c_mad2 = np.nanmedian(np.abs(clipped - c_med2[:, :, None]), axis=2) * 1.4826

    # Restrict sky statistics to interior ROI cells and quietest 40% (excluding M31/M57/bright stars)
    interior_cell_mask = np.zeros((bh, bw), dtype=bool)
    c_margin = max(1, ROI_MARGIN // cell_size)
    interior_cell_mask[c_margin : bh - c_margin, c_margin : bw - c_margin] = True

    interior_mads = c_mad2[interior_cell_mask]
    interior_meds = c_med2[interior_cell_mask]
    mad_cut = float(np.percentile(interior_mads, 45.0))
    med_cut = float(np.percentile(interior_meds, 65.0))
    sky_cell_ok = interior_cell_mask & (c_mad2 <= mad_cut) & (c_med2 <= med_cut)
    if int(np.count_nonzero(sky_cell_ok)) < 16:
        sky_cell_ok = interior_cell_mask & (c_mad2 <= float(np.percentile(interior_mads, 60.0)))

    quiet_pixels = cells[sky_cell_ok].reshape(-1)
    sky_median = float(np.median(quiet_pixels))
    quiet_zero = cells[sky_cell_ok] - np.median(cells[sky_cell_ok], axis=(1, 2), keepdims=True)
    sky_rms = float(np.median(np.abs(quiet_zero)) * 1.4826)

    # 3x3 median filter across cell medians + bilinear interpolation to (H, W)
    med_pad = np.pad(c_med2, 1, mode="edge")
    med_3x3 = np.median(sliding_window_view(med_pad, (3, 3)), axis=(-2, -1))

    gy = np.clip((np.arange(h, dtype=np.float64) - (cell_size * 0.5 - 0.5)) / cell_size, 0.0, bh - 1.000001)
    gx = np.clip((np.arange(w, dtype=np.float64) - (cell_size * 0.5 - 0.5)) / cell_size, 0.0, bw - 1.000001)
    iy = gy.astype(np.int32)
    ix = gx.astype(np.int32)
    fy = (gy - iy)[:, None]
    fx = (gx - ix)[None, :]

    bg_map = (
        (1.0 - fy) * (1.0 - fx) * med_3x3[iy[:, None], ix[None, :]]
        + (1.0 - fy) * fx * med_3x3[iy[:, None], ix[None, :] + 1]
        + fy * (1.0 - fx) * med_3x3[iy[:, None] + 1, ix[None, :]]
        + fy * fx * med_3x3[iy[:, None] + 1, ix[None, :] + 1]
    )
    return bg_map, sky_median, sky_rms, sky_cell_ok


def compute_sky_noise_correlation(
    lum: np.ndarray, sky_cell_ok: np.ndarray, cell_size: int = 64
) -> dict[str, float]:
    """Measure spatial noise correlation in star-free/nebula-free 64x64 sky cells.

    Detects hidden spatial smoothing (Gaussian blur, box blur, median filter, or
    heavy bilinear wash) via 4x4 block-variance inflation and 1-px autocorrelation.
    """
    bh, bw = sky_cell_ok.shape
    cropped = lum[: bh * cell_size, : bw * cell_size]
    cells = cropped.reshape(bh, cell_size, bw, cell_size).transpose(0, 2, 1, 3)
    sky_cells = cells[sky_cell_ok]  # (K, 64, 64)
    # Remove planar gradient inside each 64x64 cell via row/col medians so genuine large-scale
    # optical vignetting never masquerades as pixel-scale smoothing correlation.
    sky_zero = sky_cells - np.median(sky_cells, axis=(1, 2), keepdims=True)
    rms_1x1 = float(np.median(np.abs(sky_zero)) * 1.4826)
    if rms_1x1 <= 1e-12:
        return {"rms_1x1": 0.0, "block_4x4_ratio": 0.0, "rho_1px": 0.0}

    n_sub = cell_size // 4
    b4 = sky_zero.reshape(sky_zero.shape[0], n_sub, 4, n_sub, 4).mean(axis=(2, 4))
    rms_4x4 = float(np.median(np.abs(b4)) * 1.4826)
    block_4x4_ratio = (4.0 * rms_4x4) / rms_1x1

    clipped = np.clip(sky_zero, -3.5 * rms_1x1, 3.5 * rms_1x1)
    var1 = float(np.mean(clipped * clipped))
    cov_x = float(np.mean(clipped[:, :, 1:] * clipped[:, :, :-1]))
    cov_y = float(np.mean(clipped[:, 1:, :] * clipped[:, :-1, :]))
    rho_1px = 0.5 * (cov_x + cov_y) / max(1e-12, var1)
    return {
        "rms_1x1": rms_1x1,
        "block_4x4_ratio": block_4x4_ratio,
        "rho_1px": rho_1px,
    }


def check_dark_clipping_on_sky(
    rgb: np.ndarray, lum: np.ndarray, sky_cell_ok: np.ndarray, cell_size: int = 64
) -> dict[str, Any]:
    """Check if >0.05% (0.0005) of sky background pixels share an identical floor/mode value
    or exhibit an artificial clipped histogram spike on L, R, G, or B.
    """
    bh, bw = sky_cell_ok.shape
    max_identical_frac = 0.0
    worst_channel = "L"
    per_channel = {}

    planes = [
        ("L", lum),
        ("R", rgb[:, :, 0]),
        ("G", rgb[:, :, 1]),
        ("B", rgb[:, :, 2]),
    ]
    for cname, plane in planes:
        cropped = plane[: bh * cell_size, : bw * cell_size]
        cells = cropped.reshape(bh, cell_size, bw, cell_size).transpose(0, 2, 1, 3)
        sky_pix = cells[sky_cell_ok].reshape(-1)
        med = float(np.median(sky_pix))
        mad = float(np.median(np.abs(sky_pix - med)) * 1.4826)
        # Exclude any stray bright star wing > med + 2.5*mad
        clean_sky = sky_pix[sky_pix <= med + 2.5 * max(mad, 1e-9)]
        if clean_sky.size == 0:
            per_channel[cname] = {"max_identical_fraction": 1.0, "clipped_bin_spike_fraction": 1.0}
            max_identical_frac = 1.0
            worst_channel = cname
            continue

        vals, cnts = np.unique(clean_sky, return_counts=True)
        imax = int(np.argmax(cnts))
        ident_frac = float(cnts[imax]) / float(clean_sky.size)

        # Secondary anti-dither check: check narrow 0.002*sigma histogram bins in the lower tail
        # (<= median + 0.05*mad). Theoretical Gaussian peak in width 0.002*sigma is 0.00080;
        # a dithered floor clip >0.5% creates a >0.005 spike in a single micro-bin.
        bin_spike_frac = 0.0
        if mad > 1e-9:
            lo_edge = med - 5.0 * mad
            hi_edge = med + 0.1 * mad
            in_tail = clean_sky[(clean_sky >= lo_edge) & (clean_sky <= hi_edge)]
            if in_tail.size > 0:
                hist, _ = np.histogram(in_tail, bins=2550, range=(lo_edge, hi_edge))
                bin_spike_frac = float(np.max(hist)) / float(clean_sky.size)

        per_channel[cname] = {
            "max_identical_fraction": round(ident_frac, 7),
            "mode_value": round(float(vals[imax]), 5),
            "sky_median": round(med, 5),
            "microbin_lower_tail_max_fraction": round(bin_spike_frac, 7),
        }
        if ident_frac > max_identical_frac:
            max_identical_frac = ident_frac
            worst_channel = cname

    # Hard specification threshold: > 0.05% (0.0005) identical value in sky background
    # (plus micro-bin dithered floor guard > 0.50% = 0.0050).
    dither_spike_max = max(v["microbin_lower_tail_max_fraction"] for v in per_channel.values())
    clipped = (max_identical_frac > 0.0005) or (dither_spike_max > 0.0050)
    return {
        "dark_clipping_triggered": clipped,
        "max_identical_sky_fraction": round(max_identical_frac, 7),
        "threshold_fraction": 0.0005,
        "worst_channel": worst_channel,
        "per_channel": per_channel,
    }


def detect_and_fit_stars(
    lum_res: np.ndarray,
    sky_rms: float,
    saturation_above_bg: float | None = None,
    max_candidates: int = 800,
) -> dict[str, Any]:
    """Detect 5-sigma connected components (minarea >= 5) inside interior ROI and fit
    7-parameter 2D Elliptical Gaussians via damped Levenberg-Marquardt.
    """
    h, w = lum_res.shape
    if sky_rms <= 1e-12:
        return {
            "stars": [],
            "warm_pixel_artifacts": 0,
            "total_5sigma_minarea5": 0,
        }

    thresh = 5.0 * sky_rms
    mask = lum_res > thresh
    mask[:ROI_MARGIN, :] = False
    mask[h - ROI_MARGIN :, :] = False
    mask[:, :ROI_MARGIN] = False
    mask[:, w - ROI_MARGIN :] = False

    ys, xs = np.nonzero(mask)
    visited = np.zeros_like(mask, dtype=bool)
    candidates = []
    total_minarea5 = 0

    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue
        q_y = [int(y0)]
        q_x = [int(x0)]
        visited[y0, x0] = True
        head = 0
        py = int(y0)
        px = int(x0)
        pval = float(lum_res[y0, x0])
        while head < len(q_y):
            cy = q_y[head]
            cx = q_x[head]
            head += 1
            v = float(lum_res[cy, cx])
            if v > pval:
                pval = v
                py = cy
                px = cx
            for ny in (cy - 1, cy, cy + 1):
                for nx in (cx - 1, cx, cx + 1):
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q_y.append(ny)
                        q_x.append(nx)

        area = len(q_y)
        if area < 5:
            continue
        total_minarea5 += 1
        if area > 2500:
            continue
        if saturation_above_bg is not None and pval >= saturation_above_bg:
            continue
        if py < ROI_MARGIN or py >= h - ROI_MARGIN or px < ROI_MARGIN or px >= w - ROI_MARGIN:
            continue
        candidates.append((pval, py, px, area))

    candidates.sort(key=lambda t: t[0], reverse=True)
    candidates = candidates[:max_candidates]

    r_half = 12
    yy, xx = np.mgrid[-r_half : r_half + 1, -r_half : r_half + 1].astype(np.float64)
    cx_img = (w - 1) * 0.5
    cy_img = (h - 1) * 0.5
    corner_radius = 0.65 * math.hypot(w * 0.5, h * 0.5)

    clean_stars = []
    warm_pixel_count = 0

    for pval, py, px, area in candidates:
        p3 = np.maximum(0.0, lum_res[py - 1 : py + 2, px - 1 : px + 2])
        sharpness_3x3 = pval / (float(np.sum(p3)) + 1e-12)
        ax2_mean = 0.25 * float(
            lum_res[py - 2, px] + lum_res[py + 2, px] + lum_res[py, px - 2] + lum_res[py, px + 2]
        )
        ring_r2_ratio = ax2_mean / max(pval, 1e-12)

        # Reject MHC 5x5 single-pixel warm/hot impulses (sharpness > 0.36 or negative r=2 lobe)
        if sharpness_3x3 > 0.36 or ring_r2_ratio < -0.02:
            warm_pixel_count += 1
            continue

        cut = lum_res[py - r_half : py + r_half + 1, px - r_half : px + r_half + 1].astype(np.float64)
        b0 = float(np.percentile(cut, 15.0))
        amp0 = max(float(cut[r_half, r_half]) - b0, sky_rms)
        params = np.array([b0, amp0, 0.0, 0.0, math.log(3.5), math.log(3.5), 0.0], dtype=np.float64)
        lam = 1e-2
        fit_ok = True

        for _ in range(16):
            b, amp, x0, y0, lsx, lsy, th = params
            sx = math.exp(min(max(lsx, math.log(0.4)), math.log(16.0)))
            sy = math.exp(min(max(lsy, math.log(0.4)), math.log(16.0)))
            ct = math.cos(th)
            st = math.sin(th)
            dx = xx - x0
            dy = yy - y0
            u = ct * dx + st * dy
            v = -st * dx + ct * dy
            q_arg = 0.5 * ((u / sx) ** 2 + (v / sy) ** 2)
            ee = np.exp(-np.clip(q_arg, 0.0, 50.0))
            err = (cut - (b + amp * ee)).ravel()

            jac = np.empty((cut.size, 7), dtype=np.float64)
            ae = (amp * ee).ravel()
            u_r = u.ravel()
            v_r = v.ravel()
            inv_sx2 = 1.0 / (sx * sx)
            inv_sy2 = 1.0 / (sy * sy)
            jac[:, 0] = 1.0
            jac[:, 1] = ee.ravel()
            jac[:, 2] = -ae * (-ct * u_r * inv_sx2 + st * v_r * inv_sy2)
            jac[:, 3] = -ae * (-st * u_r * inv_sx2 - ct * v_r * inv_sy2)
            jac[:, 4] = ae * (u_r * u_r * inv_sx2)
            jac[:, 5] = ae * (v_r * v_r * inv_sy2)
            jac[:, 6] = -ae * (u_r * v_r * (inv_sx2 - inv_sy2))

            jtj = jac.T @ jac
            diag_damping = np.maximum(np.diag(jtj), 1e-12)
            try:
                dp = np.linalg.solve(jtj + lam * np.diag(diag_damping), jac.T @ err)
            except np.linalg.LinAlgError:
                fit_ok = False
                break

            cand_p = params + dp
            if abs(cand_p[2]) > 6.5 or abs(cand_p[3]) > 6.5:
                fit_ok = False
                break
            params = cand_p
            step_norm = max(
                abs(float(dp[0])) / max(sky_rms, 1e-12),
                abs(float(dp[1])) / max(sky_rms, 1e-12),
                float(np.max(np.abs(dp[2:]))),
            )
            if step_norm < 1e-5:
                break

        if not fit_ok:
            continue

        b_f, amp_f, x0_f, y0_f, lsx_f, lsy_f, _ = params
        sx_f = math.exp(min(max(lsx_f, math.log(0.4)), math.log(16.0)))
        sy_f = math.exp(min(max(lsy_f, math.log(0.4)), math.log(16.0)))
        s_maj = max(sx_f, sy_f)
        s_min = min(sx_f, sy_f)
        fwhm_geom = 2.354820045 * math.sqrt(s_maj * s_min)
        fwhm_maj = 2.354820045 * s_maj
        fwhm_min = 2.354820045 * s_min

        if fwhm_geom < 2.0:
            warm_pixel_count += 1
            continue
        if amp_f < 3.5 * sky_rms or fwhm_geom > 22.0 or fwhm_maj > 28.0:
            continue

        # Reduced chi-squared goodness-of-fit check to reject blended non-PSF nebular knots,
        # incorporating an amplitude-proportional Moffat/coma outer-wing floor so high-SNR
        # stacked stars are not falsely rejected when sky_rms drops as 1/sqrt(N).
        ct_f = math.cos(params[6])
        st_f = math.sin(params[6])
        u_f = ct_f * (xx - x0_f) + st_f * (yy - y0_f)
        v_f = -st_f * (xx - x0_f) + ct_f * (yy - y0_f)
        mod_f = b_f + amp_f * np.exp(-np.clip(0.5 * ((u_f / sx_f) ** 2 + (v_f / sy_f) ** 2), 0.0, 50.0))
        pix_var = (
            sky_rms * sky_rms
            + (0.08 * np.maximum(0.0, mod_f - b_f)) ** 2
            + (0.03 * max(0.0, amp_f)) ** 2
        )
        chi2_red = float(np.sum((cut - mod_f) ** 2 / pix_var)) / max(1, cut.size - 7)
        if chi2_red > 5.0:
            continue

        xc = float(px + x0_f)
        yc = float(py + y0_f)
        ecc = math.sqrt(max(0.0, 1.0 - (s_min / s_maj) ** 2))
        is_corner = math.hypot(xc - cx_img, yc - cy_img) > corner_radius

        clean_stars.append(
            {
                "x": xc,
                "y": yc,
                "py": py,
                "px": px,
                "area": area,
                "amp": float(amp_f),
                "flux_gauss": float(2.0 * math.pi * amp_f * s_maj * s_min),
                "fwhm_px": float(fwhm_geom),
                "fwhm_major_px": float(fwhm_maj),
                "fwhm_minor_px": float(fwhm_min),
                "eccentricity": float(ecc),
                "is_corner": bool(is_corner),
            }
        )

    return {
        "stars": clean_stars,
        "warm_pixel_artifacts": warm_pixel_count,
        "total_5sigma_minarea5": total_minarea5,
    }


def measure_aperture_flux(lum: np.ndarray, x: float, y: float) -> float:
    """Linear circular aperture flux (R=12 px) minus local sky annulus median (R=16..22 px)."""
    ix = int(round(x))
    iy = int(round(y))
    r_pad = 24
    h, w = lum.shape
    if iy < r_pad or iy >= h - r_pad or ix < r_pad or ix >= w - r_pad:
        return float("nan")
    patch = lum[iy - r_pad : iy + r_pad + 1, ix - r_pad : ix + r_pad + 1]
    yy, xx = np.mgrid[-r_pad : r_pad + 1, -r_pad : r_pad + 1]
    rr2 = (xx - (x - ix)) ** 2 + (yy - (y - iy)) ** 2
    ap_pix = patch[rr2 <= APERTURE_R_PX * APERTURE_R_PX]
    ann_pix = patch[(rr2 >= ANNULUS_RIN_PX * ANNULUS_RIN_PX) & (rr2 <= ANNULUS_ROUT_PX * ANNULUS_ROUT_PX)]
    bg_ann = float(np.median(ann_pix))
    return float(np.sum(ap_pix - bg_ann))


def select_isolated_photometric_stars(
    ref_stars: list[dict[str, Any]], top_n: int = PHOT_TOP_N_STARS, min_sep_px: float = 32.0
) -> list[dict[str, Any]]:
    """Select top N bright, well-isolated, non-saturated reference stars for flux calibration."""
    ordered = sorted(ref_stars, key=lambda s: s["flux_gauss"], reverse=True)
    selected: list[dict[str, Any]] = []
    for s in ordered:
        if s["x"] < ROI_MARGIN + 26 or s["x"] >= EXPECTED_SHAPE[1] - (ROI_MARGIN + 26):
            continue
        if s["y"] < ROI_MARGIN + 26 or s["y"] >= EXPECTED_SHAPE[0] - (ROI_MARGIN + 26):
            continue
        too_close = any(
            math.hypot(s["x"] - o["x"], s["y"] - o["y"]) < min_sep_px for o in selected
        )
        if not too_close:
            selected.append(s)
            if len(selected) >= top_n:
                break
    return selected


def compute_extended_flux_preservation(
    lum_stack_norm: np.ndarray,
    lum_ref_norm: np.ndarray,
    stack_stars: list[dict[str, Any]],
    ref_stars: list[dict[str, Any]],
) -> dict[str, float]:
    """Normalized cross-correlation of star-masked, 16x16 binned medium-scale structure.

    Masks only compact point-source stars (preserving M57 Ring Nebula and M31 Andromeda
    extended emission!), removes global 1st-degree planar sky tilt so legitimate planar
    sky-gradient equalization is allowed, and evaluates 16x16 binned structure preservation.
    """
    h, w = lum_ref_norm.shape
    star_mask = np.zeros((h, w), dtype=bool)

    for catalog in (ref_stars, stack_stars):
        for s in catalog:
            r_m = int(math.ceil(max(12.0, 1.4 * s["fwhm_px"])))
            ix = int(round(s["x"]))
            iy = int(round(s["y"]))
            y0 = max(0, iy - r_m)
            y1 = min(h, iy + r_m + 1)
            x0 = max(0, ix - r_m)
            x1 = min(w, ix + r_m + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            star_mask[y0:y1, x0:x1] |= (xx - ix) ** 2 + (yy - iy) ** 2 <= r_m * r_m

    def _binned_plane_detrended_ncc(block_px: int, y_end_aligned: int, x_end_aligned: int) -> tuple[float, float, int]:
        y0_b, x0_b = 64, 64
        bh_b = (y_end_aligned - y0_b) // block_px
        bw_b = (x_end_aligned - x0_b) // block_px
        n_pix = block_px * block_px

        s_r = np.where(star_mask[y0_b:y_end_aligned, x0_b:x_end_aligned], np.nan, lum_stack_norm[y0_b:y_end_aligned, x0_b:x_end_aligned])
        r_r = np.where(star_mask[y0_b:y_end_aligned, x0_b:x_end_aligned], np.nan, lum_ref_norm[y0_b:y_end_aligned, x0_b:x_end_aligned])

        sb = s_r.reshape(bh_b, block_px, bw_b, block_px).transpose(0, 2, 1, 3).reshape(bh_b, bw_b, n_pix)
        rb = r_r.reshape(bh_b, block_px, bw_b, block_px).transpose(0, 2, 1, 3).reshape(bh_b, bw_b, n_pix)

        v_cnt = np.sum(~np.isnan(rb), axis=2)
        v_blk = v_cnt >= (n_pix // 2)

        s_sf = np.where(v_blk[:, :, None], sb, 0.0)
        r_sf = np.where(v_blk[:, :, None], rb, 0.0)
        s_mn = np.nanmean(s_sf, axis=2)
        r_mn = np.nanmean(r_sf, axis=2)

        by_i, bx_i = np.mgrid[0:bh_b, 0:bw_b].astype(np.float64)
        vy_i = (by_i[v_blk] / bh_b) - 0.5
        vx_i = (bx_i[v_blk] / bw_b) - 0.5
        des = np.column_stack([np.ones_like(vx_i), vx_i, vy_i])

        sv_i = s_mn[v_blk]
        rv_i = r_mn[v_blk]
        cs, _, _, _ = np.linalg.lstsq(des, sv_i, rcond=None)
        cr, _, _, _ = np.linalg.lstsq(des, rv_i, rcond=None)
        sm = sv_i - des @ cs
        rm = rv_i - des @ cr

        den = math.sqrt(float(np.sum(sm * sm) * np.sum(rm * rm)))
        ncc_val = float(np.clip(float(np.sum(sm * rm)) / max(1e-12, den), -1.0, 1.0))
        rms_rat = math.sqrt(float(np.sum(sm * sm)) / max(1e-12, float(np.sum(rm * rm))))
        return ncc_val, rms_rat, int(np.count_nonzero(v_blk))

    # 16x16 medium-scale blocks ([64:2784, 64:4224] -> 170 x 260) AND
    # 64x64 meso-scale blocks ([64:2752, 64:4224] -> 42 x 65) to catch 64x64 SEP mesh erasure
    ncc_16, amp_16, n_blk_16 = _binned_plane_detrended_ncc(16, 2784, 4224)
    ncc_64, amp_64, n_blk_64 = _binned_plane_detrended_ncc(64, 2752, 4224)
    ncc_combined = min(ncc_16, ncc_64)
    return {
        "ncc": float(ncc_combined),
        "ncc_16x16": round(ncc_16, 6),
        "ncc_64x64": round(ncc_64, 6),
        "valid_16x16_blocks": n_blk_16,
        "valid_64x64_blocks": n_blk_64,
        "structure_rms_ratio_16x16": round(amp_16, 4),
        "structure_rms_ratio_64x64": round(amp_64, 4),
    }


def validate_transforms_json(
    transforms_path: str | None,
    min_required_frames: int,
    single_frame_baseline: bool,
) -> dict[str, Any]:
    """Inspect transforms.json for frame-drop cheating and stationary hot-pixel alignment."""
    if single_frame_baseline:
        return {
            "mode": "single_frame_baseline",
            "integrated_frame_count": 1,
            "effective_sample_size": 1.0,
            "hot_pixel_alignment_triggered": False,
            "frame_drop_triggered": False,
            "reasons": [],
        }

    reasons: list[str] = []
    if not transforms_path or not Path(transforms_path).is_file():
        return {
            "mode": "stack",
            "integrated_frame_count": 0,
            "effective_sample_size": 0.0,
            "hot_pixel_alignment_triggered": True,
            "frame_drop_triggered": True,
            "reasons": ["missing_transforms_json"],
        }

    try:
        with open(transforms_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        return {
            "mode": "stack",
            "integrated_frame_count": 0,
            "effective_sample_size": 0.0,
            "hot_pixel_alignment_triggered": True,
            "frame_drop_triggered": True,
            "reasons": [f"invalid_transforms_json: {exc}"],
        }

    frames_list = payload.get("frames", payload) if isinstance(payload, dict) else payload
    if not isinstance(frames_list, list) or len(frames_list) == 0:
        return {
            "mode": "stack",
            "integrated_frame_count": 0,
            "effective_sample_size": 0.0,
            "hot_pixel_alignment_triggered": True,
            "frame_drop_triggered": True,
            "reasons": ["empty_frames_in_transforms_json"],
        }

    integrated_entries = []
    seen_filenames = set()
    duplicate_filenames = False
    for entry in frames_list:
        if not isinstance(entry, dict):
            continue
        is_int = bool(entry.get("integrated", True))
        w = float(entry.get("weight", 1.0))
        if not is_int or w <= 0.0:
            continue
        fname = str(entry.get("filename", entry.get("frame", f"idx_{len(integrated_entries)}")))
        if fname in seen_filenames:
            duplicate_filenames = True
        seen_filenames.add(fname)
        dx = float(entry.get("dx", entry.get("dx_px", 0.0)))
        dy = float(entry.get("dy", entry.get("dy_px", 0.0)))
        if "dtheta_deg" in entry:
            dtheta_deg = float(entry["dtheta_deg"])
        elif "dtheta_rad" in entry:
            dtheta_deg = math.degrees(float(entry["dtheta_rad"]))
        else:
            dtheta_deg = float(entry.get("dtheta", 0.0))
        is_ref = bool(entry.get("is_reference", False))
        integrated_entries.append(
            {
                "filename": fname,
                "weight": w,
                "dx": dx,
                "dy": dy,
                "dr": math.hypot(dx, dy),
                "dtheta_deg": dtheta_deg,
                "is_reference": is_ref,
            }
        )

    n_int = len(integrated_entries)
    weights = np.array([e["weight"] for e in integrated_entries], dtype=np.float64)
    if n_int > 0 and float(np.sum(weights**2)) > 0.0:
        n_eff = float((np.sum(weights) ** 2) / np.sum(weights**2))
    else:
        n_eff = 0.0

    frame_drop_triggered = False
    if duplicate_filenames:
        frame_drop_triggered = True
        reasons.append("duplicate_frame_filenames_in_transforms")
    if n_int < min_required_frames:
        frame_drop_triggered = True
        reasons.append(f"integrated_frame_count={n_int} < min_required={min_required_frames}")
    if n_eff < 0.65 * min_required_frames:
        frame_drop_triggered = True
        reasons.append(
            f"effective_kish_sample_size={n_eff:.2f} < {0.65 * min_required_frames:.2f} (extreme weight concentration)"
        )

    # Identify non-reference frames (if none explicitly marked is_reference, exclude the 1 closest to (0,0))
    ref_marked = [e for e in integrated_entries if e["is_reference"]]
    if len(ref_marked) == 1:
        non_ref = [e for e in integrated_entries if not e["is_reference"]]
    elif n_int > 1:
        idx_min = int(np.argmin([e["dr"] for e in integrated_entries]))
        non_ref = [e for idx, e in enumerate(integrated_entries) if idx != idx_min]
    else:
        non_ref = []

    hot_pixel_triggered = False
    if len(non_ref) == 0:
        hot_pixel_triggered = True
        reasons.append("no_non_reference_frames_present")
    else:
        drs = np.array([e["dr"] for e in non_ref], dtype=np.float64)
        dxs = np.array([e["dx"] for e in integrated_entries], dtype=np.float64)
        dys = np.array([e["dy"] for e in integrated_entries], dtype=np.float64)
        dthetas = np.array([abs(e["dtheta_deg"]) for e in non_ref], dtype=np.float64)
        near_zero_frac = float(np.mean((drs < 0.75) & (dthetas < 0.005)))
        median_dr = float(np.median(drs))
        bbox_diag = math.hypot(float(np.max(dxs) - np.min(dxs)), float(np.max(dys) - np.min(dys)))

        if median_dr < 3.0 or near_zero_frac > 0.20 or bbox_diag < 15.0:
            hot_pixel_triggered = True
            reasons.append(
                f"hot_pixel_or_zero_shift_transforms (median_dr={median_dr:.3f}px, "
                f"near_zero_frac={near_zero_frac:.2%}, bbox_diag={bbox_diag:.3f}px)"
            )

    return {
        "mode": "stack",
        "integrated_frame_count": n_int,
        "min_required_frames": min_required_frames,
        "effective_sample_size": round(n_eff, 3),
        "median_nonref_shift_px": round(float(np.median([e["dr"] for e in non_ref])), 4) if non_ref else 0.0,
        "shift_bbox_diag_px": round(
            math.hypot(
                float(max(e["dx"] for e in integrated_entries) - min(e["dx"] for e in integrated_entries)),
                float(max(e["dy"] for e in integrated_entries) - min(e["dy"] for e in integrated_entries)),
            ),
            4,
        )
        if integrated_entries
        else 0.0,
        "hot_pixel_alignment_triggered": hot_pixel_triggered,
        "frame_drop_triggered": frame_drop_triggered,
        "reasons": reasons,
    }


def evaluate_stack(
    stack_path: str,
    ref_path: str,
    transforms_path: str | None,
    single_frame_baseline: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    cheat_reasons: list[str] = []

    ref_arr = np.load(ref_path)
    stack_arr = np.load(stack_path)

    if ref_arr.shape != EXPECTED_SHAPE or ref_arr.dtype != np.float32:
        raise ValueError(f"--ref must have shape {EXPECTED_SHAPE} and dtype float32; got {ref_arr.shape}, {ref_arr.dtype}")

    dataset_id, ds_profile = identify_dataset_from_ref(ref_arr)
    min_req_frames = int(ds_profile["min_integrated_frames"])

    # Gate 0: Shape, dtype, NaN/Inf, and RGB channel non-degeneracy
    if stack_arr.shape != EXPECTED_SHAPE or not np.issubdtype(stack_arr.dtype, np.floating):
        return {
            "dataset": dataset_id,
            "score": float("-inf"),
            "composite_score": float("-inf"),
            "cheating_detected": True,
            "cheat_reasons": [f"invalid_stack_shape_or_dtype: {stack_arr.shape}, {stack_arr.dtype}"],
            "runtime_s": round(time.perf_counter() - t0, 4),
        }

    if not np.all(np.isfinite(stack_arr)):
        return {
            "dataset": dataset_id,
            "score": float("-inf"),
            "composite_score": float("-inf"),
            "cheating_detected": True,
            "cheat_reasons": ["nan_or_inf_in_stack"],
            "runtime_s": round(time.perf_counter() - t0, 4),
        }

    roi_stack = stack_arr[ROI_MARGIN : EXPECTED_SHAPE[0] - ROI_MARGIN, ROI_MARGIN : EXPECTED_SHAPE[1] - ROI_MARGIN]
    ch_stds = [float(np.std(roi_stack[:, :, c])) for c in range(3)]
    if min(ch_stds) <= 1e-6:
        cheat_reasons.append(f"zero_or_constant_channel_collapse (stds={ch_stds})")
    else:
        for c1, c2 in ((0, 1), (1, 2), (0, 2)):
            diff_std = float(np.std(roi_stack[:, :, c1] - roi_stack[:, :, c2]))
            if diff_std <= 1e-4 * min(ch_stds[c1], ch_stds[c2]):
                cheat_reasons.append(f"duplicate_rgb_channels_{c1}_{c2}")

    # Validate transforms.json (Hot-Pixel Alignment & Frame-Count Gate)
    trans_report = validate_transforms_json(transforms_path, min_req_frames, single_frame_baseline)
    if trans_report["frame_drop_triggered"] or trans_report["hot_pixel_alignment_triggered"]:
        cheat_reasons.extend(trans_report["reasons"])

    # If catastrophic zero/constant collapse already happened, return immediately
    if any("zero_or_constant" in r for r in cheat_reasons):
        return {
            "dataset": dataset_id,
            "score": float("-inf"),
            "composite_score": float("-inf"),
            "cheating_detected": True,
            "cheat_reasons": cheat_reasons,
            "transforms_validation": trans_report,
            "runtime_s": round(time.perf_counter() - t0, 4),
        }

    lum_ref = compute_bayer_luminance(ref_arr)
    lum_stack = compute_bayer_luminance(stack_arr)

    bg_ref_map, sky_med_ref, sky_rms_ref, sky_cells_ref = estimate_detection_background_mesh(lum_ref)
    bg_stack_map, sky_med_stack, sky_rms_stack, sky_cells_stack = estimate_detection_background_mesh(lum_stack)

    if sky_rms_stack <= 1e-8:
        cheat_reasons.append("zero_sky_rms_collapse")
        return {
            "dataset": dataset_id,
            "score": float("-inf"),
            "composite_score": float("-inf"),
            "cheating_detected": True,
            "cheat_reasons": cheat_reasons,
            "transforms_validation": trans_report,
            "runtime_s": round(time.perf_counter() - t0, 4),
        }

    # Gate 1: Dark-Clipping Check (>0.05% identical floor/sky value on L, R, G, or B)
    clip_report = check_dark_clipping_on_sky(stack_arr, lum_stack, sky_cells_stack)
    if clip_report["dark_clipping_triggered"]:
        cheat_reasons.append(
            f"dark_floor_clipping_detected (max_identical_sky_fraction="
            f"{clip_report['max_identical_sky_fraction']:.6f} > 0.0005 on {clip_report['worst_channel']})"
        )

    # Detect initial reference stars on raw ADU to select top-N isolated photometric anchors
    ref_det_init = detect_and_fit_stars(lum_ref - bg_ref_map, sky_rms_ref, saturation_above_bg=10500.0)
    phot_ref_stars = select_isolated_photometric_stars(ref_det_init["stars"], top_n=PHOT_TOP_N_STARS)

    # First-pass detection on stack to cross-match photometric reference stars & estimate scale
    stack_det_raw = detect_and_fit_stars(lum_stack - bg_stack_map, sky_rms_stack, saturation_above_bg=None)
    stack_stars_raw = stack_det_raw["stars"]

    # Photometric Normalization (Anti-Zero-Multiplier & Anti-Nonlinear-Stretch Gate)
    ref_ap_fluxes_all = []
    for rs in phot_ref_stars:
        f_r = measure_aperture_flux(lum_ref, rs["x"], rs["y"])
        if math.isfinite(f_r) and f_r > 0.0:
            ref_ap_fluxes_all.append((rs, f_r))

    ref_median_flux_adu = float(np.median([f for _, f in ref_ap_fluxes_all]))
    scale_ref = TARGET_STAR_FLUX_F0 / ref_median_flux_adu
    offset_ref = -sky_med_ref * scale_ref
    lum_ref_norm = lum_ref * scale_ref + offset_ref
    sat_norm = 10500.0 * scale_ref

    # Run canonical reference star detection on normalized reference scale (identical to stack_det)
    ref_det = detect_and_fit_stars(
        (lum_ref - bg_ref_map) * scale_ref,
        sky_rms_ref * scale_ref,
        saturation_above_bg=sat_norm,
    )
    ref_stars = ref_det["stars"]

    matched_flux_pairs = []
    matched_fwhm_pairs = []
    if stack_stars_raw:
        s_xy = np.array([[s["x"], s["y"]] for s in stack_stars_raw], dtype=np.float64)
        for rs, f_r in ref_ap_fluxes_all:
            dists = np.hypot(s_xy[:, 0] - rs["x"], s_xy[:, 1] - rs["y"])
            j = int(np.argmin(dists))
            if dists[j] <= 4.0:
                ms = stack_stars_raw[j]
                f_s = measure_aperture_flux(lum_stack, ms["x"], ms["y"])
                if math.isfinite(f_s) and f_s > 0.0:
                    matched_flux_pairs.append((f_r, f_s))
                    matched_fwhm_pairs.append((rs["fwhm_px"], ms["fwhm_px"]))

    if len(matched_flux_pairs) < 20:
        cheat_reasons.append(
            f"photometric_star_match_failure (matched={len(matched_flux_pairs)} < 20 of top {len(ref_ap_fluxes_all)})"
        )
        return {
            "dataset": dataset_id,
            "score": float("-inf"),
            "composite_score": float("-inf"),
            "cheating_detected": True,
            "cheat_reasons": cheat_reasons,
            "transforms_validation": trans_report,
            "dark_clipping_check": clip_report,
            "runtime_s": round(time.perf_counter() - t0, 4),
        }

    f_ref_vec = np.array([p[0] for p in matched_flux_pairs], dtype=np.float64)
    f_stk_vec = np.array([p[1] for p in matched_flux_pairs], dtype=np.float64)
    flux_ratios = f_stk_vec / f_ref_vec
    median_flux_ratio = float(np.median(flux_ratios))

    if median_flux_ratio <= 0.0 or not math.isfinite(median_flux_ratio):
        cheat_reasons.append("non_positive_photometric_flux_scale")
        return {
            "dataset": dataset_id,
            "score": float("-inf"),
            "composite_score": float("-inf"),
            "cheating_detected": True,
            "cheat_reasons": cheat_reasons,
            "runtime_s": round(time.perf_counter() - t0, 4),
        }

    # Check linearity across bright-to-medium photometric stars (including bright vs faint half slope)
    if float(np.std(f_ref_vec)) > 0.0 and float(np.std(f_stk_vec)) > 0.0:
        flux_corr = float(np.corrcoef(f_ref_vec, f_stk_vec)[0, 1])
    else:
        flux_corr = 0.0
    ratio_rel_mad = float(np.median(np.abs(flux_ratios - median_flux_ratio)) * 1.4826 / median_flux_ratio)
    order_by_ref = np.argsort(f_ref_vec)
    half_n = len(order_by_ref) // 2
    faint_med_ratio = float(np.median(flux_ratios[order_by_ref[:half_n]]))
    bright_med_ratio = float(np.median(flux_ratios[order_by_ref[half_n:]]))
    bright_faint_asym = abs(bright_med_ratio / max(1e-12, faint_med_ratio) - 1.0)
    if flux_corr < 0.96 or ratio_rel_mad > 0.15 or bright_faint_asym > 0.10:
        cheat_reasons.append(
            f"nonlinear_photometric_stretch_detected (r={flux_corr:.4f}, rel_mad={ratio_rel_mad:.4f}, "
            f"bright_faint_asym={bright_faint_asym:.4f})"
        )

    scale_stack = scale_ref / median_flux_ratio
    offset_stack = -sky_med_stack * scale_stack
    lum_stack_norm = lum_stack * scale_stack + offset_stack

    # Reject saturated stars consistently on normalized scale (10500 ADU * scale_ref)
    stack_det = detect_and_fit_stars(
        (lum_stack - bg_stack_map) * scale_stack,
        sky_rms_stack * scale_stack,
        saturation_above_bg=sat_norm,
    )
    stack_stars = stack_det["stars"]

    if len(stack_stars) < 20:
        cheat_reasons.append(f"insufficient_valid_stars_detected ({len(stack_stars)} < 20)")
        return {
            "dataset": dataset_id,
            "score": float("-inf"),
            "composite_score": float("-inf"),
            "cheating_detected": True,
            "cheat_reasons": cheat_reasons,
            "runtime_s": round(time.perf_counter() - t0, 4),
        }

    # Core Metrics on photometrically normalized linear stack & reference
    ref_median_fwhm = float(np.median([s["fwhm_px"] for s in ref_stars]))
    ref_corner_stars = [s["fwhm_px"] for s in ref_stars if s["is_corner"]]
    ref_corner_fwhm = float(np.median(ref_corner_stars)) if ref_corner_stars else ref_median_fwhm
    ref_median_ecc = float(np.median([s["eccentricity"] for s in ref_stars]))
    ref_norm_bg_rms = float(sky_rms_ref * scale_ref)
    ref_star_count = len(ref_stars)

    median_fwhm_px = float(np.median([s["fwhm_px"] for s in stack_stars]))
    corner_stars = [s["fwhm_px"] for s in stack_stars if s["is_corner"]]
    corner_fwhm_px = float(np.median(corner_stars)) if len(corner_stars) >= 3 else float(
        np.percentile([s["fwhm_px"] for s in stack_stars], 85.0)
    )
    median_eccentricity = float(np.median([s["eccentricity"] for s in stack_stars]))
    normalized_bg_rms = float(sky_rms_stack * scale_stack)
    matched_star_count = len(stack_stars)

    ext_report = compute_extended_flux_preservation(
        lum_stack_norm, lum_ref_norm, stack_stars, ref_stars
    )
    extended_flux_preservation = float(ext_report["ncc"])

    # Gate 2: Blurring & Spatial-Smoothing Check
    noise_corr_ref = compute_sky_noise_correlation(lum_ref_norm, sky_cells_ref)
    noise_corr_stack = compute_sky_noise_correlation(lum_stack_norm, sky_cells_stack)

    # (2a) Sky spatial-smoothing noise correlation check (catches Gaussian/box/median/bilinear smoothing)
    if (
        noise_corr_stack["block_4x4_ratio"] > 1.25 * noise_corr_ref["block_4x4_ratio"]
        or noise_corr_stack["rho_1px"] > noise_corr_ref["rho_1px"] + 0.18
    ):
        cheat_reasons.append(
            f"spatial_smoothing_noise_correlation_detected (block_4x4_ratio="
            f"{noise_corr_stack['block_4x4_ratio']:.4f} vs ref {noise_corr_ref['block_4x4_ratio']:.4f}, "
            f"rho_1px={noise_corr_stack['rho_1px']:.4f} vs ref {noise_corr_ref['rho_1px']:.4f})"
        )

    # (2b) Stellar FWHM degradation check (>10% over reference / exposure ensemble)
    matched_ref_fwhm = float(np.median([p[0] for p in matched_fwhm_pairs]))
    matched_stack_fwhm = float(np.median([p[1] for p in matched_fwhm_pairs]))
    if single_frame_baseline:
        fwhm_ceiling = 1.10 * ref_median_fwhm
        if median_fwhm_px > fwhm_ceiling:
            cheat_reasons.append(
                f"fwhm_blur_degradation (median_fwhm_px={median_fwhm_px:.4f} > 1.10*ref={fwhm_ceiling:.4f})"
            )
    else:
        ens_base = ds_profile["exposure_ensemble_fwhm_px"] or ref_median_fwhm
        base_fwhm = max(ref_median_fwhm, float(ens_base))
        ens_factor = base_fwhm / max(1e-6, ref_median_fwhm)
        matched_base_fwhm = max(matched_ref_fwhm * ens_factor, base_fwhm)
        if matched_stack_fwhm > 1.10 * matched_base_fwhm or median_fwhm_px > 1.10 * base_fwhm:
            cheat_reasons.append(
                f"fwhm_blur_degradation (matched_fwhm={matched_stack_fwhm:.4f}px [ceil={1.10 * matched_base_fwhm:.4f}px], "
                f"median_fwhm={median_fwhm_px:.4f}px [ceil={1.10 * base_fwhm:.4f}px])"
            )

    # Gate 3: Pixel-level Multi-Frame Noise Reduction Floor & Ceiling Consistency (Guardrail 3)
    if not single_frame_baseline and trans_report["integrated_frame_count"] >= min_req_frames:
        n_int = trans_report["integrated_frame_count"]
        n_eff = max(1.0, float(trans_report["effective_sample_size"]))
        theo_floor = 0.50 * (ref_norm_bg_rms / math.sqrt(n_int))
        upper_ceil = min(0.60, 1.85 / math.sqrt(n_eff)) * ref_norm_bg_rms
        if normalized_bg_rms > upper_ceil:
            cheat_reasons.append(
                f"insufficient_empirical_noise_reduction (normalized_bg_rms={normalized_bg_rms:.4f} > "
                f"ceiling={upper_ceil:.4f}; transforms.json claims {n_int} frames, n_eff={n_eff:.2f})"
            )
        elif normalized_bg_rms < theo_floor:
            cheat_reasons.append(
                f"unphysical_sub_poisson_noise_floor (normalized_bg_rms={normalized_bg_rms:.4f} < "
                f"0.50*ref/sqrt({n_int})={theo_floor:.4f})"
            )

    # Gate 4: Extended Flux Erasure Check (Guardrail 4: prevents fine SEP mesh subtraction)
    if extended_flux_preservation < 0.85 or ext_report["structure_rms_ratio_64x64"] < 0.70:
        cheat_reasons.append(
            f"extended_flux_erasure_detected (extended_flux_preservation={extended_flux_preservation:.4f} < 0.85, "
            f"rms_ratio_64x64={ext_report['structure_rms_ratio_64x64']:.4f})"
        )

    cheating_detected = len(cheat_reasons) > 0

    ref_fwhm_eff = 0.65 * ref_median_fwhm + 0.35 * ref_corner_fwhm
    stack_fwhm_eff = 0.65 * median_fwhm_px + 0.35 * corner_fwhm_px

    term_bg = 25.0 * math.log2(ref_norm_bg_rms / normalized_bg_rms)
    term_fwhm = 35.0 * math.log2(ref_fwhm_eff / stack_fwhm_eff)
    term_stars = 10.0 * math.log2(matched_star_count / max(1, ref_star_count))
    term_ecc = 15.0 * (ref_median_ecc - median_eccentricity)
    term_ext = -40.0 * (1.0 - extended_flux_preservation)

    raw_composite = 50.0 + term_bg + term_fwhm + term_stars + term_ecc + term_ext
    final_score = float("-inf") if cheating_detected else round(raw_composite, 6)

    return {
        "dataset": dataset_id,
        "score": final_score,
        "composite_score": final_score,
        "cheating_detected": cheating_detected,
        "cheat_reasons": cheat_reasons,
        "core_metrics": {
            "median_fwhm_px": round(median_fwhm_px, 4),
            "corner_fwhm_px": round(corner_fwhm_px, 4),
            "median_eccentricity": round(median_eccentricity, 4),
            "normalized_bg_rms": round(normalized_bg_rms, 6),
            "matched_star_count": int(matched_star_count),
            "extended_flux_preservation": round(extended_flux_preservation, 6),
        },
        "reference_metrics": {
            "ref_median_fwhm_px": round(ref_median_fwhm, 4),
            "ref_corner_fwhm_px": round(ref_corner_fwhm, 4),
            "ref_median_eccentricity": round(ref_median_ecc, 4),
            "ref_normalized_bg_rms": round(ref_norm_bg_rms, 6),
            "ref_star_count": int(ref_star_count),
            "exposure_ensemble_fwhm_px": ds_profile["exposure_ensemble_fwhm_px"],
        },
        "score_breakdown": {
            "anchor": 50.0,
            "bg_noise_term": round(term_bg, 4),
            "fwhm_sharpness_term": round(term_fwhm, 4),
            "star_depth_term": round(term_stars, 4),
            "eccentricity_term": round(term_ecc, 4),
            "extended_flux_penalty": round(term_ext, 4),
            "unclamped_composite": round(raw_composite, 6),
        },
        "photometric_normalization": {
            "target_star_flux_f0": TARGET_STAR_FLUX_F0,
            "matched_photometric_stars": len(matched_flux_pairs),
            "stack_scale": round(scale_stack, 8),
            "stack_offset": round(offset_stack, 6),
            "flux_linearity_r": round(flux_corr, 5),
            "flux_ratio_rel_mad": round(ratio_rel_mad, 5),
            "matched_ref_star_median_fwhm_px": round(matched_stack_fwhm, 4),
        },
        "anti_cheat_diagnostics": {
            "dark_clipping": clip_report,
            "noise_spatial_correlation_stack": {
                k: round(v, 5) for k, v in noise_corr_stack.items()
            },
            "noise_spatial_correlation_ref": {
                k: round(v, 5) for k, v in noise_corr_ref.items()
            },
            "extended_structure": ext_report,
            "residual_warm_pixel_impulses": int(stack_det["warm_pixel_artifacts"]),
            "transforms": trans_report,
        },
        "runtime_s": round(time.perf_counter() - t0, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Blind Astrophotography Stack Referee")
    parser.add_argument("--stack", required=True, help="Path to stacked float32 .npy (2856, 4290, 3)")
    parser.add_argument("--ref", required=True, help="Path to reference float32 .npy (2856, 4290, 3)")
    parser.add_argument("--transforms", default=None, help="Path to per-frame transforms.json")
    parser.add_argument(
        "--single-frame-baseline",
        action="store_true",
        help="Evaluate a single-frame reference baseline (Phase 2 calibration only).",
    )
    args = parser.parse_args()

    result = evaluate_stack(
        stack_path=args.stack,
        ref_path=args.ref,
        transforms_path=args.transforms,
        single_frame_baseline=args.single_frame_baseline,
    )
    print(json.dumps(result, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

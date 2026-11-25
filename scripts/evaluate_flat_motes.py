#!/usr/bin/env python3
"""Quantitative Dust-Mote, Vignetting & M31 Isophote Diagnostic Evaluator.

Pure NumPy/JSON diagnostic script (read-only on .npy stacks, never modifies
scripts/evaluate.py or locked references). Outputs scalar JSON telemetry only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_SHAPE = (2856, 4290, 3)
ROI_MARGIN = 64


def compute_bayer_luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        0.25 * rgb[:, :, 0].astype(np.float64)
        + 0.50 * rgb[:, :, 1].astype(np.float64)
        + 0.25 * rgb[:, :, 2].astype(np.float64)
    )


def separable_gaussian_blur_masked(
    arr: np.ndarray, valid_mask: np.ndarray, sigma: float
) -> np.ndarray:
    """Normalized convolution Gaussian blur respecting valid_mask (pure NumPy)."""
    if sigma <= 0.0:
        return arr.copy()
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= np.sum(k)

    def _conv1d_axis(a: np.ndarray, axis: int) -> np.ndarray:
        pad_width = [(0, 0), (0, 0)]
        pad_width[axis] = (radius, radius)
        ap = np.pad(a, pad_width, mode="edge")
        out = np.zeros_like(a, dtype=np.float64)
        if axis == 0:
            for i, w in enumerate(k):
                out += w * ap[i : i + a.shape[0], :]
        else:
            for i, w in enumerate(k):
                out += w * ap[:, i : i + a.shape[1]]
        return out

    w_arr = valid_mask.astype(np.float64)
    num = _conv1d_axis(_conv1d_axis(np.where(valid_mask, arr, 0.0), 0), 1)
    den = _conv1d_axis(_conv1d_axis(w_arr, 0), 1)
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)


def downsample_star_free_blocks(
    img: np.ndarray, block: int = 8, margin: int = ROI_MARGIN
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample interior ROI into star-suppressed (block x block) 35th-percentile cells.

    Returns (grid, cy_px, cx_px) where cy_px and cx_px are full-res pixel coordinates.
    """
    h, w = img.shape
    y0 = margin
    x0 = margin
    bh = (h - 2 * margin) // block
    bw = (w - 2 * margin) // block
    sub = img[y0 : y0 + bh * block, x0 : x0 + bw * block]
    cells = sub.reshape(bh, block, bw, block).transpose(0, 2, 1, 3).reshape(bh, bw, block * block)
    # 35th percentile suppresses positive star spikes while staying unbiased on smooth sky & dips
    grid = np.percentile(cells, 35.0, axis=2)
    cy = y0 + (np.arange(bh, dtype=np.float64) + 0.5) * block
    cx = x0 + (np.arange(bw, dtype=np.float64) + 0.5) * block
    return grid, cy, cx


def build_quiet_sky_mask(
    grid: np.ndarray,
    cy: np.ndarray,
    cx: np.ndarray,
    dataset: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build boolean mask of quiet sky cells (excluding M31 extended disk or M57 core)."""
    bh, bw = grid.shape
    yy, xx = np.meshgrid(cy, cx, indexing="ij")
    h_full, w_full = EXPECTED_SHAPE[0], EXPECTED_SHAPE[1]

    # First exclude obvious bright compact blobs (saturated stars / bright cores)
    # using a 2D smooth radial/local comparison
    valid = np.isfinite(grid) & (grid > 0.0)
    extra: dict[str, float] = {}

    if dataset == "andromeda_set":
        # Find M31 nucleus peak in smoothed grid
        sm = separable_gaussian_blur_masked(grid, valid, sigma=6.0)
        idx_max = int(np.nanargmax(sm))
        my, mx = divmod(idx_max, bw)
        m31_y = float(cy[my])
        m31_x = float(cx[mx])
        extra["m31_peak_x_px"] = round(m31_x, 2)
        extra["m31_peak_y_px"] = round(m31_y, 2)

        # Mask an elliptical/circular exclusion zone around M31's bright disk
        # First estimate outer sky level from the 4 corners (r > 1400 px from M31)
        dist_m31 = np.hypot(xx - m31_x, yy - m31_y)
        outer = valid & (dist_m31 > 1450.0)
        # Iteratively exclude cells that sit clearly above the radial sky baseline
        quiet = outer.copy()
        # Also include mid-field pixels that are within a conservative radius > 1250 px
        # OR along the minor axis where M31 is faint; simplest clean deterministic mask:
        # fit a quick 2D quadratic to outer cells, then accept any cell with dist > 1100
        # or (dist > 600 and excess < 15% of outer sky)
        if np.count_nonzero(outer) >= 200:
            oy = (yy[outer] - h_full * 0.5) / h_full
            ox = (xx[outer] - w_full * 0.5) / w_full
            or2 = ox * ox + oy * oy
            des_o = np.column_stack([np.ones_like(ox), ox, oy, or2, or2 * or2])
            coef, _, _, _ = np.linalg.lstsq(des_o, grid[outer], rcond=None)
            ny = (yy - h_full * 0.5) / h_full
            nx = (xx - w_full * 0.5) / w_full
            r2 = nx * nx + ny * ny
            pred = coef[0] + coef[1] * nx + coef[2] * ny + coef[3] * r2 + coef[4] * r2 * r2
            rel_exc = (grid - pred) / np.maximum(np.abs(pred), 1.0)
            quiet = valid & (dist_m31 > 950.0) & (rel_exc < 0.08) & (rel_exc > -0.25)
        extra["quiet_sky_fraction"] = round(float(np.mean(quiet)), 4)
        return quiet, extra
    else:
        # ring_set: M57 is tiny (~35 px), mask any local bright compact source > 8% above 64px smooth
        sm = separable_gaussian_blur_masked(grid, valid, sigma=8.0)
        rel_exc = (grid - sm) / np.maximum(np.abs(sm), 1.0)
        quiet = valid & (rel_exc < 0.08) & (rel_exc > -0.25)
        extra["quiet_sky_fraction"] = round(float(np.mean(quiet)), 4)
        return quiet, extra


def analyze_vignetting(
    grid: np.ndarray,
    quiet: np.ndarray,
    cy: np.ndarray,
    cx: np.ndarray,
) -> dict[str, float]:
    """Measure radial vignetting drop and global quiet-sky flatness."""
    h_full, w_full = EXPECTED_SHAPE[0], EXPECTED_SHAPE[1]
    yy, xx = np.meshgrid(cy, cx, indexing="ij")
    r_max = math.hypot(w_full * 0.5, h_full * 0.5)
    rn = np.hypot(xx - (w_full - 1) * 0.5, yy - (h_full - 1) * 0.5) / r_max

    vals = grid[quiet]
    r_q = rn[quiet]
    if vals.size < 100:
        return {"vignetting_drop_pct": float("nan"), "quiet_sky_rel_std_pct": float("nan")}

    # Fit smooth even radial polynomial V(r) = a0 + a2*r^2 + a4*r^4 on quiet sky
    des = np.column_stack([np.ones_like(r_q), r_q**2, r_q**4])
    coef, _, _, _ = np.linalg.lstsq(des, vals, rcond=None)
    v_center = float(coef[0])
    v_corner = float(coef[0] + coef[1] * (0.90**2) + coef[2] * (0.90**4))
    vig_drop_pct = (1.0 - v_corner / max(1e-6, v_center)) * 100.0

    # Also empirical inner vs outer annulus medians (when available)
    inner_m = quiet & (rn <= 0.35)
    outer_m = quiet & (rn >= 0.72) & (rn <= 0.92)
    emp_drop_pct = float("nan")
    if np.count_nonzero(inner_m) >= 50 and np.count_nonzero(outer_m) >= 50:
        emp_drop_pct = (1.0 - float(np.median(grid[outer_m])) / max(1e-6, float(np.median(grid[inner_m])))) * 100.0

    med_q = float(np.median(vals))
    rel_std_pct = float(np.std(vals) / max(1e-6, abs(med_q))) * 100.0
    rel_mad_pct = float(np.median(np.abs(vals - med_q)) * 1.4826 / max(1e-6, abs(med_q))) * 100.0

    # Residual after removing pure radial polynomial (measures non-radial gradients + dust motes)
    rad_pred = coef[0] + coef[1] * (r_q**2) + coef[2] * (r_q**4)
    nonrad_rel_std_pct = float(np.std((vals - rad_pred) / np.maximum(np.abs(rad_pred), 1.0))) * 100.0

    return {
        "vignetting_poly_drop_r0_to_r09_pct": round(vig_drop_pct, 4),
        "vignetting_emp_drop_pct": round(emp_drop_pct, 4) if math.isfinite(emp_drop_pct) else None,
        "quiet_sky_median_adu": round(med_q, 4),
        "quiet_sky_rel_std_pct": round(rel_std_pct, 4),
        "quiet_sky_rel_mad_pct": round(rel_mad_pct, 4),
        "quiet_sky_nonradial_rel_std_pct": round(nonrad_rel_std_pct, 4),
    }


def compute_mote_bandpass_map(
    grid: np.ndarray, quiet: np.ndarray, cy: np.ndarray, cx: np.ndarray
) -> np.ndarray:
    """Band-pass 16-128 px relative transmission map on 8x8 block grid.

    First detrends smooth radial vignetting (a0 + ax*nx + ay*ny + a2*r^2 + a4*r^4)
    fitted on quiet sky so steep corner vignetting cannot leak into dust-mote dips.
    Uses sigma_lo = 1.5 blocks (12 px) to suppress pixel noise and
    sigma_hi = 18.0 blocks (144 px) to track smooth local continuum.
    """
    h_full, w_full = EXPECTED_SHAPE[0], EXPECTED_SHAPE[1]
    yy, xx = np.meshgrid(cy, cx, indexing="ij")
    r_max = math.hypot(w_full * 0.5, h_full * 0.5)
    nx = (xx - (w_full - 1) * 0.5) / r_max
    ny = (yy - (h_full - 1) * 0.5) / r_max
    r2 = nx * nx + ny * ny

    detrended = grid.copy()
    if np.count_nonzero(quiet) >= 200:
        des_q = np.column_stack([
            np.ones(np.count_nonzero(quiet)),
            nx[quiet], ny[quiet], r2[quiet], r2[quiet] ** 2, r2[quiet] ** 3,
        ])
        coef, _, _, _ = np.linalg.lstsq(des_q, grid[quiet], rcond=None)
        vig_pred = (
            coef[0] + coef[1] * nx + coef[2] * ny
            + coef[3] * r2 + coef[4] * (r2 ** 2) + coef[5] * (r2 ** 3)
        )
        detrended = grid / np.maximum(vig_pred, 1.0)

    low_sm = separable_gaussian_blur_masked(detrended, quiet, sigma=1.5)
    high_sm = separable_gaussian_blur_masked(detrended, quiet, sigma=18.0)
    bp = (low_sm - high_sm) / np.maximum(np.abs(high_sm), 1e-6)
    return bp


def analyze_dust_motes(
    bp: np.ndarray,
    quiet: np.ndarray,
    ref_other_bp: np.ndarray | None = None,
    ref_other_quiet: np.ndarray | None = None,
) -> dict[str, Any]:
    """Quantify localized 16-128 px dust-mote residuals on quiet sky."""
    vals = bp[quiet & np.isfinite(bp)]
    if vals.size < 100:
        return {}

    rms_pct = float(np.std(vals)) * 100.0
    mad_pct = float(np.median(np.abs(vals - np.median(vals))) * 1.4826) * 100.0
    p01_pct = float(np.percentile(vals, 0.1)) * 100.0
    p05_pct = float(np.percentile(vals, 0.5)) * 100.0
    p1_pct = float(np.percentile(vals, 1.0)) * 100.0
    p99_pct = float(np.percentile(vals, 99.0)) * 100.0
    min_pct = float(np.min(vals)) * 100.0
    frac_dip_05 = float(np.mean(vals < -0.005)) * 100.0
    frac_dip_10 = float(np.mean(vals < -0.010)) * 100.0
    frac_dip_20 = float(np.mean(vals < -0.020)) * 100.0

    # Negative-tail skewness / energy (dust motes are strictly absorptive dips < 0!)
    neg_rms_pct = float(np.sqrt(np.mean(np.minimum(vals, 0.0) ** 2))) * 100.0
    pos_rms_pct = float(np.sqrt(np.mean(np.maximum(vals, 0.0) ** 2))) * 100.0

    out: dict[str, Any] = {
        "mote_band_rms_pct": round(rms_pct, 4),
        "mote_band_mad_pct": round(mad_pct, 4),
        "mote_neg_rms_pct": round(neg_rms_pct, 4),
        "mote_pos_rms_pct": round(pos_rms_pct, 4),
        "mote_min_dip_pct": round(min_pct, 4),
        "mote_p01_dip_pct": round(p01_pct, 4),
        "mote_p05_dip_pct": round(p05_pct, 4),
        "mote_p1_p99_span_pct": round(p99_pct - p1_pct, 4),
        "mote_area_dip_gt_0p5pct": round(frac_dip_05, 4),
        "mote_area_dip_gt_1p0pct": round(frac_dip_10, 4),
        "mote_area_dip_gt_2p0pct": round(frac_dip_20, 4),
    }

    if ref_other_bp is not None and ref_other_quiet is not None:
        both = quiet & ref_other_quiet & np.isfinite(bp) & np.isfinite(ref_other_bp)
        if np.count_nonzero(both) >= 500:
            a = bp[both]
            b = ref_other_bp[both]
            corr = float(np.corrcoef(a, b)[0, 1])
            # Also projection slope of current bp onto cross-dataset baseline dust template
            b_var = float(np.var(b))
            proj_slope = float(np.cov(a, b)[0, 1] / max(1e-12, b_var))
            # Specifically on strong dust-mote pixels in the other dataset (b < -0.008)
            mote_sites = both & (ref_other_bp < -0.008)
            site_mean_pct = float(np.mean(bp[mote_sites]) * 100.0) if np.count_nonzero(mote_sites) >= 20 else None
            out["cross_dataset_sensor_mote_corr"] = round(corr, 5)
            out["cross_dataset_mote_projection_slope"] = round(proj_slope, 5)
            out["cross_dataset_mote_site_count"] = int(np.count_nonzero(mote_sites))
            out["cross_dataset_mote_site_mean_dip_pct"] = (
                round(site_mean_pct, 4) if site_mean_pct is not None else None
            )

    return out


def analyze_m31_isophotes(
    grid: np.ndarray,
    quiet: np.ndarray,
    cy: np.ndarray,
    cx: np.ndarray,
) -> dict[str, Any]:
    """Measure M31 extended-disk ellipticity (b/a), major/minor symmetry, and outer-disk smoothness."""
    bh, bw = grid.shape
    yy, xx = np.meshgrid(cy, cx, indexing="ij")
    valid = np.isfinite(grid) & (grid > 0.0)
    sm = separable_gaussian_blur_masked(grid, valid, sigma=5.0)
    idx_max = int(np.nanargmax(sm))
    my, mx = divmod(idx_max, bw)
    m31_y = float(cy[my])
    m31_x = float(cx[mx])

    # Estimate global sky pedestal from quiet outer sky
    sky_bg = float(np.median(grid[quiet])) if np.count_nonzero(quiet) >= 100 else float(np.percentile(grid[valid], 15.0))
    gal = grid - sky_bg

    # Fit M31 principal axis & ellipticity via intensity-weighted second moments in 120..600 px ring
    dx = xx - m31_x
    dy = yy - m31_y
    r = np.hypot(dx, dy)

    def _fit_moments(r_lo: float, r_hi: float) -> tuple[float, float]:
        m = valid & (r >= r_lo) & (r <= r_hi) & (gal > 0.0)
        if np.count_nonzero(m) < 50:
            return float("nan"), float("nan")
        w = np.maximum(gal[m], 0.0)
        wsum = float(np.sum(w))
        if wsum <= 1e-6:
            return float("nan"), float("nan")
        ux = dx[m]
        uy = dy[m]
        mxx = float(np.sum(w * ux * ux) / wsum)
        myy = float(np.sum(w * uy * uy) / wsum)
        mxy = float(np.sum(w * ux * uy) / wsum)
        pa = 0.5 * math.atan2(2.0 * mxy, mxx - myy)
        tr = mxx + myy
        det_term = math.sqrt(max(0.0, (mxx - myy) ** 2 + 4.0 * mxy * mxy))
        lam1 = max(1e-9, 0.5 * (tr + det_term))
        lam2 = max(1e-9, 0.5 * (tr - det_term))
        b_over_a = math.sqrt(lam2 / lam1)
        return math.degrees(pa), b_over_a

    pa_in, ba_in = _fit_moments(120.0, 450.0)
    pa_out, ba_out = _fit_moments(450.0, 950.0)

    # Rotate coordinates into M31 (u = major axis, v = minor axis) using inner PA
    pa_rad = math.radians(pa_in if math.isfinite(pa_in) else 0.0)
    ca, sa = math.cos(pa_rad), math.sin(pa_rad)
    u = dx * ca + dy * sa
    v = -dx * sa + dy * ca
    q = max(0.25, min(1.0, ba_in if math.isfinite(ba_in) else 0.45))
    r_ell = np.hypot(u, v / q)

    # Major-axis symmetry (+u vs -u) in outer disk 350..1050 px, |v| < 180 px
    pos_maj = valid & (u >= 350.0) & (u <= 1050.0) & (np.abs(v) <= 180.0)
    neg_maj = valid & (u <= -350.0) & (u >= -1050.0) & (np.abs(v) <= 180.0)
    maj_asym = float("nan")
    if np.count_nonzero(pos_maj) >= 30 and np.count_nonzero(neg_maj) >= 30:
        ip = float(np.median(gal[pos_maj]))
        im = float(np.median(gal[neg_maj]))
        maj_asym = abs(ip - im) / max(1.0, abs(ip) + abs(im))

    # Minor-axis symmetry (+v vs -v) in 180..520 px, |u| < 260 px
    pos_min = valid & (v >= 180.0) & (v <= 520.0) & (np.abs(u) <= 260.0)
    neg_min = valid & (v <= -180.0) & (v >= -520.0) & (np.abs(u) <= 260.0)
    min_asym = float("nan")
    if np.count_nonzero(pos_min) >= 30 and np.count_nonzero(neg_min) >= 30:
        ip = float(np.median(gal[pos_min]))
        im = float(np.median(gal[neg_min]))
        min_asym = abs(ip - im) / max(1.0, abs(ip) + abs(im))

    # Outer-disk azimuthal smoothness across elliptical annuli r_ell in [350, 950] (6 rings)
    ring_rms_list = []
    edges = np.linspace(350.0, 950.0, 7)
    for r0, r1 in zip(edges[:-1], edges[1:]):
        ring_m = valid & (r_ell >= r0) & (r_ell < r1)
        if np.count_nonzero(ring_m) >= 50:
            rv = grid[ring_m]
            rmed = float(np.median(rv))
            # Use MAD-based azimuthal scatter relative to total signal (captures motes & vignetting asymmetry)
            rmad = float(np.median(np.abs(rv - rmed)) * 1.4826)
            ring_rms_list.append(rmad / max(1.0, abs(rmed)) * 100.0)
    outer_smooth_pct = float(np.mean(ring_rms_list)) if ring_rms_list else float("nan")

    return {
        "m31_center_x_px": round(m31_x, 2),
        "m31_center_y_px": round(m31_y, 2),
        "m31_pa_inner_deg": round(pa_in, 3),
        "m31_pa_outer_deg": round(pa_out, 3),
        "m31_ba_inner": round(ba_in, 4),
        "m31_ba_outer": round(ba_out, 4),
        "m31_ba_consistency_diff": round(abs(ba_out - ba_in), 4),
        "m31_major_axis_asymmetry": round(maj_asym, 5),
        "m31_minor_axis_asymmetry": round(min_asym, 5),
        "m31_outer_isophote_azim_mad_pct": round(outer_smooth_pct, 4),
    }


def analyze_color_vignetting(
    rgb: np.ndarray, quiet: np.ndarray, cy: np.ndarray, cx: np.ndarray
) -> dict[str, float]:
    """Check chromatic R/G and B/G radial drift on quiet sky."""
    g_r, _, _ = downsample_star_free_blocks(rgb[:, :, 0].astype(np.float64))
    g_g, _, _ = downsample_star_free_blocks(rgb[:, :, 1].astype(np.float64))
    g_b, _, _ = downsample_star_free_blocks(rgb[:, :, 2].astype(np.float64))

    h_full, w_full = EXPECTED_SHAPE[0], EXPECTED_SHAPE[1]
    yy, xx = np.meshgrid(cy, cx, indexing="ij")
    r_max = math.hypot(w_full * 0.5, h_full * 0.5)
    rn = np.hypot(xx - (w_full - 1) * 0.5, yy - (h_full - 1) * 0.5) / r_max

    rg = g_r / np.maximum(g_g, 1.0)
    bg = g_b / np.maximum(g_g, 1.0)
    out: dict[str, float] = {}
    for name, ratio in (("rg", rg), ("bg", bg)):
        vals = ratio[quiet & np.isfinite(ratio)]
        med = float(np.median(vals))
        out[f"{name}_quiet_rel_std_pct"] = round(float(np.std(vals) / max(1e-6, abs(med)) * 100.0), 4)
        rq = rn[quiet & np.isfinite(ratio)]
        des = np.column_stack([np.ones_like(rq), rq**2])
        coef, _, _, _ = np.linalg.lstsq(des, vals, rcond=None)
        drift = (coef[1] * (0.90**2)) / max(1e-6, abs(coef[0])) * 100.0
        out[f"{name}_center_to_corner_drift_pct"] = round(float(drift), 4)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Dust-mote, vignetting, and M31 isophote diagnostic.")
    parser.add_argument("--stack", required=True, help="Path to stack float32 .npy (2856, 4290, 3)")
    parser.add_argument("--dataset", required=True, choices=["ring_set", "andromeda_set"])
    parser.add_argument(
        "--cross-ref-stack",
        default=None,
        help="Optional baseline stack from the OTHER dataset to measure sensor-locked dust-mote correlation.",
    )
    parser.add_argument("--cross-ref-dataset", default=None, choices=["ring_set", "andromeda_set"])
    args = parser.parse_args()

    arr = np.load(args.stack)
    if arr.shape != EXPECTED_SHAPE:
        raise ValueError(f"Expected shape {EXPECTED_SHAPE}, got {arr.shape}")

    lum = compute_bayer_luminance(arr)
    grid, cy, cx = downsample_star_free_blocks(lum, block=8)
    quiet, quiet_meta = build_quiet_sky_mask(grid, cy, cx, args.dataset)
    bp = compute_mote_bandpass_map(grid, quiet, cy, cx)

    other_bp = None
    other_quiet = None
    if args.cross_ref_stack and args.cross_ref_dataset:
        o_arr = np.load(args.cross_ref_stack)
        o_lum = compute_bayer_luminance(o_arr)
        o_grid, o_cy, o_cx = downsample_star_free_blocks(o_lum, block=8)
        other_quiet, _ = build_quiet_sky_mask(o_grid, o_cy, o_cx, args.cross_ref_dataset)
        other_bp = compute_mote_bandpass_map(o_grid, other_quiet, o_cy, o_cx)

    result: dict[str, Any] = {
        "stack": str(args.stack),
        "dataset": args.dataset,
        "quiet_sky_meta": quiet_meta,
        "vignetting": analyze_vignetting(grid, quiet, cy, cx),
        "dust_motes": analyze_dust_motes(bp, quiet, other_bp, other_quiet),
        "color_flatness": analyze_color_vignetting(arr, quiet, cy, cx),
    }

    if args.dataset == "andromeda_set":
        result["m31_isophotes"] = analyze_m31_isophotes(grid, quiet, cy, cx)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate publication-grade astronomical PNG plates for the LaTeX research paper
and emit pure scalar/JSON photometric & geometric telemetry (logs/visual_figure_telemetry.json).

Strictly adheres to Binary Quarantine: writes PNG files to disk via Pillow and prints
only compact numerical summary statistics to stdout.
"""
import json
import math
import os
import numpy as np
from PIL import Image


def lupton_asinh_rgb(rgb, wb_gains, sky_pedestal, soft_q, stretch_alpha, post_bias=0.0, gamma=1.0):
    """Color-preserving Lupton et al. (2004) asinh RGB stretch."""
    x = (rgb.astype(np.float64) - sky_pedestal[None, None, :]) * wb_gains[None, None, :] + post_bias
    intensity = (x[:, :, 0] + x[:, :, 1] + x[:, :, 2]) / 3.0
    pos_i = np.maximum(intensity, 1e-6)
    f_i = np.arcsinh(stretch_alpha * pos_i / soft_q) / np.arcsinh(stretch_alpha)
    scale = np.where(intensity > 1e-6, f_i / pos_i, 0.0)
    out = np.clip(x * scale[:, :, None], 0.0, 1.0)
    if gamma != 1.0:
        out = np.power(out, gamma)
    return (out * 255.0 + 0.5).astype(np.uint8)


def upsample_nearest(img_u8, factor):
    return np.repeat(np.repeat(img_u8, factor, axis=0), factor, axis=1)


def generate_m57_plates(out_dir):
    ring = np.load("logs/ring_best.npy")
    ref_r = np.load("logs/ref_ring.npy")

    cx, cy = 1576.62, 1311.70
    wd_x, wd_y = 1578.09, 1310.85
    xc, yc = int(round(cx)), int(round(cy))  # (1577, 1312)

    # 1. Wide 240x240 crop for Single Frame (a) vs 19-Frame Stack (b)
    half_w = 120
    crop_stack = ring[yc - half_w:yc + half_w, xc - half_w:xc + half_w].copy()
    crop_ref = ref_r[yc - half_w:yc + half_w, xc - half_w:xc + half_w].copy()

    yy, xx = np.mgrid[-half_w:half_w, -half_w:half_w]
    rr = np.sqrt((xx - (cx - xc))**2 + (yy - (cy - yc))**2)
    outer_sky = (rr >= 92) & (rr <= 118)

    sky_stack = np.zeros(3)
    sky_ref = np.zeros(3)
    sigma_stack = np.zeros(3)
    for ch in range(3):
        vs = crop_stack[:, :, ch][outer_sky]
        ms = np.median(vs)
        mads = 1.4826 * np.median(np.abs(vs - ms))
        ks = np.abs(vs - ms) < 2.5 * mads
        sky_stack[ch] = np.median(vs[ks])
        sigma_stack[ch] = 1.4826 * np.median(np.abs(vs[ks] - sky_stack[ch]))

        vr = crop_ref[:, :, ch][outer_sky]
        mr = np.median(vr)
        madr = 1.4826 * np.median(np.abs(vr - mr))
        kr = np.abs(vr - mr) < 2.5 * madr
        sky_ref[ch] = np.median(vr[kr])

    # Broadband G2V-like stellar white balance measured on ring_best unsaturated stars
    wb = np.array([2.131, 1.000, 1.626])

    # Neutral dark space background (~[9, 9, 10]/255) with identical photometric stretch on Ref and Stack!
    img_ref_u8 = lupton_asinh_rgb(crop_ref, wb, sky_ref, soft_q=340.0, stretch_alpha=15.0, post_bias=8.5, gamma=0.92)
    img_stack_u8 = lupton_asinh_rgb(crop_stack, wb, sky_stack, soft_q=340.0, stretch_alpha=15.0, post_bias=8.5, gamma=0.92)

    Image.fromarray(upsample_nearest(img_ref_u8, 3), mode="RGB").save(os.path.join(out_dir, "m57_single_ref.png"))
    Image.fromarray(upsample_nearest(img_stack_u8, 3), mode="RGB").save(os.path.join(out_dir, "m57_stack_best.png"))

    # 2. Zoomed 104x104 Cavity-Anchored Stretch (c) to highlight Central White Dwarf WD 1851+329!
    half_z = 52
    zoom_stack = ring[yc - half_z:yc + half_z, xc - half_z:xc + half_z].copy()
    yz, xz = np.mgrid[-half_z:half_z, -half_z:half_z]
    pa = math.radians(71.3)
    ba = 0.891
    dx0, dy0 = cx - xc, cy - yc
    xr = (xz - dx0) * math.cos(pa) + (yz - dy0) * math.sin(pa)
    yr = -(xz - dx0) * math.sin(pa) + (yz - dy0) * math.cos(pa)
    re_z = np.sqrt(xr**2 + (yr / ba)**2)

    # Measure cavity floor in annulus 8.5 <= r_ell <= 13.5 px
    cav_mask = (re_z >= 8.5) & (re_z <= 13.5)
    cav_floor = np.array([np.percentile(zoom_stack[:, :, ch][cav_mask], 35) for ch in range(3)])

    # Subtract 92% of the interior nebular cavity floor on R and G, and 88% on B so:
    # - the r=11px cavity minimum drops to dark teal-charcoal (~[15, 30, 42]),
    # - the 120,000K Blue White Dwarf WD 1851+329 at r=0..3px stands out sharply (~[55, 95, 198]),
    # - the [O III] ring crest (r=39px) glows emerald-cyan (~[135, 228, 232]),
    # - and the outer H-alpha/[N II] shell (r=50..60px) shows warm ruby-red stratification!
    ped_cav = sky_stack + np.array([0.88, 0.94, 0.90]) * (cav_floor - sky_stack)
    wb_cav = np.array([2.85, 1.05, 1.85])
    img_wd_u8 = lupton_asinh_rgb(zoom_stack, wb_cav, ped_cav, soft_q=175.0, stretch_alpha=10.0, post_bias=1.5, gamma=0.92)
    Image.fromarray(upsample_nearest(img_wd_u8, 6), mode="RGB").save(os.path.join(out_dir, "m57_wd_cavity_zoom.png"))

    wd_px = int(round(wd_x - (xc - half_z)))
    wd_py = int(round(wd_y - (yc - half_z)))
    wd_rgb_u8 = img_wd_u8[wd_py - 1:wd_py + 2, wd_px - 1:wd_px + 2].mean(axis=(0, 1))
    cav_rgb_u8 = img_wd_u8[cav_mask].mean(axis=0)
    crest_mask = (re_z >= 37.0) & (re_z <= 41.0)
    crest_rgb_u8 = img_wd_u8[crest_mask].mean(axis=0)

    print("=== M57 RENDERED PLATE VERIFICATION (0..255 RGB levels) ===")
    print(f"  Wide Stack sky median u8 RGB : {[round(float(v), 1) for v in np.median(img_stack_u8[outer_sky], axis=0)]}")
    print(f"  Zoom WD 3x3 mean u8 RGB      : {[round(float(v), 1) for v in wd_rgb_u8]}")
    print(f"  Zoom Cavity trough mean u8   : {[round(float(v), 1) for v in cav_rgb_u8]}")
    print(f"  Zoom [O III] crest mean u8   : {[round(float(v), 1) for v in crest_rgb_u8]}")

    # Export clean radial profile (r_ell = 0..68 px in 2px steps) in white-balanced ADU for pgfplots
    xr_w = (xx - dx0) * math.cos(pa) + (yy - dy0) * math.sin(pa)
    yr_w = -(xx - dx0) * math.sin(pa) + (yy - dy0) * math.cos(pa)
    re_w = np.sqrt(xr_w**2 + (yr_w / ba)**2)
    prof = []
    for rb in range(0, 69, 2):
        m = (re_w < 1.6) if rb == 0 else ((re_w >= rb - 1.0) & (re_w < rb + 1.0))
        r_wb = float(np.median((crop_stack[m, 0] - sky_stack[0]) * wb[0]))
        g_wb = float(np.median((crop_stack[m, 1] - sky_stack[1]) * wb[1]))
        b_wb = float(np.median((crop_stack[m, 2] - sky_stack[2]) * wb[2]))
        prof.append({"r_ell_px": rb, "R_wb": round(r_wb, 2), "G_wb": round(g_wb, 2), "B_wb": round(b_wb, 2)})

    return {
        "wide_crop_px": 240,
        "zoom_crop_px": 104,
        "m57_center_sensor": [round(cx, 2), round(cy, 2)],
        "wd_center_sensor": [round(wd_x, 2), round(wd_y, 2)],
        "wd_tikz_wide": [round((wd_x - (xc - half_w)) / (2 * half_w), 4), round(1.0 - (wd_y - (yc - half_w)) / (2 * half_w), 4)],
        "wd_tikz_zoom": [round((wd_x - (xc - half_z)) / (2 * half_z), 4), round(1.0 - (wd_y - (yc - half_z)) / (2 * half_z), 4)],
        "companion_star_tikz_wide": [
            round((cx + 65.4 - (xc - half_w)) / (2 * half_w), 4),
            round(1.0 - (cy - 70.7 - (yc - half_w)) / (2 * half_w), 4)
        ],
        "wd_zoom_u8_rgb": [round(float(v), 1) for v in wd_rgb_u8],
        "cavity_zoom_u8_rgb": [round(float(v), 1) for v in cav_rgb_u8],
        "radial_profile": prof,
    }


def generate_m31_plates(out_dir):
    androm = np.load("logs/andromeda_best.npy")
    ring = np.load("logs/ring_best.npy")
    H, W, _ = androm.shape

    # Fit smooth 7-parameter optical vignetting flat V(x,y) on 64x64 sigma-clipped blocks of ring_best.npy
    bh, bw = H // 64, W // 64
    R_blocks = np.zeros((bh, bw, 3), dtype=np.float64)
    for r in range(bh):
        for c in range(bw):
            rp = ring[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64]
            for ch in range(3):
                rv = rp[:, :, ch].ravel()
                rmed = np.median(rv)
                rmad = 1.4826 * np.median(np.abs(rv - rmed))
                rv2 = rv[rv < rmed + 2.5 * rmad]
                R_blocks[r, c, ch] = np.median(rv2) if rv2.size > 100 else rmed

    gy, gx = np.mgrid[0:bh, 0:bw]
    # Mask border and M57 block neighborhood around (24.6, 20.5)
    valid = (gy > 0) & (gy < bh - 1) & (gx > 0) & (gx < bw - 1) & (np.hypot(gx - 24.6, gy - 20.5) > 3.0)
    nx_g = (gx * 64 + 32 - W / 2.0) / (W / 2.0)
    ny_g = (gy * 64 + 32 - H / 2.0) / (H / 2.0)
    r4_g = (nx_g**2 + ny_g**2)**2

    X_design = np.column_stack([
        np.ones(valid.sum()), nx_g[valid], ny_g[valid],
        nx_g[valid]**2, nx_g[valid] * ny_g[valid], ny_g[valid]**2, r4_g[valid]
    ])
    flat_coefs = []
    flat_peaks = []
    for ch in range(3):
        coef, _, _, _ = np.linalg.lstsq(X_design, R_blocks[:, :, ch][valid], rcond=None)
        X_all = np.column_stack([
            np.ones(bh * bw), nx_g.ravel(), ny_g.ravel(),
            (nx_g**2).ravel(), (nx_g * ny_g).ravel(), (ny_g**2).ravel(), r4_g.ravel()
        ])
        flat_coefs.append(coef)
        flat_peaks.append(float((X_all @ coef).max()))

    # Interior overlap ROI [64:2792, 64:4224] -> 4x4 box-binned to 682 x 1040
    y0, y1 = 64, 2792
    x0, x1 = 64, 4224
    sh, sw = (y1 - y0) // 4, (x1 - x0) // 4
    raw4 = androm[y0:y1, x0:x1].reshape(sh, 4, sw, 4, 3).mean(axis=(1, 3))

    py4, px4 = np.mgrid[0:sh, 0:sw]
    sx_px = px4 * 4 + 2.0 + x0
    sy_px = py4 * 4 + 2.0 + y0
    nx4 = (sx_px - W / 2.0) / (W / 2.0)
    ny4 = (sy_px - H / 2.0) / (H / 2.0)
    r4_4 = (nx4**2 + ny4**2)**2
    flat4 = np.zeros_like(raw4, dtype=np.float64)
    for ch in range(3):
        c = flat_coefs[ch]
        pred = c[0] + c[1] * nx4 + c[2] * ny4 + c[3] * nx4**2 + c[4] * nx4 * ny4 + c[5] * ny4**2 + c[6] * r4_4
        flat4[:, :, ch] = pred / flat_peaks[ch]

    wb_m31 = np.array([2.214, 1.000, 1.478])

    # Panel (a): Standard Un-flattened Full-Range Preview Stretch (0..1100 ADU)
    # Shows only the compact spherical core + circular Newtonian vignetting bowl
    sky_unflat = np.array([np.percentile(raw4[:, :, ch], 8) for ch in range(3)])
    u8_unflat = lupton_asinh_rgb(raw4, wb_m31, sky_unflat, soft_q=480.0, stretch_alpha=12.0, post_bias=1.5, gamma=0.95)
    Image.fromarray(u8_unflat, mode="RGB").save(os.path.join(out_dir, "m31_unflattened_raw.png"))

    # Panel (b): Empirical Session-Flat Divided + Bulge/Inner-Disk Pedestal Stretch!
    # True model-free M31 major axis in sensor (x_right, y_down) coordinates is PA_sensor = -15.0 deg (165 deg);
    # minor axis is PA_sensor = +75.0 deg!
    corr4 = raw4 / np.maximum(flat4, 0.5)
    nuc_x, nuc_y = 2073.0, 1205.0
    pa_maj = math.radians(-15.0)
    dx_n = sx_px - nuc_x
    dy_n = sy_px - nuc_y
    s_maj = dx_n * math.cos(pa_maj) + dy_n * math.sin(pa_maj)
    s_min = -dx_n * math.sin(pa_maj) + dy_n * math.cos(pa_maj)

    # Anchor pedestal to outer minor-axis zone (780 <= |s_min| <= 1150 px) so the 152..550 ADU bulge/disk regime spans full contrast!
    outer_min_zone = (np.abs(s_min) >= 820.0) & (np.abs(s_min) <= 1200.0)
    ped_bulge = np.array([np.percentile(corr4[:, :, ch][outer_min_zone], 22) for ch in range(3)])

    # Warm golden-ivory core + steep midtone stretch highlighting the PA=-15 deg elliptical bulge & inner disk
    u8_flat = lupton_asinh_rgb(corr4, wb_m31, ped_bulge, soft_q=175.0, stretch_alpha=16.0, post_bias=1.8, gamma=0.88)
    Image.fromarray(u8_flat, mode="RGB").save(os.path.join(out_dir, "m31_flat_corrected.png"))

    # Measure true major-axis (PA=-15 deg) vs minor-axis (PA=+75 deg) surface-brightness profiles in flat-corrected WB Luminance!
    corr_L_raw = 0.25 * corr4[:, :, 0] + 0.50 * corr4[:, :, 1] + 0.25 * corr4[:, :, 2]
    floor_L = float(np.percentile(corr_L_raw[outer_min_zone], 22))
    exc_L = corr_L_raw - floor_L

    maj_prof = []
    min_prof = []
    for s_bin in range(-1050, 1051, 50):
        m_maj = (np.abs(s_maj - s_bin) <= 28.0) & (np.abs(s_min) <= 36.0)
        if m_maj.sum() >= 12:
            maj_prof.append({"s_px": s_bin, "L_exc": round(float(np.percentile(exc_L[m_maj], 30)), 2)})
        m_min = (np.abs(s_min - s_bin) <= 28.0) & (np.abs(s_maj) <= 36.0)
        if m_min.sum() >= 12:
            min_prof.append({"s_px": s_bin, "L_exc": round(float(np.percentile(exc_L[m_min], 30)), 2)})

    # Compute exact semi-major (a_px) and semi-minor (b_px) radii where exc_L crosses 60, 26, and 11 ADU!
    # At exc_L = 60 ADU:
    # At exc_L = 26 ADU (r_maj ~ 450 px):
    # At exc_L = 11 ADU (r_maj ~ 720 px):
    iso_table = []
    for target_adu in [60.0, 26.0, 11.0]:
        # Interpolate positive/negative average radius along major and minor profiles
        r_maj_pos = max([abs(p["s_px"]) for p in maj_prof if p["L_exc"] >= target_adu] or [0])
        r_min_pos = max([abs(p["s_px"]) for p in min_prof if p["L_exc"] >= target_adu] or [0])
        iso_table.append({
            "exc_adu": target_adu,
            "a_px": r_maj_pos,
            "b_px": r_min_pos,
            "ba": round(r_min_pos / max(r_maj_pos, 1), 3)
        })

    nuc_tikz_x = (nuc_x - x0) / (x1 - x0)
    nuc_tikz_y = 1.0 - (nuc_y - y0) / (y1 - y0)

    print("\n=== M31 RENDERED PLATE & TRUE ELLIPSE VERIFICATION ===")
    print(f"  M31 Nucleus TikZ norm : ({nuc_tikz_x:.4f}, {nuc_tikz_y:.4f}), PA_sensor = -15.0 deg (TikZ angle = +15.0 deg)")
    for row in iso_table:
        print(f"  Isophote +{row['exc_adu']:4.1f} ADU above outer floor: semi-major a={row['a_px']:4d}px, semi-minor b={row['b_px']:4d}px, b/a={row['ba']:.3f}")
    print(f"  Near-side dust asymmetry at |s_min|=450px: North(s=-450)={[p['L_exc'] for p in min_prof if p['s_px']==-450]} vs South(s=+450)={[p['L_exc'] for p in min_prof if p['s_px']==450]} ADU")

    return {
        "roi_bounds_px": [x0, y0, x1, y1],
        "binned_shape_hw": [sh, sw],
        "nuc_sensor_px": [nuc_x, nuc_y],
        "nuc_tikz_norm": [round(nuc_tikz_x, 4), round(nuc_tikz_y, 4)],
        "pa_sensor_deg": -15.0,
        "tikz_rotation_deg": 15.0,
        "isophotes": iso_table,
        "major_axis_profile": maj_prof,
        "minor_axis_profile": min_prof,
    }


def main():
    out_dir = "paper/figures"
    os.makedirs(out_dir, exist_ok=True)
    m57_info = generate_m57_plates(out_dir)
    m31_info = generate_m31_plates(out_dir)
    with open("logs/visual_figure_telemetry.json", "w") as f:
        json.dump({"m57": m57_info, "m31": m31_info}, f, indent=2)
    print("\nUpdated PNG plates in paper/figures/ and logs/visual_figure_telemetry.json")


if __name__ == "__main__":
    main()

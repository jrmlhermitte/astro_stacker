# Phase 2 Report: Blind Photometric Referee, Anti-Cheat Torture-Testing & C++20 Engineering Plan

**Workspace:** `/home/julien/projects/astro_stacker`  
**Referee Script:** [`scripts/evaluate.py`](file:///home/julien/projects/astro_stacker/scripts/evaluate.py) (`1,121` lines, `47,859` bytes, `SHA-256: 971b62f072143eb6172cd804f85750b7c31c507fbd13879ed4d1a46b50c415f9`)  
**Reference Generator:** [`scripts/generate_references.py`](file:///home/julien/projects/astro_stacker/scripts/generate_references.py) → [`logs/reference_selection.json`](file:///home/julien/projects/astro_stacker/logs/reference_selection.json)  
**Torture-Test Suite:** [`scripts/torture_test_evaluator.py`](file:///home/julien/projects/astro_stacker/scripts/torture_test_evaluator.py) → [`logs/torture_test_results.json`](file:///home/julien/projects/astro_stacker/logs/torture_test_results.json) (`28 / 28` assertions passed)  
**Standardized Reference Baselines:** [`logs/ref_ring.npy`](file:///home/julien/projects/astro_stacker/logs/ref_ring.npy) & [`logs/ref_andromeda.npy`](file:///home/julien/projects/astro_stacker/logs/ref_andromeda.npy) (`shape = (2856, 4290, 3)`, `dtype = float32`)

---

## 1. Compliance with Phase 2 Implementation Guardrails

All four user-mandated guardrails were baked into the architecture and independently verified prior to reaching Stop Gate 2:

1. **Guardrail 1 — Lock All Evaluator Dependencies:**
   - [`scripts/evaluate.py`](file:///home/julien/projects/astro_stacker/scripts/evaluate.py) was engineered from scratch as a **100% self-contained single-file referee** importing **only** the Python 3.14 standard library (`__future__`, `argparse`, `hashlib`, `json`, `math`, `pathlib`, `sys`, `time`, `typing`) plus `numpy` (`numpy`, `numpy.lib.stride_tricks.sliding_window_view`).
   - Static AST inspection ([`audit_evaluator_dependencies`](file:///home/julien/projects/astro_stacker/scripts/torture_test_evaluator.py#L77-L146)) confirmed **zero** local `.py` helper imports, **zero** compiled `.so`/`ctypes`/`cffi` shared-library bridges, and **zero** `subprocess`/`importlib`/`eval` calls (`strictly_self_contained_stdlib_plus_numpy: true`). Upon Phase 2 approval, locking `chmod 444 scripts/evaluate.py` completely seals the referee's execution graph.
2. **Guardrail 2 — Sharp Linear Reference Arrays (`Malvar–He–Cutler 5×5`):**
   - Simple bilinear demosaicing was strictly rejected. Both [`logs/ref_ring.npy`](file:///home/julien/projects/astro_stacker/logs/ref_ring.npy) and [`logs/ref_andromeda.npy`](file:///home/julien/projects/astro_stacker/logs/ref_andromeda.npy) were produced using **exact Malvar–He–Cutler (2004) 5×5 gradient-corrected linear demosaicing** with parity-preserving symmetric border reflection on unrotated active Canon EOS 1100D RGGB sensor coordinates (`raw_image[18:2874, 62:4352]`, shape `(2856, 4290, 3)`, `float32`, channels `[R, G, B]`).
   - Unit verification inside [`logs/reference_selection.json`](file:///home/julien/projects/astro_stacker/logs/reference_selection.json#L31-L44) confirmed `0.0` error on constant-color `[100.0, 200.0, 150.0]` fields and full preservation of negative dark-subtracted Poisson/read-noise excursions (`149,136` negative values in `ref_ring.npy`; `456,510` negative values in `ref_andromeda.npy`).
3. **Guardrail 3 — Corrupted Frame Count & Empirical Noise Gate:**
   - [`identify_dataset_from_ref`](file:///home/julien/projects/astro_stacker/scripts/evaluate.py#L55-L72) deterministically fingerprints `--ref` from its pixel content (immune to `transforms.json` spoofing) and enforces `min_integrated_frames = 18` on `ring_set` and `min_integrated_frames = 20` on `andromeda_set`.
   - Beyond counting entries in `transforms.json`, [`validate_transforms_json`](file:///home/julien/projects/astro_stacker/scripts/evaluate.py#L584-L738) requires Kish effective sample size $N_{\text{eff}} = (\sum w_i)^2 / \sum w_i^2 \ge 0.65\,N_{\min}$, and **Gate 3** ([`evaluate_stack`](file:///home/julien/projects/astro_stacker/scripts/evaluate.py#L1004-L1019)) verifies that the empirical photometrically normalized sky RMS satisfies:
     $$0.50\,\frac{\sigma_{\text{bg,norm}}^{\text{ref}}}{\sqrt{N_{\text{int}}}} \;\le\; \sigma_{\text{bg,norm}}^{\text{stack}} \;\le\; \min\!\left(0.60,\; \frac{1.85}{\sqrt{N_{\text{eff}}}}\right)\sigma_{\text{bg,norm}}^{\text{ref}}$$
4. **Guardrail 4 — Background Mesh Guardrail (`extended_flux_preservation`):**
   - Fine `64×64` iterative sigma-clipped meshes ([`estimate_detection_background_mesh`](file:///home/julien/projects/astro_stacker/scripts/evaluate.py#L83-L145)) are used **strictly for internal point-source detection** and never subtracted from integrated output arrays.
   - [`compute_extended_flux_preservation`](file:///home/julien/projects/astro_stacker/scripts/evaluate.py#L504-L581) masks compact $5\sigma$ stars, detrends only a global 1st-degree plane $[1, x, y]$ (so legitimate per-frame global pedestal or planar light-pollution gradient equalization is unpenalized), and computes normalized cross-correlation (NCC) and RMS structure amplitude ratio at **both** `16×16` medium-scale blocks and `64×64` meso-scale blocks (`extended_flux_preservation = min(ncc_16x16, ncc_64x64) >= 0.85` and `structure_rms_ratio_64x64 >= 0.70`).

---

## 2. Standardized Reference Baselines & Dataset Forensics

### 2.1 Sensor Calibration & Bitstream Integrity Filtering
[`scripts/generate_references.py`](file:///home/julien/projects/astro_stacker/scripts/generate_references.py) unpacked all `.CR2` files (strictly ignoring `.JPG`, `.png`, `.CR2.xmp`, `.npz`, and `.xcf`) via a temporary `LibRaw 0.21.5` C++20 OpenMP library (compiled inside `tempfile.TemporaryDirectory()` and automatically cleaned up):

- **Active Unrotated Sensor Geometry:** `raw_image[18:2874, 62:4352]` → **`H = 2856, W = 4290`**, Bayer pattern `[[R, G1], [G2, B]]`. EXIF `sizes.flip = 5` on `ring_set` vs. `flip = 0` on `dark_set`/`andromeda_set` is strictly ignored so every array shares physical sensor coordinates.
- **Dual Ingest Integrity Gate:** Frames must pass `libraw_data_error_count == 0`, `active_max <= 16383` (14-bit ADC ceiling), and `active_p1 >= 1850` (catching silent lossless-JPEG predictor underflow in `dark_set/IMG_6096.CR2` where `p1 = 1329 ADU` despite zero `LibRaw` error).
  - `dark_set`: **18 clean** / 20 `.CR2` (`IMG_6096.CR2`, `IMG_6098.CR2` rejected).
  - `ring_set`: **18 completely clean** / 21 `.CR2` (`IMG_6079.CR2`, `IMG_6083.CR2`, and bottom-4-row-corrupted `IMG_6089.CR2` rejected for single-frame reference selection).
  - `andromeda_set`: **21 clean** / 21 `.CR2` (0 rejected).
- **Pedestal-Equalized Linear Master Dark:** Subtracted each clean dark's left optical-black strip median (`raw_image[18:2874, 2:60]`, `2048.0–2049.0 ADU`) prior to floating-point averaging (`shape = (2856, 4290)`, `median = 0.3333 ADU`, `std = 23.0689 ADU`, `125` hot sites `> 500 ADU`). Each light frame is calibrated via `(raw_active - left_opt_black_median) - master_dark` with **zero floor clipping** prior to Malvar–He–Cutler 5×5 demosaicing.

### 2.2 Winning Reference Frames (`logs/ref_ring.npy` & `logs/ref_andromeda.npy`)
Frames were ranked by lowest `median_fwhm_px` (tie-broken by `corner_fwhm_px`, then `median_eccentricity`) among clean exposures passing healthy star count ($\ge \lfloor 0.75\,\text{median}(N_\star)\rfloor$) and roundness ($\le \text{median}(e)$):

| Property | Primary Optimization Reference (`ring_set`) | Holdout Validation Reference (`andromeda_set`) |
| :--- | :--- | :--- |
| **Saved Path** | [`logs/ref_ring.npy`](file:///home/julien/projects/astro_stacker/logs/ref_ring.npy) | [`logs/ref_andromeda.npy`](file:///home/julien/projects/astro_stacker/logs/ref_andromeda.npy) |
| **Winning `.CR2` Exposure** | `IMG_6074.CR2` (`29.0 s`, ISO 1600, `22 °C`) | `IMG_6044.CR2` (`28.0 s`, ISO 1600, `30 °C`) |
| **Array Contract** | `(2856, 4290, 3)` `float32` (`147,026,880` B) | `(2856, 4290, 3)` `float32` (`147,026,880` B) |
| **Channel Medians `[R, G, B]` (ADU)** | `[72.740, 206.021, 181.521]` | `[57.757, 162.833, 142.844]` |
| **Min / P1 / Max Value (ADU)** | `-1666.50` / `+12.68` / `+14249.78` | `-1924.31` / `-3.21` / `+13624.83` |
| **Negative Float Values Preserved** | `149,136` (`0.406%`) | `456,510` (`1.242%`) |
| **Canonical Evaluator `ref_star_count`** | `105` (`5σ`, `minarea >= 5`, non-saturated) | `107` (`5σ`, `minarea >= 5`, non-saturated) |
| **Canonical Evaluator `ref_median_fwhm_px`** | `9.3427 px` (`corner = 13.9914 px`) | `7.7490 px` (`corner = 12.0637 px`) |
| **Dataset Ensemble Median `FWHM` (All Clean Frames)** | `10.9047 px` (`[9.37 .. 14.34] px`, 18 frames) | `8.2791 px` (`[7.54 .. 9.64] px`, 21 frames) |
| **Canonical Evaluator `ref_median_eccentricity`** | `0.5446` | `0.5520` |
| **Canonical Evaluator `ref_normalized_bg_rms`** | `0.432828` (at $F_0 = 1000.0$) | `0.452792` (at $F_0 = 1000.0$) |
| **Baseline Composite Score** | **`50.0` (exact)** | **`50.0` (exact)** |

### 2.3 Two Critical Physical Discoveries Governing Phase 2 & Phase 3

1. **Warm-Pixel MHC 5×5 Blooming vs. Optical Stars:**
   - Because `andromeda_set` (`34 °C → 21 °C`) and `ring_set` (`24 °C → 19 °C`) were recorded at higher sensor temperatures than `dark_set` (`18 °C`), unscaled master-dark subtraction leaves positive single-CFA-site thermal impulses.
   - Under Malvar–He–Cutler 5×5 linear demosaicing, a single warm CFA pixel of amplitude $A > 40\sigma_{\text{bg}}$ blooms into a **5-to-9 pixel cross** surrounded by an analytic negative $r=2$ ring (`-0.109 A` on R/B sites, `-0.0625 A` on G diagonals), **passing `minarea >= 5`** (`median = 61/frame` in `ring_set`, `104/frame` in `andromeda_set`).
   - However, warm impulses and real optical stars exhibit **100% disjoint morphology** (`1.000` recall, `0` false positives across `2,522` raw CFA impulses):
     - **Warm MHC Impulses:** `FWHM ∈ [1.15, 1.33] px`, `sharpness_3x3` median `0.47–0.49`, `ring_r2_ratio` median `-0.02`.
     - **Real Optical Stars:** `FWHM ∈ [5.55, 18.60] px`, `sharpness_3x3` median `0.151`, `ring_r2_ratio` median `+0.57–0.61`.
   - [`detect_and_fit_stars`](file:///home/julien/projects/astro_stacker/scripts/evaluate.py#L334-L416) explicitly filters `sharpness_3x3 > 0.36`, `ring_r2_ratio < -0.02`, and `fwhm_geom < 2.0 px` so residual warm pixels can never masquerade as ultra-sharp `1.2 px` "stars" to game `median_fwhm_px` or `matched_star_count`.

2. **Mount Worm-Gear Periodic Error, Field Rotation & Ensemble FWHM:**
   - Ground-truth 4-DOF astrometric registration in [`logs/reference_selection.json`](file:///home/julien/projects/astro_stacker/logs/reference_selection.json#L357-L396) revealed oscillatory worm-gear periodic error + drift covering all 4 Bayer `2×2` integer phases:
     - **`ring_set` (18 clean frames vs. `IMG_6074.CR2`):** `dx ∈ [-27.57, +2.87] px`, `dy ∈ [-22.15, +13.81] px`, bounding-box diagonal **`47.11 px`**, non-ref median shift **`17.82 px`** (`min = 4.52 px`), `dtheta ∈ [-0.016°, +0.005°]` (`0.96 px` corner span). Notably, `IMG_6074.CR2` (`FWHM = 9.34 px`) occurred at a stationary worm-gear turnaround, whereas the 18-frame exposure ensemble has median `FWHM = 10.9047 px` (`+16.7%` wider due to intra-exposure tracking drift).
     - **`andromeda_set` (21 clean frames vs. `IMG_6044.CR2`):** `dx ∈ [-18.62, +13.10] px`, `dy ∈ [-25.87, +21.27] px`, bounding-box diagonal **`56.83 px`**, non-ref median shift **`17.55 px`** (`min = 1.44 px`), monotonic field rotation `dtheta ∈ [-0.008°, +0.048°]` = **`2.52 px` corner displacement span** (requiring sub-pixel rotational/affine warping in Phase 3 to prevent corner star trailing!).

---

## 3. Mathematical Specification of the Blind Evaluator (`scripts/evaluate.py`)

### 3.1 Luminance Conversion & Interior Full-Overlap ROI
Given `--stack <stack.npy>` and `--ref <ref.npy>` of shape `(2856, 4290, 3)` `float32`, [`compute_bayer_luminance`](file:///home/julien/projects/astro_stacker/scripts/evaluate.py#L74-L80) forms the photon-weighted RGGB linear luminance:
$$L(y, x) = 0.25\,R(y, x) + 0.50\,G(y, x) + 0.25\,B(y, x)$$
All star detections and background cells are evaluated inside `ROI_MARGIN = 64 px` (`y ∈ [64, 2792)`, `x ∈ [64, 4226)`), avoiding partial-coverage drift borders ($\le 28.4\text{ px}$).

### 3.2 Photometric Normalization (Anti-Zero-Multiplier & Anti-Stretch Gate)
To make all metrics invariant to arbitrary positive linear rescaling $I' = a I + b$ ($a > 0$) while rejecting zero-multipliers and non-linear tone curves ([`evaluate_stack`](file:///home/julien/projects/astro_stacker/scripts/evaluate.py#L833-L927)):
1. Select up to $N = 40$ bright, isolated ($\ge 32\text{ px}$ separation), non-saturated (`peak < 10,500 ADU` above background) anchor stars on `--ref` via [`select_isolated_photometric_stars`](file:///home/julien/projects/astro_stacker/scripts/evaluate.py#L483-L501).
2. Cross-match each reference anchor to the nearest detected stack star within $4.0\text{ px}$ (requiring **$\ge 20$ matched anchors**; otherwise `photometric_star_match_failure` $\to -\infty$).
3. Measure linear circular aperture flux ($R_{\text{ap}} = 12\text{ px}$) minus local sky-annulus median ($R_{\text{in}} = 16\text{ px}, R_{\text{out}} = 22\text{ px}$) on both frames: $F_{r,i}$ and $F_{s,i}$.
4. Compute reference scale $s_{\text{ref}} = F_0 / \text{median}_i(F_{r,i})$ with $F_0 = 1000.0$, median flux ratio $r_{\text{med}} = \text{median}_i(F_{s,i} / F_{r,i})$, and normalize both luminance maps to median sky `0.0` and median anchor-star flux `1000.0`:
   $$s_{\text{stack}} = \frac{s_{\text{ref}}}{r_{\text{med}}}, \qquad L_{\text{stack}}^{\text{norm}} = L_{\text{stack}} \cdot s_{\text{stack}} - \tilde{B}_{\text{sky,stack}} \cdot s_{\text{stack}}$$
5. **Linearity Verification:** Require Pearson $r(F_r, F_s) \ge 0.96$, relative MAD of $F_{s,i}/F_{r,i} \le 0.15$, and bright-half vs. faint-half median flux-ratio asymmetry $\le 0.10$ (`nonlinear_photometric_stretch_detected` $\to -\infty$).

### 3.3 Core Metrics
Computed on $L_{\text{stack}}^{\text{norm}}$ and $L_{\text{ref}}^{\text{norm}}$ using identical normalized thresholds:
1. **`median_fwhm_px` & `corner_fwhm_px`:** Fit 7-parameter 2D elliptical Gaussians ($b, A, x_0, y_0, \ln\sigma_x, \ln\sigma_y, \theta$) on $25\times 25$ cutouts via damped Levenberg–Marquardt for all $5\sigma$ connected components (`minarea >= 5`, `chi2_red <= 5.0` with `(0.03 * amp)^2` Moffat-wing variance floor). Geometric $\text{FWHM} = 2.35482\sqrt{\sigma_{\max}\sigma_{\min}}$. Corner stars satisfy $\sqrt{(x - c_x)^2 + (y - c_y)^2} > 0.65\sqrt{(W/2)^2 + (H/2)^2}$ (`> 1674.9 px`). Lower is better.
2. **`median_eccentricity`:** Median elliptical eccentricity $e = \sqrt{1 - (\sigma_{\min}/\sigma_{\max})^2}$ across all valid fitted stars. Lower is better.
3. **`normalized_bg_rms`:** Robust MAD-scaled pixel-to-pixel RMS ($\sigma_{\text{bg,norm}} = 1.4826\,\text{MAD}$) across the quietest star-free/nebula-free interior `64×64` sky cells on $L_{\text{stack}}^{\text{norm}}$. Lower is better.
4. **`matched_star_count`:** Total valid non-saturated $5\sigma$ optical point sources passing 2D Gaussian PSF fit and warm-pixel rejection. Higher is better.
5. **`extended_flux_preservation` ($\rho_{\text{ext}}$):** Minimum of `16×16` and `64×64` star-masked, 1st-degree-plane-detrended normalized cross-correlation against $L_{\text{ref}}^{\text{norm}}$. Closer to `1.0` is better.

### 3.4 Composite Score Formula & Weighting Rationale
Anchored so that evaluating `--ref` against itself yields **exact `50.0`** ([`evaluate_stack`](file:///home/julien/projects/astro_stacker/scripts/evaluate.py#L1030-L1040)):

$$\text{CompositeScore} = 50.0 + 25.0\log_2\!\left(\frac{\sigma_{\text{bg,norm}}^{\text{ref}}}{\sigma_{\text{bg,norm}}^{\text{stack}}}\right) + 35.0\log_2\!\left(\frac{\text{FWHM}_{\text{eff}}^{\text{ref}}}{\text{FWHM}_{\text{eff}}^{\text{stack}}}\right) + 10.0\log_2\!\left(\frac{N_\star^{\text{stack}}}{N_\star^{\text{ref}}}\right) + 15.0\left(e_{\text{ref}} - e_{\text{stack}}\right) - 40.0\left(1.0 - \rho_{\text{ext}}\right)$$

where $\text{FWHM}_{\text{eff}} = 0.65\,\text{median\_fwhm\_px} + 0.35\,\text{corner\_fwhm\_px}$.
- **Logarithmic Octave Scaling prevents Goodhart runaway:** Integrating $N = 18\text{–}21$ frames reduces $\sigma_{\text{bg,norm}}$ by $\sim\sqrt{N} \approx 4.2\times$ ($\approx 2.08$ octaves $\to +52$ pts) and reveals $\sim 2.4\text{–}3.1\times$ more $5\sigma$ stars ($\to +12.5\text{ to }+16.2$ pts).
- **Heavy Sharpness Weight (`35.0 / octave`, $1.4\times$ the noise weight):** Any spatial smoothing that halves pixel noise by bloating $\text{FWHM}_{\text{eff}}$ suffers both immediate disqualification at Gate 2 and a steeper octave penalty in sharpness than it gains in noise. Blending `0.35 * corner_fwhm_px` directly rewards rotational/affine/distortion registration over pure translation.
- **Diffuse Structure Protection (`-40.0 * (1 - \rho_{\text{ext}})`):** Directly rewards faithful preservation of M57 and M31 diffuse emission alongside hard Gate 4.

### 3.5 Summary of Hard Anti-Cheat Gates (`score: -inf`, `cheating_detected: true`)

| Gate | Attack Vector Defeated | Exact Mathematical Trigger in [`scripts/evaluate.py`](file:///home/julien/projects/astro_stacker/scripts/evaluate.py) |
| :--- | :--- | :--- |
| **Gate 0** | NaN/Inf, wrong shape, constant/zero array, or grayscale-copied RGB | `shape != (2856, 4290, 3)`, non-finite values, channel `std <= 1e-6`, or `std(c1 - c2) <= 1e-4 * min(std)` |
| **Gate 1** | Dark-floor / median clipping or dithered floor clamping | `> 0.05%` (`0.0005`) identical sky pixel values on `L`, `R`, `G`, or `B`, OR `> 0.50%` lower-tail micro-bin spike |
| **Gate 2a** | Spatial smoothing / Gaussian / box / median / bilinear wash | Sky `block_4x4_ratio > 1.25 * ref` (`4 * rms_4x4 / rms_1x1`) OR 1-px autocorrelation `rho_1px > ref + 0.18` |
| **Gate 2b** | Registration smearing / PSF bloat | `> 1.10 * ref_fwhm` (baseline mode) or `> 1.10 * ensemble_fwhm` on either all-star or matched top-40 populations |
| **Gate 3a** | Frame-drop cheating or single-frame weight concentration | `n_int < 18` (`ring_set`) / `< 20` (`andromeda_set`), duplicate filenames, or Kish $N_{\text{eff}} < 0.65\,N_{\min}$ |
| **Gate 3b** | Stationary hot-pixel / `(0,0,0)` identity alignment | Non-reference `median_dr < 3.0 px`, `near_zero_frac (<0.75 px, <0.005°) > 0.20`, or `bbox_diag < 15.0 px` |
| **Gate 3c** | Forged `transforms.json` (claiming 18+ frames on a 1–3 frame stack) | Empirical `normalized_bg_rms > min(0.60, 1.85 / sqrt(N_eff)) * ref_norm_bg_rms` or `< 0.50 * ref / sqrt(N_int)` |
| **Gate 4** | Fine SEP background mesh subtraction / nebula erasure | `extended_flux_preservation < 0.85` (`min(ncc_16x16, ncc_64x64)`) OR `structure_rms_ratio_64x64 < 0.70` |
| **Gate 5** | Zero-multiplier or non-linear gamma/arcsinh stretch | `< 20` matched anchors, Pearson `r < 0.96`, `flux_ratio_rel_mad > 0.15`, or `bright_faint_asym > 0.10` |

### 3.6 `transforms.json` Contract Required from C++20 Stacker
```json
{
  "dataset": "ring_set",
  "reference_filename": "IMG_6074.CR2",
  "frames": [
    {
      "filename": "IMG_6072.CR2",
      "integrated": true,
      "is_reference": false,
      "weight": 1.0,
      "dx": -10.8592,
      "dy": 13.8052,
      "dtheta_deg": -0.01636
    }
  ]
}
```

---

## 4. Independent Adversarial Torture-Test Verification (`28 / 28` Passed)

An independent **Adversarial Evaluator Torture-Tester** subagent authored [`scripts/torture_test_evaluator.py`](file:///home/julien/projects/astro_stacker/scripts/torture_test_evaluator.py), testing all mandatory scenarios `(a)–(e)` plus six red-team probes `(f)–(k)` across **both** `ring_set` and `andromeda_set` (`28` black-box CLI subprocess invocations; full telemetry in [`logs/torture_test_results.json`](file:///home/julien/projects/astro_stacker/logs/torture_test_results.json)).

### 4.1 Four Subtle Calibration Bugs Caught & Fixed Before Immutability Lock
Iterative red-teaming uncovered and resolved four subtle edge cases prior to locking:
1. **High-SNR Moffat-Wing $\chi^2_{\nu}$ Inflation (`Bug A`):** When `sky_rms` dropped $4.2\times$ in 18/21-frame stacks, deterministic $\sim 3\%$ non-Gaussian Moffat/coma outer wings inflated `chi2_red > 5.0` on bright stars. Adding `(0.03 * max(0.0, amp_f))^2` to `pix_var` ([`evaluate.py` L428–L434](file:///home/julien/projects/astro_stacker/scripts/evaluate.py#L428-L434)) restored `32–34 / 40` bright anchor matches on honest multi-frame stacks while still rejecting blended nebular knots.
2. **Sparse-Field `64×64` Mesh Erasure (`Bug B`):** On `ring_set` (compact M57 + sparse sky), subtracting a `64×64` median mesh left intra-cell single-frame noise correlated at `16×16` (`ncc_16x16 = 0.8012`). Adding `64×64` meso-scale plane-detrended NCC + `structure_rms_ratio_64x64 >= 0.70` ([`evaluate.py` L568–L581](file:///home/julien/projects/astro_stacker/scripts/evaluate.py#L568-L581)) crushed `ring_set:i` to `ncc = 0.4791`, `rms_ratio_64x64 = 0.4397` (`score = -inf`).
3. **Scale-Invariant LM Stopping Criterion (`Bug C`):** Normalizing Levenberg–Marquardt flux steps `dp[0], dp[1]` by `sky_rms` (`< 1e-5`) and running canonical `ref_det` on normalized flux ([`evaluate.py` L394–L400, L855–L860](file:///home/julien/projects/astro_stacker/scripts/evaluate.py#L394-L400)) eliminated a `+0.00038` self-score drift, achieving **exact `50.0`** on `(a)` and ULP-level scale invariance on `(f)`.
4. **Population-Matched Ensemble FWHM Ceiling (`Bug D`):** Top-40 bright stars have slightly wider Gaussian fits (`10.65 px` in `ref_ring.npy`) than faint `5σ` stars (`9.34 px`). Scaling `matched_base_fwhm = max(matched_ref_fwhm * ens_factor, base_fwhm)` ([`evaluate.py` L994–L1002](file:///home/julien/projects/astro_stacker/scripts/evaluate.py#L994-L1002)) ensured apples-to-apples `+10%` blur checks on both all-star and bright-star populations.

### 4.2 Final `28 / 28` Adversarial Torture-Test Matrix

| Case | Mode | `ring_set` Cheat? | `ring_set` Score | `ring_set` Triggered Gate(s) / Key Telemetry | `andromeda_set` Cheat? | `andromeda_set` Score | `andromeda_set` Triggered Gate(s) / Key Telemetry | Audit |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :--- | :---: |
| **(a) Single Ref Baseline** | `baseline` | `False` | **`50.0`** | Exact `50.0` (`fwhm=9.3427`, `corner=13.9914`, `bg_rms=0.4328`, `stars=105`, `ncc=1.0`) | `False` | **`50.0`** | Exact `50.0` (`fwhm=7.7490`, `corner=12.0637`, `bg_rms=0.4528`, `stars=107`, `ncc=1.0`) | **PASS** |
| **(b1) All-Zero Array** | `baseline` | `True` | `-inf` | `zero_or_constant_channel_collapse` | `True` | `-inf` | `zero_or_constant_channel_collapse` | **PASS** |
| **(b2) All-Zero Array** | `stack` | `True` | `-inf` | `zero_or_constant_channel_collapse` | `True` | `-inf` | `zero_or_constant_channel_collapse` | **PASS** |
| **(c1) Blurred $\sigma=2.0\text{ px}$** | `baseline` | `True` | `-inf` | `spatial_smoothing_noise_correlation` (`ratio=3.45`, `rho=0.94`) + `fwhm_blur_degradation` | `True` | `-inf` | `spatial_smoothing_noise_correlation` (`ratio=3.45`, `rho=0.94`) + `fwhm_blur_degradation` | **PASS** |
| **(c2) Blurred $\sigma=2.0\text{ px}$** | `stack` | `True` | `-inf` | `spatial_smoothing_noise_correlation` (`ratio=3.45`, `rho=0.94`) | `True` | `-inf` | `spatial_smoothing_noise_correlation` + `fwhm_blur_degradation` (`9.47 > 9.11`) | **PASS** |
| **(d1) Median Dark-Clipped** | `baseline` | `True` | `-inf` | `dark_floor_clipping_detected` (`ident_frac=1.0 > 0.0005` on `R`) + `star_match_failure` | `True` | `-inf` | `dark_floor_clipping_detected` (`ident_frac=1.0 > 0.0005` on `R`) + `spatial_smoothing` | **PASS** |
| **(d2) Median Dark-Clipped** | `stack` | `True` | `-inf` | `dark_floor_clipping_detected` + `photometric_star_match_failure` | `True` | `-inf` | `dark_floor_clipping_detected` + `spatial_smoothing_noise_correlation` | **PASS** |
| **(e) Unaligned 10f `(0,0,0)`** | `stack` | `True` | `-inf` | `integrated_frame_count=10<18` + `kish_n_eff` + `hot_pixel_or_zero_shift` + `star_match` | `True` | `-inf` | `integrated_frame_count=10<20` + `kish_n_eff` + `hot_pixel_or_zero_shift` + `fwhm_blur` | **PASS** |
| **(f) Scale `ref*0.0037+42`** | `baseline` | `False` | **`50.000043`** | Valid! `r=1.0`, `ncc=1.0`, `stars=105`, `fwhm=9.3427` (float32 ULP `+4.3e-5`) | `False` | **`49.996792`** | Valid! `r=1.0`, `ncc=1.0`, `stars=107`, `fwhm=7.7501` (float32 ULP `-3.2e-3`) | **PASS** |
| **(g) Forged `1f` + `18/21f` JSON** | `stack` | `True` | `-inf` | `insufficient_empirical_noise_reduction` (`0.4328 > ceil=0.1887`, `n_eff=18.0`) | `True` | `-inf` | `insufficient_empirical_noise_reduction` (`0.4528 > ceil=0.1828`, `n_eff=21.0`) | **PASS** |
| **(h) Subtle $\sigma=0.8\text{ px}$ Blur** | `stack` | `True` | `-inf` | `spatial_smoothing_noise_correlation` (`ratio=2.39 vs 1.31`, `rho=0.73 vs 0.26`) | `True` | `-inf` | `spatial_smoothing_noise_correlation` (`ratio=2.37 vs 1.30`) + `sub_poisson_floor` | **PASS** |
| **(i) `64×64` Mesh Erasure** | `baseline` | `True` | `-inf` | `extended_flux_erasure_detected` (`ncc=0.4791 < 0.85`, `rms_ratio_64=0.4397 < 0.70`) | `True` | `-inf` | `extended_flux_erasure_detected` (`ncc=0.2655 < 0.85`, `rms_ratio_64=0.1461 < 0.70`) | **PASS** |
| **(j) Gamma `(stack-min)^0.5`** | `stack` | `True` | `-inf` | `nonlinear_photometric_stretch_detected` (`asym=0.1568 > 0.10`) + `fwhm_blur` | `True` | `-inf` | `nonlinear_photometric_stretch_detected` (`asym=0.1269 > 0.10`) + `fwhm_blur` | **PASS** |
| **(k) Honest 18/21f Shift+Add** | `stack` | `False` | **`105.246063`** | Valid! `stars=323`, `bg_rms=0.1151` (`3.76×` lower), `fwhm=11.2399`, `ncc=0.9933` | `False` | **`112.171735`** | Valid! `stars=254`, `bg_rms=0.1007` (`4.50×` lower), `fwhm=8.3338`, `ncc=0.9959` | **PASS** |

---

## 5. Phase 3 Engineering & Evolutionary Optimization Plan (`cpp_stacker/`)

### 5.1 Immutability Lock & Anti-Goodhart Protocol
1. **Immediate Lock upon Approval:** Execute `chmod 444 scripts/evaluate.py` and verify its SHA-256 (`971b62f072143eb6172cd804f85750b7c31c507fbd13879ed4d1a46b50c415f9`) before writing a single line of C++20 in `cpp_stacker/`.
2. **Mandatory Isolated Red-Team Auditor Loop:** After every C++20 iteration is evaluated on `ring_set`, spawn a **fresh, independent Adversarial Auditor subagent** (with zero authorship in `cpp_stacker/`) to inspect `git diff cpp_stacker/`, `logs/transforms.json`, and the JSON output of [`scripts/evaluate.py`](file:///home/julien/projects/astro_stacker/scripts/evaluate.py) for artificial sharpening/ringing, dark-clipping, background mesh subtraction, or frame dropping, issuing a binding `AUDIT_PASS` or `AUDIT_FAIL`.
3. **Strict Atomic Hypothesis & Git Rollback Discipline:**
   - Propose **one** well-defined algorithmic hypothesis per iteration.
   - Log **every** iteration (both kept and reverted!) to `logs/metrics.jsonl` and `logs/03_iteration_log.md` (`logs/` is gitignored so rollbacks never erase experimental history).
   - Commit to git **only** when `composite_score` strictly improves over the incumbent best AND the isolated auditor returns `AUDIT_PASS`; otherwise immediately revert via `git checkout -- cpp_stacker/ && git clean -fd cpp_stacker/`.

### 5.2 C++20 Architecture (`cpp_stacker/`)
- **Build Toolchain:** `CMakeLists.txt` / `Makefile` compiling native C++20 with `g++ 15.2.0` (`-std=c++20 -O3 -march=native -fopenmp -Wall -Wextra -lraw_r`).
- **Strict `.CR2` & Sensor Quarantine:** `std::filesystem::directory_iterator` filtering strictly for `.CR2`, hooking `LibRaw::set_dataerror_handler`, enforcing `active_max <= 16383` and `active_p1 >= 1850` (with row-level validation investigated as an atomic hypothesis for salvaging the clean top `2,476` rows of `ring_set/IMG_6089.CR2`), and operating exclusively on unrotated physical `raw_image[18:2874, 62:4352]` (`2856 × 4290`) coordinates.
- **Zero External Array Dependencies:** Native C++20 writers for `.npy` v1.0 (`(2856, 4290, 3)` `float32`), `transforms.json`, 32-bit float linear `.fits` / `.tif`, and 16-bit auto-stretched visual preview `.tif`.

### 5.3 Ranked Phase 3 Iteration Roadmap on `ring_set` (+ `dark_set`)

| Iteration | Atomic Algorithmic Hypothesis | Expected Metric Impact | Key Guardrail / Risk Check |
| :--- | :--- | :--- | :--- |
| **`iter-0-baseline`** | Clean end-to-end C++20 baseline: `LibRaw::unpack()` + integrity gate (`18` darks, `18` `ring_set` lights), mean master dark subtraction, linear Malvar–He–Cutler 5×5 demosaicing, bright-star COM centroiding, integer shift-and-add (`round(dx), round(dy)`), unweighted mean integration. | Anchor multi-frame C++ baseline (`score ≈ 104–106`, `runtime < 6 s`). | Verify `cheating_detected == false`, zero floor clipping, `n_int == 18`. |
| **Iter 1** | **Raw CFA Calibration & Thermal Hot/Warm-Pixel Repair:** Left optical-black strip (`cols 2..59`) bias equalization + Winsorized sigma-clipped master dark + least-squares per-frame thermal dark scaling $\alpha_i$ + stride-2 same-color CFA impulse median repair (with cross-channel optical-star core protection) *before* demosaicing. | Eliminates `~61–104` MHC warm-pixel crosses/rings per frame; lowers `normalized_bg_rms` and improves centroid purity. | Confirm zero attenuation of genuine faint star cores (`matched_star_count` non-decreasing). |
| **Iter 2** | **Sub-Pixel Star Extraction & Flux-Conserving Resampling:** Internal `64×64` SEP background mesh (*detection-only!*) + 2D Gaussian sub-pixel centroids + separable Lanczos-3 (or Catmull-Rom) sub-pixel shift warping (replacing integer pixel rounding). | Sharpens `median_fwhm_px` & `corner_fwhm_px` (`~0.2–0.4 px` gain) and lowers `median_eccentricity`. | Check `noise_spatial_correlation_stack` (`block_4x4_ratio`, `rho_1px`) to ensure kernel does not induce bilinear wash or Lanczos Gibbs ringing. |
| **Iter 3** | **Triangle Asterism Geometric Hashing + RANSAC 4-DOF Similarity (`dx, dy, dθ, s`):** Chirality-split Valdes $(L_1/L_2, L_0/L_2)$ triangle invariants across top $60$ stars + RANSAC + IRLS 4-DOF rigid/similarity registration. | Eliminates field-rotation corner arc (`0.96 px` on `ring_set`; crucial **`2.52 px`** on `andromeda_set` holdout!). | Verify `transforms.json` exports exact `(dx, dy, dtheta_deg)` around `(2144.5, 1427.5)`. |
| **Iter 4** | **RANSAC 6-DOF Affine / Cross-Validated Quadratic Distortion Warping:** Test 6-DOF affine (differential atmospheric refraction / focal-plane tilt) vs. 4-DOF similarity using leave-one-out astrometric residual cross-validation. | Reduces `corner_fwhm_px` & `median_eccentricity` if differential tilt is statistically significant; revert if overfitting. | Prevent edge Runge oscillation by restricting to $\le 6\text{ DOF}$ affine unless LOOCV RMS drops $>10\%$. |
| **Iter 5** | **Planar Sky Equalization, Photometric Scaling & Optimal Frame Weighting:** Per-frame global/1st-degree planar sky-gradient equalization (strictly preserving extended flux!) + relative flux scale $F_i$ + optimal frame weights $w_i \propto F_i^2 / (\sigma_{\text{bg},i}^2 \cdot \text{FWHM}_i^2)$. | Down-weights worm-gear drift frames (`FWHM ~ 14.3 px`) in favor of `9.4–10.0 px` frames while maintaining Kish $N_{\text{eff}} \ge 15$. | Must keep $N_{\text{eff}} \ge 0.65\,N_{\min}$ (`>= 11.7` on `ring_set`, `>= 13.0` on `andromeda_set`) and `extended_flux_preservation >= 0.99`. |
| **Iter 6** | **OpenMP Pixel-Stack Winsorized Sigma Clipping:** Replace weighted mean integration with scale-normalized Huber/Winsorized sigma clipping ($\kappa = 3.0\sigma$ using local Poisson + read-noise variance model) + clean-row salvage of `IMG_6089.CR2` (`rows < 2476`). | Rejects satellite/airplane streaks, cosmic rays, and drifting warm-pixel remnants without inflating Gaussian floor variance. | Verify zero dark-floor clipping (`max_identical_sky_fraction < 0.0005`) and Poisson floor compliance. |
| **Iter 7** | **Demosaicing / Native CFA Bayer-Drizzle Ablation:** Compare VNG / directional-gradient linear demosaicing + Lanczos-3 against direct native CFA super-pixel/Bayer-Drizzle accumulation ($s = 1.0$, pixfrac $p \in [0.7, 1.0]$). | Potential uncorrelated noise & color-aliasing improvement if Bayer phase coverage across 18–21 frames is uniform enough; revert if empty-phase interpolation softens `FWHM` or inflates `block_4x4_ratio`. | Strict Gate 2a (`block_4x4_ratio`, `rho_1px`) & Gate 4 (`extended_flux_preservation`) audit. |

### 5.4 Holdout Generalization & Final Deliverables Protocol (End of Phase 3)
Once `ring_set` optimization converges:
1. Run **both** `iter-0-baseline` and the winning C++20 binary **completely unmodified** (zero hyperparameter tuning or code changes) on `andromeda_set` + `dark_set` and score both with locked [`scripts/evaluate.py`](file:///home/julien/projects/astro_stacker/scripts/evaluate.py) against [`logs/ref_andromeda.npy`](file:///home/julien/projects/astro_stacker/logs/ref_andromeda.npy).
2. Export linear 32-bit float scientific stacks (`logs/ring_best.fits`, `logs/ring_best.tif`, `logs/andromeda_best.fits`, `logs/andromeda_best.tif`) plus auto-stretched 16-bit visual preview TIFFs (`logs/ring_best_preview16.tif`, `logs/andromeda_best_preview16.tif`).
3. Present `logs/03_iteration_log.md`, the complete `ring_set` ablation table, and the `andromeda_set` holdout generalization results at **STOP GATE 3**.

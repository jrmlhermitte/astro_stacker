# Phase 1 Report: Legacy 2020 Audit, Dataset Forensics & C++20 Algorithmic Roadmap

## 1. Executive Summary & System Verification

Phase 1 audited the 2020 Python codebase (`./legacy_2020/`), profiled all `.CR2` raw frames across `dark_set`, `ring_set`, and `andromeda_set` via `scripts/inspect_cr2_dataset.py` (`logs/dataset_manifest.json`), verified the C++20 / OpenMP / `LibRaw` build toolchain, and synthesized academic astrophotography literature into a ranked C++20 evolutionary roadmap.

### Toolchain & Environment Verification
- **C++20 Compiler & Parallelism:** `g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0` (`-std=c++20 -O3 -march=native -fopenmp`), 16 logical OpenMP threads verified.
- **Raw Decoding Library:** `libraw-dev` `0.21.5-Release` (`<libraw/libraw.h>`, `-lraw_r`) compiles cleanly and was empirically verified bit-for-bit deterministic across 1-thread and 8-thread parallel runs.
- **Python Referee Environment (`python3`):** Python `3.14.4` with `numpy 2.3.5`, `Pillow 12.1.1`, `joblib 1.5.2`, `pytest 9.0.2`. *(Note: `scipy`, `rawpy`, `tifffile`, and `astropy` are not installed in system `python3`; our Python profiling/evaluation scripts use vectorized `numpy` plus a lightweight C++20 `LibRaw` `ctypes` bridge so zero external package installs or legacy Poetry environments are required.)*

---

## 2. Static Technical Audit of the 2020 Python Codebase (`./legacy_2020/`)

### 2.1 Architecture & Workflow Overview
Located in `./legacy_2020/` (`pyproject.toml`, `tasks.py`, `astro_stacker/`), the 2020 codebase targeted Python `^3.8` (`numpy`, `rawpy`, `scikit-image`, `scipy`, `lmfit`, `tqdm`) for interactive Jupyter/IPython exploration rather than an end-to-end CLI stacker (`tasks.py` defines only QA wrappers for `pylint`, `mypy`, `flake8`, and `pre-commit`).

| Module (`./legacy_2020/astro_stacker/`) | Key Classes / Functions | Intended Role & Critical Implementation Defects |
| :--- | :--- | :--- |
| `image_readers/cannon_image_reader.py` | `CannonImageReader` (`L18–42`) | Reads `.cr2` files via `rawpy.imread(filepath).postprocess()` (`L39`). Uses unsorted `os.listdir()` (`L29`) and never applies `region_of_interest` (`L24`). |
| `models/image.py` | `Image` (`L14–83`), `_sgsmooth_3d_image` (`L86–94`) | Wraps `(H, W, 3)` RGB array, `mask`, and `dark`. `get_data()` (`L77–83`) performs `(data - dark) * mask`, but downstream correlators bypass `get_data()`. `smooth()` (`L63–67`) runs cubic Savitzky–Golay on `mask.astype(int)` and casts `.astype(bool)`, turning almost all `False` entries into `True` via polynomial ringing. |
| `processing/image_analyzer.py` | `ImageAnalyzer.analyze` (`L7–26`) | Accumulates pixel sums and sums-of-squares across frames to compute `AnalyzedImage(average, variance)`. Suffers from **`uint8` modular wrap-around overflow** and a broken variance formula (`L22`). |
| `processing/variance_analyzer.py` | `VarianceAnalyzer.generate_mask` (`L12–21`) | Flags pixels above the `0.9999` per-channel variance quantile (`L5`) as `False` (`0`) in a keep-mask. |
| `processing/spatial_correlator.py` | `SpatialCorrelator.correlate` (`L7–44`) | Full-frame 2D FFT cross-correlation (`scipy.signal.fftconvolve`, `mode="full"`) on greyscale-averaged images with scalar mean subtraction (`bgsub=True`). |
| `processing/difference_correlator.py` | `DifferenceCorrelator` (`L30–149`) | Brute-force `[-40, 40] × [-40, 40]` shift grid (`np.roll`) minimizing $L_\infty$ (`np.max(np.abs(...))`) difference (`L142`). |
| `curve_fitters/gaussian_fitter.py` & `functions/stats.py` | `GaussianCurveFitter` (`L11–39`), `center_of_mass`, `sigma` | 1D `lmfit` Gaussian peak fitter. `sigma()` (`stats.py:17–18`) divides `x_1**2` by `np.sum(y)` twice and yields `NaN` when zero-mean `bgsub=True` makes $\sum y_i \le 0$. |
| `processing/shift_analyzer.py` | `Shift(xshift: int, yshift: int)`, `ShiftAnalyzer.analyze` | Empty `pass` stub (`L14`). **No image warping or frame-stacking module was ever completed.** |

### 2.2 Concrete Code Bugs Compounding Algorithmic Struggles
1. **Consumer sRGB `postprocess()` Destruction (`cannon_image_reader.py:39`):** Calling `rawpy.imread(filepath).postprocess()` with default arguments applies (a) **8-bit quantization (`dtype=np.uint8`)**, discarding 6 bits ($64\times$) of Canon's 14-bit ADC precision; (b) **non-linear BT.709/sRGB gamma (`2.222`)**, violating dark-subtraction linearity $(S + D)^{1/\gamma} - D^{1/\gamma} \neq S^{1/\gamma}$; (c) **per-frame auto-brightening (`no_auto_bright=False`)**, boosting dark-frame read noise to `255`; and (d) **AHD demosaicing prior to calibration**, smearing single hot pixels into $5\times 5$ polychromatic blobs.
2. **`uint8` Modular Wrap-Around & Variance Typo (`image_analyzer.py:14–22`):** `final_image = image.data` initializes accumulators as `uint8`, so `final_image += image.data` and `image.data ** 2` wrap modulo `256` on every frame! In addition, `L22` omits the square on `data_average`: $\frac{1}{N+1}\sum x_n^2 - \frac{N}{N+1}\bar{x}$ instead of $\frac{1}{N-1}\left(\sum x_n^2 - N\bar{x}^2\right)$.
3. **Silent Bypass of Dark Subtraction & Masking (`spatial_correlator.py:14, 19, 28`):** `SpatialCorrelator.correlate` reads `image_1.data` directly instead of calling `image_1.get_data()`, completely skipping `self.dark` subtraction! Furthermore, `convolution_grey = cross_correlate_2d(data_1_grey, data_2_grey)` (`L28–29`) cross-correlates **unmasked** pixel intensities while dividing by `cross_correlate_2d(mask_1, mask_2)` (`L35`), leaving hot pixels at full amplitude in the numerator. When `symmetric_averaging=True` (`L40–43`) is combined with `bgsub=True`, the zero-mean denominator factors $L(\mathbf{k})\,R(\mathbf{k})$ cross zero and create poles across the lag plane.

### 2.3 Physical & Mathematical Root-Cause Failure Analysis
Even if all surface bugs above were fixed in float32, full-frame spatial cross-correlation
$$C_{\mathbf{k}} = \frac{1}{N_{\mathbf{k}}} \sum_{\mathbf{x}} M_i(\mathbf{x})\,M_j(\mathbf{x}+\mathbf{k})\,\big(\delta\eta_i(\mathbf{x}) + s_i(\mathbf{x})\big)\big(\delta\eta_j(\mathbf{x}+\mathbf{k}) + s_j(\mathbf{x}+\mathbf{k})\big)$$
fails fundamentally on prime-focus / unguided DSLR frames for five physical reasons:
1. **Coherent Finite Master-Dark Residual Spike at $\mathbf{k} = \mathbf{0}$:** Subtracting an $N_D$-frame master dark $\tilde{\bar{\eta}}(\mathbf{x}) = \bar{\eta}(\mathbf{x}) + \epsilon_D(\mathbf{x})$ leaves the **exact same static error** $r(\mathbf{x}) = -\epsilon_D(\mathbf{x}) + \Delta\bar{\eta}_{\text{thermal}}(\mathbf{x})$ imprinted on *every* calibrated light frame. At $\mathbf{k} = \mathbf{0}$, all $N_{\text{pix}} = 1.225 \times 10^7$ pixels add strictly positively in phase ($\sum_{\mathbf{x}} r(\mathbf{x})^2 \propto N_{\text{pix}} \sigma_D^2 / N_D$), **and ensemble-averaging $\langle C_{\mathbf{k}} \rangle$ across frame pairs cannot attenuate it** because every light frame shares the same subtracted master dark!
2. **Stationary Vignetting & Skyglow Pedestal:** Subtracting a scalar frame mean $\bar{I}$ leaves the radially symmetric telescope vignetting bowl $V(\mathbf{x})B_{\text{sky}} - \overline{VB_{\text{sky}}}$ stationary in sensor coordinates across 100% of pixels, creating a massive broad correlation hill centered at $\mathbf{k} = \mathbf{0}$.
3. **Extreme Point-Source Sparsity (<0.5% Occupancy):** Stars occupy $<0.5\%$ of pixels; $>99.5\%$ of pixels contribute only sky shot noise, read noise, and sensor FPN to the global inner product.
4. **Non-Uniform Drift & Cadence Jitter:** Varying inter-frame gaps (e.g., `46s` and `75s` initial gaps vs `31s` median cadence observed in `dataset_manifest.json`, plus periodic tracking error) smear the stellar correlation peak across lags during ensemble averaging while the false sensor-locked peak stays razor-sharp at $\mathbf{k} = (0, 0)$.
5. **Field Rotation & Optical Distortion:** Rigid 2D translation $(dx, dy)$ cannot represent prime-focus field rotation $d\theta$ or off-axis coma/distortion across a $4290 \times 2856$ APS-C sensor.

---

## 3. Dataset Forensics & Calibration Diagnostics (`logs/dataset_manifest.json`)

`scripts/inspect_cr2_dataset.py` inspected `pictures/2020-11-28-deep-sky-ring-andromeda/` in **11.41s** using our `LibRaw` C++20 bridge, strictly filtering for `.CR2` files and ignoring all 66 non-`.CR2` files.

### 3.1 File Inventory & Strict `.CR2` Quarantine Verification
| Dataset Directory | Total Files | `.CR2` Inspected | Clean `.CR2` | Corrupted `.CR2` | Ignored Non-`.CR2` | Ignored Breakdown (By Filename Only) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `andromeda_set` (Holdout Test) | 44 | **21** | **21** | **0** | 23 | `21 .JPG`, `1 .CR2.xmp`, `1 .xcf` |
| `ring_set` (Primary Train) | 43 | **21** | **18** | **3** | 22 | `21 .JPG`, `1 .CR2.xmp` |
| `dark_set` (Calibration) | 41 | **20** | **18** | **2** | 21 | `20 .JPG`, `1 .npz` |
| **Total** | **128** | **62** | **57** | **5** | **66** | — |

### 3.2 Sensor Geometry, Acquisition Match & Pixel Statistics (Clean Frames)
All 62 `.CR2` files originate from the **exact same camera body and continuous 36.58-minute observing session**:
- **Camera & ADC:** `Canon EOS 1100D` (`BodySerial: 162063089193`), **ISO 1600**, **14-bit raw ADC** (`raw_bps = 14`).
- **Raw Sensor Layout:** `raw_width = 4352`, `raw_height = 2874`; **Active Visible Area:** `width = 4290`, `height = 2856` (`12,252,240` pixels) starting at even offsets `top_margin = 18`, `left_margin = 62` (preserving a 62-column left optical-black strip for per-frame/per-row bias tracking).
- **CFA Pattern:** `RGGB` (`[[R, G1], [G2, B]]`, `LibRaw` indices `[[0, 1], [3, 2]]`) in both raw and visible sensor coordinates.
- **Hardware Pedestal & Saturation:** Nominal black level `2048` ADU (`left_margin` optical black median `2047–2049` ADU); `LibRaw` linear `maximum = 13584` ADU (`NormalWhiteLevel = 14582`, `SpecularWhiteLevel = 15094`, empirical saturated star-core clip `15305–15311` ADU).

| Metric (Active Sensor Area, Clean Frames) | `andromeda_set` (M31, 1st) | `ring_set` (M57, 2nd) | `dark_set` (Capped Body, 3rd) |
| :--- | :--- | :--- | :--- |
| **Chronological Window (Elapsed)** | `t = 0s .. 693s` (`IMG_6041–6061`) | `t = 881s .. 1545s` (`IMG_6072–6092`) | `t = 1577s .. 2195s` (`IMG_6093–6112`) |
| **Optics / Attachment** | Prime-focus T-ring (`f/0`) | Prime-focus T-ring (`f/0`) | Capped `EF-S 18-55mm` (`f/4.5`) |
| **Clean `.CR2` Count / Exposure** | `21` (`2 × 29.0s` + `19 × 28.0s` = `590s`) | `18` (`18 × 29.0s` = `522s`) | `18` (`18 × 29.0s` = `522s`) |
| **EXIF Gravity `flip` Tag** | `0` (`0°` unrotated) | **`5` (`90° CCW` rotated!)** | `0` (`0°` unrotated) |
| **MakerNotes `CameraTemperature`** | **`34.0 °C → 21.0 °C`** (mean `25.4 °C`) | **`24.0 °C → 19.0 °C`** (mean `20.4 °C`) | **`18.0 °C` steady** (mean `18.0 °C`) |
| **Sample Frame `[p0, p1, p50, p99, p99.9, p100]`** | `[1550, 2047, 2187, 2364, 2603, 15311]` | `[1775, 2074, 2226, 2360, 2473, 15305]` | `[1637, 2011, 2049, 2090, 2173, 15305]` |
| **Mean Frame `std` (ADU)** | `76.8` (`85.8` at 34 °C → `72.6` at 21 °C) | `82.2` (`80.2 .. 85.0`) | `28.38` (`28.18 .. 28.79`) |
| **Per-CFA Medians `(R, G1, G2, B)`** | `(2105.3, 2211.1, 2209.7, 2189.4)` | `(2122.0, 2257.4, 2256.3, 2231.9)` | `(2049.4, 2049.1, 2048.3, 2048.0)` |
| **Sky Signal Above Black `(R, G1, G2, B)`** | `(+57.2, +163.0, +161.6, +141.6)` ADU | `(+73.8, +209.2, +208.0, +184.2)` ADU | `(+0.9, +0.3, +0.1, +0.0)` ADU |

### 3.3 Three Newly Discovered Physical/Data Hazards (Mandatory Pipeline Safeguards)
> [!WARNING]
> Static metadata inspection uncovered **three severe real-world hazards** in this dataset that would silently corrupt any naive stacker (and directly contributed to the 2020 failures):

1. **5 Corrupted `.CR2` Bitstreams on Disk (Canon Lossless-JPEG Predictor Corruption):**
   - `dark_set/IMG_6098.CR2`: `LibRaw` `data_error` at byte `9,945,668` — 100% of rows wrap to 16-bit `65535` (`p99 = 65426`, `std = 14,476.5`).
   - `dark_set/IMG_6096.CR2`: **Silent lossless-JPEG predictor underflow** starting at row `1956` (`681/2856` rows = 23.8% corrupted down to `743` ADU, `p1 = 1329`, `std = 130.1` vs `28.4` on clean darks — **`LibRaw` reports zero `data_error`s!**).
   - `ring_set/IMG_6079.CR2` & `IMG_6083.CR2`: `LibRaw` `data_error`s at bytes `7,103,945` and `12,325,909` (`100%` and `72.2%` of rows corrupted to `65535`).
   - `ring_set/IMG_6089.CR2`: `LibRaw` `data_error` at byte `15,098,935` affecting only the bottom `4` rows (`row >= 2476`, `0.14%` of frame; top `2476` rows are completely clean!).
   - **Pipeline Safeguard:** Every loader (`Phase 2` reference extractor and `Phase 3` C++20 engine) MUST install a `LibRaw::set_dataerror_handler` callback **AND** validate raw 14-bit physical sensor bounds (`max <= 16383` ADC ceiling and row/frame `p1 > 1850` sub-bias underflow check) to reject corrupted frames (or mask corrupted bottom rows in `IMG_6089.CR2`).
2. **EXIF Gravity `flip = 5` Orientation Mismatch (`ring_set` vs `dark_set`):**
   - Pointing near zenith for M57 tripped the camera's orientation switch (`flip = 5`, 90° CCW) on all `ring_set` frames, whereas `dark_set` and `andromeda_set` have `flip = 0`. Any reader that applies EXIF orientation before calibration (like `rawpy.postprocess()`) transposes `ring_set` to `2856 × 4290` against a `4290 × 2856` dark!
   - **Pipeline Safeguard:** Perform all dark/bias subtraction and CFA bad-pixel repair directly on unrotated physical `imgdata.rawdata.raw_image` coordinates (`[18:2874, 62:4352]` $\to$ `2856 × 4290` `RGGB`), ignoring `sizes.flip`.
3. **Outdoor Thermal Cooling (`34 °C → 18 °C`) & Minor Bulb Exposure Drift (`28s` vs `29s`):**
   - Because the DSLR cooled continuously outdoors from `34 °C` (`andromeda_set` start) through `24–19 °C` (`ring_set`) down to `18 °C` (`dark_set`), thermal dark current in early frames is $2\times\text{–}4\times$ higher than in the `18 °C` master dark.
   - **Pipeline Safeguard:** Track per-frame/per-row bias via the 62-column left optical-black margin, apply **optimal dark-current scaling ($\alpha_i$)** of the zero-pedestal master dark thermal pattern, and repair residual hot/warm pixels on the raw CFA planes prior to demosaicing.

---

## 4. Literature Synthesis: Modern Astrophotography Stacking & Anti-Goodhart Design

1. **Raw Bayer CFA Calibration & Demosaicing:**
   - **Master Dark Integration:** Pure median stacking incurs a $\pi/2 \approx 1.571$ variance penalty (+25.3% noise); **Winsorized Sigma Clipping** (Huber 1981; `c_H = 1.5`, consistency factor `1.134`) achieves ~98% Gaussian efficiency while rejecting cosmic rays in dark frames.
   - **Pre-Demosaic Stride-2 CFA Cosmetic Correction:** Hot/warm pixels surviving thermal mismatch must be detected on the 4 raw Bayer sub-planes independently (comparing each pixel to its 8 same-color stride-2 neighbors $N_8^{(2)}$, guarded by a stride-1 cross-color star-core protection ratio so sharp stellar peaks are never clipped) *before* demosaicing blooms them into $5\times 5$ RGB artifacts.
   - **Linear Demosaicing vs. Bayer-Drizzle:** Bilinear CFA interpolation bloats stellar FWHM by +15–25%. **Malvar–He–Cutler (2004)** $5\times 5$ gradient-corrected linear filters and **VNG (Chang et al. 1999)** preserve sharp stellar PSFs with strict linearity, while **Bayer-Drizzle (Fruchter & Hook 2002)** forward-maps un-interpolated raw CFA pixels directly onto the sky grid when unguided/dithered sub-pixel drift samples all 4 Bayer phases across $\ge 18$ frames.
2. **Point-Source Extraction (SExtractor / SEP Paradigm):**
   - Following **Bertin & Arnouts (1996)** and **Barbary (2016)**, partition luminance into $64\times 64$ or $128\times 128$ cells, compute iteratively $\kappa\sigma$-clipped background mode/median + $3\times 3$ cell median filtering + bicubic spline interpolation to strip vignetting and skyglow without erasing diffuse nebulae.
   - Convolve residual $I - B$ with a matched Gaussian PSF filter, extract 8-connected components (`minarea >= 5` to reject single hot pixels; peak `< saturation` to reject flat-topped cores), and refine sub-pixel centers via **iterative Gaussian-windowed centroids (`XWIN`/`YWIN`)** or damped Levenberg–Marquardt **2D Gaussian / Moffat PSF** fits ($\sigma_x \approx \frac{\text{FWHM}}{2.355\cdot\text{SNR}} < 0.05\text{ px}$).
3. **Geometric Hashing of Star Triangles & High-Order Warping:**
   - Construct local $k$-nearest-neighbor star triangles from the brightest $50\text{–}80$ unsaturated stars in **Valdes et al. (1995)** bounded invariant ratio space $(R_1, R_2) = (L_1/L_2,\, L_0/L_2) \in (0, 1]^2$ (related to **Groth 1986**, **Lang et al. 2010**, and **Beroiz et al. 2020 `astroalign`**), partitioned by signed cross-product chirality $\chi \in \{-1, +1\}$ in a 2D k-d tree for fast correspondence voting.
   - **Why Homography + Polynomial Warping Matters:** Rectilinear projection of a rotating celestial sphere undergoes an exact **8-DOF Projective Homography** $\mathbf{H} = \mathbf{K}\mathbf{R}(\Delta\psi)\mathbf{K}^{-1}$, plus fixed-in-sensor optical distortion (coma/radial $k_1 r^2$) that induces differential shifts as stars drift. Coarse-to-fine **RANSAC Similarity/Affine $\to$ Homography / Regularized Quadratic Chebyshev Warp** achieves $<0.1\text{ px}$ astrometric RMS across corners.
   - **Resampling Noise Physics:** 2D Bilinear resampling collapses uncorrelated sky noise variance by up to **$4\times$** at half-pixel shifts ($\sigma_{\text{out}}^2(0.5, 0.5) = 0.25\sigma_0^2$), creating concentric Moiré noise bands and artificially lowering 1-pixel $\sigma_{\text{bg}}$ via blur! **Lanczos-3 / Lanczos-4** ($6\times 6$ / $8\times 8$, with high-contrast Gibbs deringing clamp on bright star cores only) preserves both Nyquist stellar FWHM and uniform noise spectrum ($\sigma_{\text{out}}^2 \ge 0.80\sigma_0^2$).
4. **Photometric Normalization, SNR/FWHM Weighting & Winsorized Integration:**
   - Equalize per-frame additive sky background $B_i(X,Y)$ and multiplicative stellar flux scale $F_i$ to the reference frame *before* pixel rejection: $\tilde{I}_i = (I_i^{\text{warp}} - B_i^{\text{warp}})/F_i + B_0$.
   - Weight normalized frames by point-source matched-filter optimality (**Fischer & Kochanski 1994**):
     $$w_i \propto \frac{F_i^2}{\sigma_{\text{bg},i}^2 \cdot \text{FWHM}_i^2} \quad \left(\text{equivalently } \alpha_i = \frac{w_i}{F_i} \propto \frac{F_i}{\sigma_{\text{bg},i}^2 \cdot \text{FWHM}_i^2} \text{ on unscaled flux}\right)$$
   - Integrate each output pixel via **Huber Winsorized Sigma Clipping** (`WinsorizedSigmaClip`), rejecting satellite trails, cosmic rays, and drifting residual warm pixels while preserving linear flux on faint extended targets (M57 ring & M31 disk).

---

## 5. Ranked Roadmap of Atomic C++20 Iterations (Phase 3 Plan)

Each Phase 3 iteration modifies **one atomic algorithmic component** inside `cpp_stacker/`, is benchmarked on `ring_set` + `dark_set` via the locked `scripts/evaluate.py` referee, and undergoes an independent **Adversarial Red-Team Audit** before commit or rollback:

1. **Iteration 0 — Clean C++20 Scratch Baseline (`iter-0-baseline`):**
   - Multi-threaded `LibRaw::unpack()` linear `float32` reader on unrotated `2856 × 4290` active CFA coordinates with mandatory bitstream/14-bit integrity gating (rejecting the 2 corrupted darks and 3 corrupted `ring_set` frames), mean master dark subtraction, linear bilinear/green luminance extraction, basic connected-component center-of-mass bright-star detection, integer shift-and-add translation registration, and unweighted mean pixel integration.
2. **Iteration 1 — Raw CFA Calibration Upgrade (Winsorized Dark + Optical-Black Pedestal + Dark Scaling + Stride-2 Hot-Pixel Repair):**
   - Replace naive mean dark subtraction with a Winsorized sigma-clipped master dark, per-frame left-margin (`left_margin = 62`) optical-black bias subtraction, per-frame least-squares thermal dark scaling $\alpha_i$, and stride-2 same-color CFA hot/cold pixel median repair (with cross-color star-core protection) prior to demosaicing.
3. **Iteration 2 — Gradient-Corrected Linear Demosaicing (Malvar–He–Cutler $5\times 5$ / VNG):**
   - Upgrade raw CFA luminance/RGB conversion from basic bilinear/superpixel to exact $5\times 5$ Malvar–He–Cutler / VNG directional gradient demosaicing to tighten single-frame and stacked stellar FWHM without losing linearity.
4. **Iteration 3 — SEP Background Mesh + 2D Gaussian / Windowed Sub-Pixel Star Centroiding:**
   - Implement $64\times 64$ clipped background mesh subtraction + matched-filter connected components + iterative Gaussian-windowed / 2D Gaussian PSF sub-pixel centroiding (`<0.05 px` precision) paired with flux-conserving **Lanczos-3** sub-pixel resampling.
5. **Iteration 4 — Valdes Triangle Asterism Geometric Hashing + RANSAC Similarity $(dx, dy, d\theta, s)$ Registration:**
   - Replace translation-only matching with chirality-split 2D k-d tree triangle invariant matching and RANSAC 4-DOF Similarity transformation to eliminate field rotation across the 11-minute drift span (`corner_fwhm_px` & `median_eccentricity` breakthrough).
6. **Iteration 5 — Higher-Order Astrometric Warping (RANSAC 8-DOF Homography + Regularized Quadratic Chebyshev Distortion):**
   - Model gnomonic tangent-plane sky rotation ($\mathbf{H} = \mathbf{K}\mathbf{R}\mathbf{K}^{-1}$) and off-axis telescope coma/distortion via cross-validated Homography / 2nd-order Chebyshev polynomial warping (recovering outer-field corner star sharpness and faint star detections).
7. **Iteration 6 — Per-Frame Sky Gradient Equalization & Optimal $F_i^2 / (\sigma_{\text{bg},i}^2 \cdot \text{FWHM}_i^2)$ Frame Weighting:**
   - Normalize frame-to-frame sky background drift and throughput $F_i$ prior to integration, and weight frames inversely by $\sigma_{\text{bg},i}^2 \cdot \text{FWHM}_i^2$.
8. **Iteration 7 — Huber `WinsorizedSigmaClip` Robust Pixel Integration:**
   - Replace mean/weighted-mean pixel integration with OpenMP-parallel per-pixel Huber Winsorized Sigma Clipping to reject cosmic rays, satellite trails, and any surviving drifting warm pixels while strictly preserving extended nebular/galactic flux.
9. **Iteration 8 — Native CFA Bayer-Drizzle vs. VNG+Lanczos-4 Ablation:**
   - Test direct native-CFA Bayer-Drizzle integration (exploiting the 18+ frames of natural unguided Bayer-phase drift to bypass demosaic spatial smoothing altogether) against VNG + Lanczos-4, retaining whichever achieves superior audited composite score without blurring artifacts.

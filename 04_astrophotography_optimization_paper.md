# Autonomous Anti-Goodhart Evolution of a High-Performance C++20 Astrophotography Stacking Engine: From Full-Frame Correlation Failure Analysis to Out-of-Distribution Galactic Generalization

**Repository & Engine:** [`cpp_stacker/`](file:///home/julien/projects/astro_stacker/cpp_stacker) (`2,782` lines of C++20 / OpenMP / `LibRaw`, `262 KiB` native binary)  
**Immutable Blind Referee:** [`scripts/evaluate.py`](file:///home/julien/projects/astro_stacker/scripts/evaluate.py) (`chmod 444`, `SHA-256: 971b62f072143eb6172cd804f85750b7c31c507fbd13879ed4d1a46b50c415f9`)  
**Empirical Telemetry & Logs:** [`logs/dataset_manifest.json`](file:///home/julien/projects/astro_stacker/logs/dataset_manifest.json), [`logs/reference_selection.json`](file:///home/julien/projects/astro_stacker/logs/reference_selection.json), [`logs/torture_test_results.json`](file:///home/julien/projects/astro_stacker/logs/torture_test_results.json), [`logs/metrics.jsonl`](file:///home/julien/projects/astro_stacker/logs/metrics.jsonl), [`logs/03_iteration_log.md`](file:///home/julien/projects/astro_stacker/logs/03_iteration_log.md)

---

## Abstract

Stacking unguided or prime-focus deep-sky digital single-lens reflex (DSLR) exposures requires recovering faint astronomical signals buried near the Poisson sky-background floor while tracking non-uniform mount drift, celestial field rotation, outdoor sensor cooling, and raw bitstream defects. We begin with a mathematical and empirical postmortem of a legacy 2020 Python stacking prototype ([`legacy_2020/`](file:///home/julien/projects/astro_stacker/legacy_2020)) that attempted registration via full-frame spatial cross-correlation $C_{\mathbf{k}}$ and ensemble lag averaging $\langle C_{\mathbf{k}} \rangle$, proving five fundamental physical failure modes—led by coherent zero-lag master-dark estimation variance $(\tilde{\bar{\eta}} - \bar{\eta})$ that cannot be attenuated by averaging across frame pairs, stationary optical vignetting bowls, $<0.5\%$ stellar pixel occupancy, worm-gear periodic tracking drift, and field rotation.

To engineer a modern replacement without visual language model (VLM) inspection or human tuning bias, we construct a **strictly quarantined, blind autonomous algorithm-optimization harness**. Raw Canon EOS 1100D 14-bit `.CR2` forensics across 62 exposures (`dark_set`, `ring_set` M57, and holdout `andromeda_set` M31) uncover five corrupted lossless-JPEG bitstreams (including silent predictor underflow in `IMG_6096.CR2`), a zenith EXIF `flip = 5` orientation trap, a $34\,^\circ\text{C} \to 18\,^\circ\text{C}$ outdoor sensor cooling curve, and analytic $5\times 5$ Malvar–He–Cutler warm-pixel ring blooming. We lock a self-contained pure-NumPy referee ([`evaluate.py`](file:///home/julien/projects/astro_stacker/scripts/evaluate.py)) verified across **28/28 adversarial torture tests** (`Gates 0–5`) and pair it with fresh, isolated **Red-Team Auditor** subagents inspecting every `git diff` and telemetry payload. Across 10 atomic evolutionary iterations on `ring_set` (`7` kept, `4` reverted), our C++20 engine advances from a single-frame baseline of `50.0000` and an integer shift-and-add C++20 baseline of `106.0167` to **`109.5265`** (`+3.5098`), achieving $3.95\times$ background noise suppression (`0.4328 → 0.1095`), $3.34\times$ deeper $5\sigma$ point-source recovery (`105 → 351` stars), and `0` residual warm-pixel impulses. Crucially, the evolutionary trajectory uncovers a **non-commutative algorithmic ordering law**: sensor-level dark/pedestal calibration degrades composite quality (`-0.71` / `-0.79`) when attempted via spatial CFA repair *prior* to temporal outlier rejection, yet yields `+0.37` cleanly once orthogonal temporal Winsorized Sigma Clipping is in place. Evaluated **zero-shot with zero hyperparameter retuning** on the out-of-distribution Andromeda Galaxy holdout set (`andromeda_set`), the winning engine improves over the C++20 baseline from `111.4353` to **`114.8796`** (`+3.4443`, within $1.9\%$ of the training gain), sharpening stellar cores below the 21-frame exposure ensemble median (`8.2000 px < 8.2791 px`), eliminating all `477` thermal warm-pixel impulses (`477 → 0`), and preserving $100.16\%$ (`ncc_64x64 = 0.999639`) of M31's extended galactic disk.

---

## 1. Introduction & Critique of Full-Frame Spatial Cross-Correlation ($C_{\mathbf{k}}$)

### 1.1 Analytical Formulation of the 2020 Full-Frame Correlation Estimator

Let $I_m(\mathbf{x})$ denote the raw intensity recorded at discrete sensor pixel $\mathbf{x} \in \Omega \subset \mathbb{Z}^2$ ($|\Omega| = N_{\text{pix}} = 4290 \times 2856 = 12{,}252{,}240$) in exposure $m \in \{1, \dots, M\}$. Decompose each raw light frame into deterministic dark current plus readout pedestal $\bar{\eta}(\mathbf{x})$, zero-mean frame-specific readout and dark shot noise $\delta\eta_m(\mathbf{x})$, optical vignetting times sky background $V(\mathbf{x}) B_m$, and the true celestial signal $S(\mathbf{x} - \mathbf{u}_m(\mathbf{x}))$ displaced by frame-dependent astrometric motion $\mathbf{u}_m(\mathbf{x})$:

$$I_m(\mathbf{x}) = \bar{\eta}(\mathbf{x}) + \delta\eta_m(\mathbf{x}) + V(\mathbf{x}) B_m + S\big(\mathbf{x} - \mathbf{u}_m(\mathbf{x})\big) + \delta s_m(\mathbf{x})$$

where $\delta s_m(\mathbf{x})$ is zero-mean photon shot noise from the sky and astronomical sources. Given an empirical master dark $\tilde{\bar{\eta}}(\mathbf{x}) = \frac{1}{N_D} \sum_{d=1}^{N_D} D_d(\mathbf{x})$ formed from $N_D$ dark frames at sensor temperature $T_D$ and a binary bad-pixel keep-mask $M(\mathbf{x}) \in \{0, 1\}$, the 2020 Python codebase ([`spatial_correlator.py`](file:///home/julien/projects/astro_stacker/legacy_2020/astro_stacker/processing/spatial_correlator.py)) attempted to estimate inter-frame shifts $\mathbf{k} \in \mathbb{Z}^2$ by maximizing the masked, mean-subtracted spatial cross-correlation:

$$C_{\mathbf{k}}^{(i,j)} = \frac{1}{N_{\mathbf{k}}} \sum_{\mathbf{x} \in \Omega_{\mathbf{k}}} M(\mathbf{x})\,M(\mathbf{x}+\mathbf{k})\,\tilde{J}_i(\mathbf{x})\,\tilde{J}_j(\mathbf{x}+\mathbf{k}), \qquad N_{\mathbf{k}} = \sum_{\mathbf{x} \in \Omega_{\mathbf{k}}} M(\mathbf{x})\,M(\mathbf{x}+\mathbf{k})$$

where $\tilde{J}_m(\mathbf{x}) = \big(I_m(\mathbf{x}) - \tilde{\bar{\eta}}(\mathbf{x})\big) - \mu_m$ subtracts the master dark and scalar frame mean $\mu_m$, and consecutive-pair correlations were ensemble-averaged ($\langle C_{\mathbf{k}} \rangle = \frac{1}{M-1} \sum_{i=1}^{M-1} C_{\mathbf{k}}^{(i, i+1)}$) under the assumption of constant drift velocity $\mathbf{u}_{i+1} - \mathbf{u}_i \equiv \Delta\mathbf{u}$.

### 1.2 Five Fundamental Physical Failure Modes on Real DSLR Data

Expanding $\mathbb{E}\big[C_{\mathbf{k}}^{(i,j)}\big]$ over the stochastic noise realizations ($\delta\eta_i, \delta\eta_j, \delta s_i, \delta s_j$) and the finite $N_D$-sample master-dark estimation error $\epsilon_D(\mathbf{x}) \equiv \tilde{\bar{\eta}}(\mathbf{x}) - \bar{\eta}_{T_D}(\mathbf{x})$ exposes five independent physical mechanisms that cause $C_{\mathbf{k}}^{(i,j)}$ and $\langle C_{\mathbf{k}} \rangle$ to fail on real Canon EOS 1100D prime-focus exposures:

$$\mathbb{E}\big[C_{\mathbf{k}}^{(i,j)}\big] = \underbrace{\delta_{\mathbf{k},\mathbf{0}} \left(\frac{\sigma_D^2}{N_D} + \frac{1}{N_{\mathbf{0}}}\sum_{\mathbf{x}} \Delta\bar{\eta}_{\text{th},i}(\mathbf{x})\,\Delta\bar{\eta}_{\text{th},j}(\mathbf{x})\right)}_{\text{(I) Coherent Zero-Lag Dark Residual Spike}} + \underbrace{B_i B_j \, R_{\tilde{V}}(\mathbf{k})}_{\text{(II) Stationary Vignetting Hill}} + \underbrace{R_{S}^{(i,j)}(\mathbf{k} - \Delta\mathbf{u}_{ij})}_{\text{(III–V) Attenuated & Smeared Stellar Signal}}$$

1. **Coherent Finite Master-Dark Residual & Thermal Mismatch at Zero Lag ($\mathbf{k} = \mathbf{0}$):**
   Subtracting the *same* $N_D$-frame master dark ($N_D = 18$, $\sigma_D = 23.07\text{ ADU}$) from both $I_i$ and $I_j$ imprints the identical static error $-\epsilon_D(\mathbf{x}) + \Delta\bar{\eta}_{\text{th}}(\mathbf{x})$ onto every calibrated light exposure, where $\Delta\bar{\eta}_{\text{th}}(\mathbf{x}) = \bar{\eta}_{T_{\text{light}}}(\mathbf{x}) - \bar{\eta}_{18\,^\circ\text{C}}(\mathbf{x})$ is the uncompensated thermal dark-current excess at $19\text{–}34\,^\circ\text{C}$. At zero lag $\mathbf{k} = \mathbf{0}$, these errors multiply *in phase* across all $1.225 \times 10^7$ pixels:
   $$\Delta C_{\mathbf{0}}^{\text{dark}} \ge \frac{\sigma_D^2}{N_D} = \frac{(23.07)^2}{18} \approx 29.57\text{ ADU}^2$$
   plus warm-pixel variance ($\sim 15\text{–}45\text{ ADU}^2$). Crucially, **ensemble averaging $\langle C_{\mathbf{k}} \rangle$ across $M-1$ frame pairs provides zero attenuation of $\Delta C_{\mathbf{0}}^{\text{dark}}$** because $\epsilon_D(\mathbf{x})$ is deterministic across the entire light stack once $\tilde{\bar{\eta}}$ is fixed.

2. **Stationary Optical Vignetting Pedestal ($R_{\tilde{V}}(\mathbf{k})$):**
   Prime-focus Newtonian illumination $V(\mathbf{x})$ drops radially by $\sim 18\text{–}22\%$ from optical center to APS-C corners. Scalar mean subtraction $\mu_m = \overline{V} B_m + \overline{S}$ leaves the zero-mean spatial bowl $\tilde{V}(\mathbf{x}) B_m = (V(\mathbf{x}) - \overline{V}) B_m$ (`RMS ≈ 12–16 ADU`, variance $\sim 144\text{–}256\text{ ADU}^2$) locked to sensor coordinates `(0, 0)` across $100\%$ of pixels, generating a broad correlation dome peaked at $\mathbf{k} = \mathbf{0}$ that swamps stellar cross-terms.

3. **Extreme Point-Source Sparsity ($<0.5\%$ Pixel Occupancy):**
   Across `ring_set` (`105` single-frame $5\sigma$ stars of median $\text{FWHM} = 9.34\text{ px}$), stellar cores above $2\sigma_{\text{bg}}$ occupy $\approx 105 \times \pi (8\text{ px})^2 \approx 2.1 \times 10^4$ pixels—just **$0.17\%$** of the $1.225 \times 10^7$-pixel sensor. In a full-frame inner product, $99.83\%$ of pixels inject pure sky shot noise ($\sigma_{\text{bg}} \approx 28.0\text{ ADU}$), readout noise, vignetting, and fixed-pattern dark residuals while contributing zero astrometric Fisher information.

4. **Oscillatory Worm-Gear Drift & Cadence Jitter (Defeating Ensemble Averaging $\langle C_{\mathbf{k}} \rangle$):**
   Ground-truth 4-DOF astrometry ([`reference_selection.json`](file:///home/julien/projects/astro_stacker/logs/reference_selection.json)) proved that inter-frame motion is **neither linear nor uniform**:
   - Acquisition cadences jittered (`75 s` and `46 s` initial gaps vs. `31 s` steady cadence).
   - Mount worm-gear periodic error caused velocity reversals (`dx` reversing from `-10.86 px` $\to$ `+2.87 px` $\to$ `-27.57 px` $\to$ `-4.72 px` in `ring_set`; consecutive step distances ranging over a $11.5\times$ span `[1.19, 13.61] px`).
   Averaging $C_{\mathbf{k}}^{(i, i+1)}$ across frame pairs therefore smears the true stellar correlation peak across dozens of pixels in lag space while reinforcing the stationary sensor-locked peak at $\mathbf{k} = (0, 0)$.

5. **Celestial Field Rotation & Off-Axis Optical Aberrations:**
   In `andromeda_set`, polar misalignment induced monotonic field rotation $\Delta\theta \in [-0.008^\circ, +0.048^\circ]$ (`0.0561°` span), displacing corner stars at $R_{\text{corner}} = 2577\text{ px}$ by $R_{\text{corner}} \Delta\theta_{\text{rad}} = \mathbf{2.52\text{ px}}$ tangentially relative to the frame center—an transformation outside the 2-DOF translation group $(\mathbb{R}^2, +)$.

### 1.3 Compounding Implementation Defects in the 2020 Codebase

Static audit of [`legacy_2020/astro_stacker/`](file:///home/julien/projects/astro_stacker/legacy_2020/astro_stacker) revealed four severe bugs that further crippled the prototype:
- **8-Bit Non-Linear sRGB `postprocess()` Destruction ([`cannon_image_reader.py` L39](file:///home/julien/projects/astro_stacker/legacy_2020/astro_stacker/image_readers/cannon_image_reader.py#L39)):** Calling `rawpy.imread(path).postprocess()` with default parameters quantized 14-bit raw ADUs to `uint8` ($64\times$ precision loss), applied non-linear BT.709 gamma ($\gamma \approx 2.222$, breaking additive dark subtraction $(S+D)^{1/\gamma} - D^{1/\gamma} \neq S^{1/\gamma}$), auto-stretched dark read noise to `255`, and applied AHD demosaicing prior to calibration.
- **Modulo-256 `uint8` Accumulator Overflow ([`image_analyzer.py` L14–L22](file:///home/julien/projects/astro_stacker/legacy_2020/astro_stacker/processing/image_analyzer.py#L14-L22)):** Initializing `final_image = image.data` kept accumulators in `uint8`, wrapping pixel sums and squares modulo `256` on every frame, compounded by omitting the square on `data_average` in the sample variance formula.
- **Silent Bypass of Dark Subtraction & Unmasked Numerator Correlation ([`spatial_correlator.py` L14–L43](file:///home/julien/projects/astro_stacker/legacy_2020/astro_stacker/processing/spatial_correlator.py#L14-L43)):** `SpatialCorrelator.correlate` accessed `image.data` rather than `image.get_data()`, skipping `self.dark` subtraction entirely, correlated *unmasked* hot pixels in the numerator while dividing by `cross_correlate_2d(mask_1, mask_2)`, and divided by zero-crossing `bgsub=True` marginals.
- **Savitzky–Golay Boolean Mask Inflation ([`image.py` L63–L67](file:///home/julien/projects/astro_stacker/legacy_2020/astro_stacker/models/image.py#L63-L67)):** Filtering a `0/1` keep-mask with a cubic Savitzky–Golay polynomial and casting `.astype(bool)` turned almost every `0` (non-zero float ringing) back into `True`, erasing the hot-pixel mask.

---

## 2. Real-World Sensor Forensics & Calibration Physics

### 2.1 Acquisition Inventory & Sensor Geometry (`Canon EOS 1100D`)

Using [`scripts/inspect_cr2_dataset.py`](file:///home/julien/projects/astro_stacker/scripts/inspect_cr2_dataset.py) (backed by thread-safe `LibRaw 0.21.5`), all 128 files in `pictures/2020-11-28-deep-sky-ring-andromeda/` were audited under strict `.CR2` filtering (`62` `.CR2` raw files inspected; `66` `.JPG`, `.CR2.xmp`, `.npz`, and `.xcf` sidecars quarantined). All 62 `.CR2` exposures come from a single continuous **36.58-minute** outdoor session with `Canon EOS 1100D` (`BodySerial: 162063089193`, ISO 1600, 14-bit ADC):
- **Physical Raw Buffer:** `4352 × 2874` `uint16_t` CFA lattice.
- **Active Unrotated Visible Window:** `raw_image[18:2874, 62:4352]` $\to$ **`H = 2856, W = 4290`** (`12,252,240` active CFA sites), Bayer phase `[[R, G1], [G2, B]]`.
- **Optical-Black Readout Strip:** `raw_image[18:2874, 2:60]` (`58` shielded columns per row; median `2047–2049 ADU`), tracking per-frame and per-row CMOS horizontal readout bias drift.

| Dataset | Role | Chronological Window | `.CR2` Total / Clean / Bad | Exposure(s) | EXIF `flip` | MakerNotes $T_{\text{cam}}$ | Active Sky Signal Above Bias `(R, G, B)` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`andromeda_set`** | Holdout Test (M31) | `t = 0 .. 693 s` | `21 / 21 / 0` | `2×29s + 19×28s` | `0` (`0°`) | **`34 °C → 21 °C`** | `(+57.8, +162.8, +142.8)` ADU |
| **`ring_set`** | Primary Train (M57) | `t = 881 .. 1545 s` | `21 / 18* / 3` | `21×29.0s` | **`5` (`90° CCW`)** | **`24 °C → 19 °C`** | `(+72.7, +206.0, +181.5)` ADU |
| **`dark_set`** | Calibration Darks | `t = 1577 .. 2195 s` | `20 / 18 / 2` | `20×29.0s` | `0` (`0°`) | **`18 °C` steady** | `(+0.9, +0.2, +0.0)` ADU |

*\*Note: `ring_set` contains 18 completely clean `.CR2` frames plus `IMG_6089.CR2`, whose top `2,476 / 2,856` active rows (`86.7%`) are completely uncorrupted prior to an end-of-stream bitstream fault at row `2,476`.*

### 2.2 Three Real-World Sensor & Bitstream Hazards

1. **Five Corrupted Lossless-JPEG `.CR2` Bitstreams (Including Silent Predictor Underflow):**
   - **Explicit `LibRaw` Decoder Faults (4 files):** `dark_set/IMG_6098.CR2` (offset `9,945,668`), `ring_set/IMG_6079.CR2` (`7,103,945`), and `ring_set/IMG_6083.CR2` (`12,325,909`) suffer mid-stream Huffman/predictor desynchronization wrapping pixels to `65535` (`> 16383` 14-bit ADC limit). `ring_set/IMG_6089.CR2` (`15,098,935`) desynchronizes only in the final `4` bottom strips (`row >= 2476`, `max = 63272`), leaving rows `[0, 2476)` pristine (`p1 = 2078 ADU`).
   - **Silent Predictor Underflow (`dark_set/IMG_6096.CR2`):** Starting at row `1956`, Canon's 14-bit lossless-JPEG differential predictor drifts downward to `743 ADU` (`active_p1 = 1329.0 ADU` vs. `2011 ADU` on valid darks, inflating `std` from `28.4` to `130.1 ADU`) **while `LibRaw::unpack()` returns `LIBRAW_SUCCESS` (`0`) and `data_error_count == 0`**. Only our dual physics gate (`data_error_count == 0 && active_max <= 16383 && active_p1 >= 1850.0`) catches all five corrupted files.

2. **The Zenith EXIF `flip = 5` Gravity-Sensor Trap:**
   Slewing near zenith for M57 (`ring_set`) tripped the DSLR's internal ball-bearing orientation sensor to `sizes.flip = 5` (`90° CCW`), while `dark_set` and `andromeda_set` recorded `sizes.flip = 0`. High-level readers that auto-orient images transpose `ring_set` to `4290 × 2856` while leaving `dark_set` at `2856 × 4290`, misaligning every hot pixel and vignetting contour. Operating strictly on unrotated physical `raw_image[18:2874, 62:4352]` coordinates eliminates this failure mode completely.

3. **Outdoor Thermal Cooling ($34\,^\circ\text{C} \to 18\,^\circ\text{C}$) & Analytic MHC $5\times 5$ Warm-Pixel Impulse Blooming:**
   Silicon dark current doubles roughly every $\Delta T \approx 5.8\text{–}6.5\,^\circ\text{C}$. Because `andromeda_set` ($34 \to 21\,^\circ\text{C}$) and `ring_set` ($24 \to 19\,^\circ\text{C}$) were warmer than `dark_set` ($18\,^\circ\text{C}$), unscaled master-dark subtraction leaves positive single-CFA-site thermal impulses of amplitude $A \in [200, 4000]\text{ ADU}$. When passed through the linear **Malvar–He–Cutler (2004)** $5\times 5$ demosaicing kernels and combined into photon-weighted luminance $L = 0.25 R + 0.50 G + 0.25 B$, a unit single-site impulse $A\,\delta_{0,0}$ produces an exact analytic luminance response ([`reference_selection.json` L44–L80](file:///home/julien/projects/astro_stacker/logs/reference_selection.json#L44-L80)):
   - **On an `R` or `B` site:** Center $L_{0,0} = +0.6875\,A$, axial $\pm 1\text{ px}$ cross $L_{\pm 1, 0} = +0.1250\,A$, diagonal $(\pm 1, \pm 1)$ $L_{\pm 1, \pm 1} = +0.0625\,A$, and **negative axial $\pm 2\text{ px}$ ring $L_{\pm 2, 0} = -0.109375\,A$**.
   - **On a `G1` or `G2` site:** Center $L_{0,0} = +0.8125\,A$, axial $\pm 1\text{ px}$ cross $L_{\pm 1, 0} = +0.1250\,A$, **negative diagonal $(\pm 1, \pm 1)$ $L_{\pm 1, \pm 1} = -0.0625\,A$**, and negative axial $\pm 2\text{ px}$ ring $L_{\pm 2, 0} = -0.015625\,A$.

   Consequently, any warm pixel with $A > 40\sigma_{\text{bg}}$ (`~1100 ADU`) illuminates a **5-pixel cross or 9-pixel $3\times 3$ patch** above $5\sigma_{\text{bg}}$, **bypassing naive `minarea >= 5` connected-component filters** (`61/frame` median in `ring_set`; `104/frame` median in `andromeda_set`). Fortunately, empirical profiling across `2,522` raw CFA impulses proved **100% morphological disjointness** (`1.000` recall, `0` false positives) against real optical PSFs:
   - **Warm MHC Impulses:** $\text{FWHM} \in [1.15, 1.33]\text{ px}$, $3\times 3$ peak concentration $S_{3\times 3} = L_{0,0}/\sum_{3\times 3} L_+ \in [0.47, 0.62]$, axial $r=2$ ring ratio $R_2 / L_{0,0} < -0.02$.
   - **Optical Stars:** $\text{FWHM} \in [5.55, 18.60]\text{ px}$, $S_{3\times 3} \in [0.12, 0.23]$, $R_2 / L_{0,0} \in [+0.29, +0.84]$.

---

## 3. Blind Anti-Goodhart Referee & Multi-Agent Red-Team Architecture

```mermaid
flowchart LR
    subgraph Quarantine["Strict Image & Binary Quarantine (Zero VLM Pixels)"]
        RAW["Canon .CR2 Exposures\n(ring_set / dark_set)"] --> CPP["C++20 Stacking Engine\n(cpp_stacker/build/astro_stacker)"]
        CPP --> NPY["Linear Stack\n(2856 x 4290 x 3 float32 .npy)"]
        CPP --> TRJ["Per-Frame Astrometry\n(transforms.json)"]
    end
    NPY --> EVAL["Immutable Blind Referee\nscripts/evaluate.py (chmod 444)"]
    TRJ --> EVAL
    REF["Locked Reference Array\nlogs/ref_ring.npy (chmod 444)"] --> EVAL
    EVAL --> MET["JSON Telemetry\n(Core Metrics + Gates 0-5)"]
    CPP -. "git diff cpp_stacker/" .-> AUD["Fresh Isolated Subagent\nAdversarial Red-Team Auditor"]
    MET --> AUD
    TRJ --> AUD
    AUD --> DEC{"Score > Best AND\nAUDIT_PASS?"}
    DEC -- "Yes" --> GITC["git commit + metrics.jsonl"]
    DEC -- "No" --> GITR["git checkout -- cpp_stacker/\n(Log Reverted Postmortem)"]
```

### 3.1 Anchor-Star Photometric Normalization & Composite Objective

To prevent trivial scale manipulation ($I \mapsto \alpha I + \beta$, $\alpha \to 0$) while maintaining invariance to legitimate exposure-unit conventions, [`scripts/evaluate.py`](file:///home/julien/projects/astro_stacker/scripts/evaluate.py) converts RGB arrays to canonical RGGB linear luminance $L = 0.25 R + 0.50 G + 0.25 B$, extracts up to $N_{\text{anc}} = 40$ isolated ($\ge 32\text{ px}$ separation), non-saturated (`peak < 10,500 ADU`) anchor stars on `--ref`, cross-matches them within $4.0\text{ px}$ on `--stack` (requiring $\ge 20$ matches), and measures background-annulus-subtracted ($R_{\text{in}}=16, R_{\text{out}}=22\text{ px}$) circular aperture fluxes ($R_{\text{ap}}=12\text{ px}$) $F_{r,k}, F_{s,k}$. Setting $s_{\text{ref}} = F_0 / \operatorname{median}_k(F_{r,k})$ with $F_0 = 1000.0$ and $r_{\text{med}} = \operatorname{median}_k(F_{s,k}/F_{r,k})$, both luminance maps are mapped onto an identical standardized photometric scale:

$$L_{\text{stack}}^{\text{norm}}(\mathbf{x}) = \left(\frac{s_{\text{ref}}}{r_{\text{med}}}\right) L_{\text{stack}}(\mathbf{x}) - \left(\frac{s_{\text{ref}}}{r_{\text{med}}}\right) \tilde{B}_{\text{sky,stack}}, \qquad \operatorname{median}_{\text{sky}}\!\big(L_{\text{stack}}^{\text{norm}}\big) = 0.0, \qquad \operatorname{median}_{k}\!\big(F_{s,k}^{\text{norm}}\big) = 1000.0$$

On this normalized linear plane (interior full-overlap ROI `[64:2792, 64:4226]`), the referee evaluates six core metrics and combines them into a logarithmic octave **Composite Score** anchored at **exact `50.0`** on `--ref`:

$$\text{CompositeScore} = 50.0 + 25.0\log_2\!\left(\frac{\sigma_{\text{bg,norm}}^{\text{ref}}}{\sigma_{\text{bg,norm}}^{\text{stack}}}\right) + 35.0\log_2\!\left(\frac{\text{FWHM}_{\text{eff}}^{\text{ref}}}{\text{FWHM}_{\text{eff}}^{\text{stack}}}\right) + 10.0\log_2\!\left(\frac{N_\star^{\text{stack}}}{N_\star^{\text{ref}}}\right) + 15.0\big(e_{\text{ref}} - e_{\text{stack}}\big) - 40.0\big(1.0 - \rho_{\text{ext}}\big)$$

where $\text{FWHM}_{\text{eff}} = 0.65\,\text{median\_fwhm\_px} + 0.35\,\text{corner\_fwhm\_px}$ (7-DOF damped Levenberg–Marquardt 2D elliptical Gaussian fits to $5\sigma$, `minarea >= 5`, warm-impulse-filtered optical stars) and $\rho_{\text{ext}} = \min(\text{NCC}_{16\times 16}, \text{NCC}_{64\times 64})$ is the multi-scale star-masked, 1st-degree-plane-detrended normalized cross-correlation of diffuse emission. Weighting sharpness at $35.0/\text{octave}$ versus $25.0/\text{octave}$ for background RMS ensures that any hidden low-pass filter loses more points in PSF bloat than it gains in noise suppression—even before hard anti-cheat gates fire.

### 3.2 Nine Hard Physical Anti-Cheat Gates & `28 / 28` Adversarial Torture-Testing

Any violation of Gates `0–5` immediately forces `cheating_detected = true` and `score = -inf`:
- **Gate 0 (Array & Channel Integrity):** Enforces `(2856, 4290, 3)` finite floating-point shape, non-zero channel variance, and distinct RGB channels (`std(c1 - c2) > 1e-4 * min(std)`).
- **Gate 1 (Dark-Floor & Histogram-Tail Clipping):** Disqualifies stacks if $>0.05\%$ (`0.0005`) of quiet sky pixels share an identical mode/floor value on `L`, `R`, `G`, or `B`, or if narrow $0.002\sigma$ lower-tail micro-bins spike $>0.50\%$ (defeating dithered floor clamping).
- **Gate 2a (Sky Spatial-Smoothing Noise Correlation):** Measures $4\times 4$ block-averaged RMS inflation $\mathcal{R}_{4\times 4} = 4\sigma_{4\times 4}/\sigma_{1\times 1}$ and 1-pixel lag autocorrelation $\rho_{1\text{px}}$ in quiet sky cells, triggering if $\mathcal{R}_{4\times 4}^{\text{stack}} > 1.25\,\mathcal{R}_{4\times 4}^{\text{ref}}$ or $\rho_{1\text{px}}^{\text{stack}} > \rho_{1\text{px}}^{\text{ref}} + 0.18$. In torture test `(h)`, even a subtle $\sigma = 0.8\text{ px}$ Gaussian blur inflated $\mathcal{R}_{4\times 4}$ from `1.31` to `2.39` and $\rho_{1\text{px}}$ from `0.26` to `0.73`, triggering instant disqualification.
- **Gate 2b (Exposure-Ensemble-Calibrated FWHM Ceiling):** Because `IMG_6074.CR2` (`ref_ring.npy`, `FWHM = 9.3427 px`) occurred at a stationary worm-gear turnaround whereas the 18 clean `ring_set` frames have single-exposure median `FWHM = 10.9047 px`, Gate 2b enforces $\le 1.10\times$ ceiling against both the all-star ensemble baseline and the population-matched top-40 bright-star ensemble baseline.
- **Gate 3a–3c (Frame-Drop, Kish $N_{\text{eff}}$, Stationary Hot-Pixel & Empirical Poisson Bounds):** Deterministically fingerprints `--ref` from pixel medians to require $N_{\text{int}} \ge 18$ (`ring_set`) / $\ge 20$ (`andromeda_set`), Kish effective sample size $N_{\text{eff}} = (\sum w_i)^2/\sum w_i^2 \ge 0.65\,N_{\min}$, non-reference median sky shift $\ge 3.0\text{ px}$ and trajectory diagonal $\ge 15.0\text{ px}$ (rejecting `(0,0,0)` hot-pixel alignment), and empirical sky RMS consistency $0.50\,\sigma_{\text{ref}}/\sqrt{N_{\text{int}}} \le \sigma_{\text{stack}} \le \min(0.60, 1.85/\sqrt{N_{\text{eff}}})\sigma_{\text{ref}}$ (defeating forged `transforms.json` files).
- **Gate 4 (Multi-Scale Diffuse Flux Preservation):** Requires $\rho_{\text{ext}} = \min(\text{NCC}_{16\times 16}, \text{NCC}_{64\times 64}) \ge 0.85$ and `structure_rms_ratio_64x64 >= 0.70`, defeating `64×64` SEP background-mesh subtraction on output pixels (Guardrail 4).
- **Gate 5 (Photometric Linearity & Symmetry):** Requires Pearson $r(F_r, F_s) \ge 0.96$, relative MAD $\le 0.15$, and bright-half vs. faint-half median flux-ratio asymmetry $\le 0.10$ (defeating $\gamma$/arcsinh stretches).

Prior to locking [`scripts/evaluate.py`](file:///home/julien/projects/astro_stacker/scripts/evaluate.py), an independent **Adversarial Torture-Tester** subagent authored [`scripts/torture_test_evaluator.py`](file:///home/julien/projects/astro_stacker/scripts/torture_test_evaluator.py) (`28` black-box CLI runs across `ring_set` and `andromeda_set`; [`torture_test_results.json`](file:///home/julien/projects/astro_stacker/logs/torture_test_results.json)), catching and resolving four subtle referee calibration bugs (high-SNR Moffat outer-wing $\chi^2_\nu$ variance floor `(0.03 * amp)^2`, sparse-field `64×64` meso-scale NCC addition, `sky_rms`-normalized Levenberg–Marquardt convergence for exact `50.0` self-identity, and bright-star population FWHM scaling) to achieve **`28 / 28` assertions passed** (`(a)` exact `50.0`, `(f)` float32 scale invariance `50.000043` / `49.996792`, `(k)` honest integer shift-and-add controls `105.2461` / `112.1717`, and `score = -inf` on all cheat vectors `(b, c, d, e, g, h, i, j)`). Static AST verification confirmed zero external local `.py` or `.so` imports, sealing the referee via `chmod 444`.

---

## 4. Autonomous Evolutionary Optimization & 10-Iteration Ablation Study (`ring_set`)

### 4.1 Complete Evolutionary Trajectory (`logs/metrics.jsonl`)

Every iteration proposed a single atomic hypothesis on `ring_set` + `dark_set`. Improvements required passing **both** the locked referee (`score > prev_best`, `cheating_detected == false`) and a **fresh, isolated Red-Team Auditor subagent** (`VERDICT: AUDIT_PASS`) inspecting `git diff cpp_stacker/` and JSON telemetry; regressions were immediately reverted via `git checkout -- cpp_stacker/ && git clean -fd cpp_stacker/`.

| Iter | Git Tag / Atomic Hypothesis | Status | Red-Team Audit | Score | $\Delta$ vs. Best | `med_fwhm` (px) | `cor_fwhm` (px) | `ecc` | `norm_bg_rms` | `stars` | $\rho_{\text{ext}}$ | Warm Imp. | $N_{\text{int}}$ ($N_{\text{eff}}$) | Wall (s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ref** | Single Sharpest Frame (`IMG_6074.CR2`) | Anchor | Locked | `50.0000` | — | `9.3427` | `13.9914` | `0.5446` | `0.432828` | `105` | `1.0000` | — | `1 (1.00)` | — |
| **0** | `iter-0-baseline`: LibRaw 14-bit gate, 2048 bias + mean dark, MHC 5×5, COM moments, integer shift+add, mean | **KEPT** | [`AUDIT_PASS`](conversation://dfea2856-dde7-4e6d-a41e-5c01c1367678) | **`106.0167`** | `+56.0167` | `11.1838` | `14.6633` | `0.6289` | `0.114917` | `316` | `0.9938` | `11` | `18 (18.00)` | `42.07` |
| **1** | `iter-1`: OB pedestal + Winsorized dark + thermal scale + cross-color (`r=1` vs `8.06`) CFA median repair | **REVERTED** | Degraded | `105.3087` | `-0.7080` | `11.2080` | `15.2426` | `0.6283` | `0.114672` | `318` | `0.9937` | `0` | `18 (18.00)` | `71.13` |
| **2** | `iter-2`: OB pedestal + Winsorized dark + thermal scale + same-color (`r=2` vs `18`) CFA median repair | **REVERTED** | Degraded | `105.2288` | `-0.7879` | `11.2119` | `15.3489` | `0.6280` | `0.114727` | `320` | `0.9937` | `0` | `18 (18.00)` | `64.32` |
| **3** | `iter-3-subpixel-lanczos3`: Detection-only `64×64` SEP mesh + LM 2D Gaussian centroids + 2D Lanczos-3 shift | **KEPT** | [`AUDIT_PASS`](conversation://57df68a2-4362-469e-8a88-d3edde59abe8) | **`107.3607`** | `+1.3440` | `11.1166` | `15.6046` | `0.6327` | `0.110567` | `343` | `0.9932` | `24` | `18 (18.00)` | `61.55` |
| **4** | `iter-4-asterism-similarity-4dof`: Chiral Valdes triangles + RANSAC/IRLS 4-DOF Similarity (same Lanczos-3) | **KEPT** | [`AUDIT_PASS`](conversation://32130eba-aa50-4194-9cd8-f7e7785d7fec) | **`107.6978`** | `+0.3372` | `11.1734` | `15.3746` | `0.6362` | `0.110211` | `345` | `0.9936` | `26` | `18 (18.00)` | `52.77` |
| **5** | `iter-5`: 6-DOF Full Affine IRLS refinement (anisotropic atmospheric shear/scale) | **REVERTED** | Degraded | `107.3738` | `-0.3241` | `11.1018` | `15.5491` | `0.6319` | `0.110120` | `337` | `0.9930` | `25` | `18 (18.00)` | `55.43` |
| **6** | `iter-6-planar-sky-optimal-weights`: Anchor flux scale $F_i$ + 1st-deg diff sky plane + PSF/SNR weights $w_i$ | **KEPT** | [`AUDIT_PASS`](conversation://03d26177-8480-438c-acdb-6382bb596d79) | **`108.6381`** | `+0.9402` | `10.8941` | `15.1315` | `0.6114` | `0.112261` | `348` | `0.9944` | `32` | `18 (17.19)` | `64.59` |
| **7** | `iter-7-winsorized-sigma-clip`: Temporal per-pixel Winsorized Sigma Clip ($\kappa=2.8$, seeing floor, 18 frames) | **KEPT** | [`AUDIT_PASS`](conversation://523f0d58-bbf8-4ced-915c-d0f756cc13f7) | **`108.8227`** | `+0.1846` | `10.9057` | `15.0433` | `0.6096` | `0.111933` | `347` | `0.9943` | **`0`** | `18 (17.19)` | `64.08` |
| **8** | `iter-8-ob-pedestal-winsorized-dark`: 33-row OB pedestal + Winsorized dark + thermal scale (zero spatial interp) | **KEPT** | [`AUDIT_PASS`](conversation://aafc6e68-c4df-4b6c-878a-c6e44a8b2c66) | **`109.1949`** | `+0.3722` | **`10.8520`** | **`14.8577`** | `0.6127` | `0.111904` | `347` | `0.9943` | **`0`** | `18 (17.21)` | `78.31` |
| **9** | `iter-9`: Two-pass directional Hamilton–Adams demosaicing vs. linear MHC 5×5 | **REVERTED** | Degraded | `108.2440` | `-0.9509` | `10.9212` | `15.0799` | `0.6109` | `0.113102` | `345` | `0.9941` | `0` | `18 (17.22)` | `99.29` |
| **10** | `iter-10-partial-row-salvage`: Row-level 14-bit ADC corruption boundary salvage (`IMG_6089.CR2` top 2,468 rows) | **KEPT** | [`AUDIT_PASS`](conversation://cbb27c5e-ae93-425c-82f2-1f87c158d8a6) | **`109.5265`** | `+0.3316` | `10.9170` | `15.1489` | `0.6140` | **`0.109508`** | **`351`** | `0.9942` | **`0`** | `19 (18.09)` | `69.94` |

### 4.2 Mathematical Formulation of the Winning Pipeline Stages

1. **Scale-Invariant Baseline Reference Selection & Linear MHC Demosaicing (`Iter 0`, `+56.0167`):**
   [`raw_loader.cpp`](file:///home/julien/projects/astro_stacker/cpp_stacker/src/raw_loader.cpp) enforces strict `.CR2` ingestion and dual 14-bit ADC validation on unrotated `2856 × 4290` RGGB coordinates; [`demosaic.cpp`](file:///home/julien/projects/astro_stacker/cpp_stacker/src/demosaic.cpp) applies exact Malvar–He–Cutler $5\times 5$ linear kernels with parity-preserving `reflect101` borders. Critically, an initial moment prototype using an absolute $2\sigma_{\text{sky}}$ isophote suffered faint-star truncation bias (selecting `IMG_6082.CR2`). Replacing the fixed threshold with a **scale-invariant $15\%$-of-peak amplitude gate** ($I - b_{\text{ann}} > \max(2\sigma_{\text{sky}}, 0.15 A)$ over $r \le 14\text{ px}$, corrected by exact truncated-Gaussian integral factor $\big[\frac{1 - (1 + q_{\max})e^{-q_{\max}}}{1 - e^{-q_{\max}}}\big]^{-1} = 1.50328$ where $q_{\max} = -\ln 0.15$) selected `IMG_6074.CR2` autonomously and established `106.0167`.

2. **Detection-Only `64×64` SEP Mesh, 7-DOF Levenberg–Marquardt Centroids & 2D Lanczos-3 Resampling (`Iter 3`, `+1.3440`):**
   [`star_detector.cpp`](file:///home/julien/projects/astro_stacker/cpp_stacker/src/star_detector.cpp) computes an internal $64\times 64$ $\kappa\sigma$-clipped median/MAD mesh smoothed by a $3\times 3$ cell median filter **strictly on a temporary luminance buffer** (never modifying output RGB frames), removing the radial vignetting bowl to detect faint corner stars down to $5\sigma$. Each candidate is fitted on a $25\times 25$ cutout with a 7-parameter 2D elliptical Gaussian via damped Levenberg–Marquardt ($\mathbf{J}^\top\mathbf{J} + \lambda\operatorname{diag}(\mathbf{J}^\top\mathbf{J})$). Integer shift rounding ($\pm 0.5\text{ px}$ uniform quantization error, variance $1/12\text{ px}^2$) is replaced in [`integrator.cpp`](file:///home/julien/projects/astro_stacker/cpp_stacker/src/integrator.cpp) by continuous inverse coordinate mapping through a partition-of-unity normalized $6\times 6$ **2D Lanczos-3** kernel:
   $$L_3(u) = \operatorname{sinc}(u)\operatorname{sinc}(u/3)\,\mathbf{1}_{|u|<3}, \qquad \tilde{w}_k(\delta x) = \frac{L_3(\delta x - k)}{\sum_{m=-2}^{3} L_3(\delta x - m)}$$
   Unlike bilinear interpolation—which collapses white-noise variance by $4\times$ at half-pixel shifts ($\sum w_k(0.5)^2 = 0.5$ per axis, creating Moiré smoothing bands)—Lanczos-3 preserves $\mathbb{E}[\sum \tilde{w}_k^2] \approx 0.89$ per axis, recovering `+27` faint $5\sigma$ stars (`316 → 343`) and sharpening `median_fwhm_px` (`11.1838 → 11.1166 px`).

3. **Chiral Valdes Triangle Geometric Hashing & 4-DOF Procrustes Similarity (`Iter 4`, `+0.3372`):**
   Using the **exact same 2D Lanczos-3 kernel** (`0`-line diff in `integrator.cpp`), [`registration.cpp`](file:///home/julien/projects/astro_stacker/cpp_stacker/src/registration.cpp) constructs scale- and rotation-invariant **Valdes et al. (1995)** triangle descriptors $(u, v) = (L_1/L_2, L_0/L_2) \in (0, 1]^2$ ($L_0 \le L_1 \le L_2$, $L_2 \ge 140\text{ px}$) augmented with signed cross-product chirality $\chi \in \{-1, +1\}$ (forbidding unphysical mirror reflections) indexed in a $64\times 64$ invariant hash table. Candidate correspondences seed full-field RANSAC (`2.2 px` threshold) followed by 5-pass Huber-weighted ($\delta = 0.45\text{ px}$) IRLS solving the exact linear Procrustes system centered at $(c_x, c_y) = (2144.5, 1427.5)$:
   $$\begin{bmatrix} f_x \\ f_y \end{bmatrix} = \begin{bmatrix} a & -b \\ b & a \end{bmatrix} \begin{bmatrix} r_x \\ r_y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}, \qquad a = \frac{\sum w_i (\tilde{r}_{x,i}\tilde{f}_{x,i} + \tilde{r}_{y,i}\tilde{f}_{y,i})}{\sum w_i (\tilde{r}_{x,i}^2 + \tilde{r}_{y,i}^2)}, \quad b = \frac{\sum w_i (\tilde{r}_{x,i}\tilde{f}_{y,i} - \tilde{r}_{y,i}\tilde{f}_{x,i})}{\sum w_i (\tilde{r}_{x,i}^2 + \tilde{r}_{y,i}^2)}$$
   Even on `ring_set`'s modest $0.021^\circ$ rotation span, 4-DOF alignment sharpened `corner_fwhm_px` by `-0.230 px` (`15.6046 → 15.3746 px`).

4. **Differential 1st-Degree Sky-Plane Equalization & Optimal PSF/SNR Weighting (`Iter 6`, `+0.9402`):**
   After estimating each frame's stellar throughput scale $F_i = \operatorname{median}_{k}(F_{i,k}/F_{\text{ref},k})$, [`integrator.cpp`](file:///home/julien/projects/astro_stacker/cpp_stacker/src/integrator.cpp) fits a **3-parameter 1st-degree polynomial plane** $\Delta P_i^{(c)}(n_x, n_y) = c_{0,i}^{(c)} + c_{x,i}^{(c)} n_x + c_{y,i}^{(c)} n_y$ via 3-pass Huber IRLS to $96\times 96$ block medians of the *astrometrically aligned difference* $\frac{1}{F_i} I_i^{(c)}(\mathbf{T}_i\mathbf{x}) - I_{\text{ref}}^{(c)}(\mathbf{x})$. Because stationary astrophysical emission ($S(\mathbf{x})$, including M57 and M31's entire disk) cancels identically in $\frac{1}{F_i}I_i - I_{\text{ref}}$, $\Delta P_i^{(c)}$ removes strictly time-varying skyglow pedestals and linear gradients (`extended_flux_preservation` rose to `0.994358`). Frames are weighted by point-source SNR and sharpness:
   $$w_i \propto \frac{F_i^2}{\sigma_{\text{bg},i}^2 \cdot \text{FWHM}_{\text{eff},i}^3 \cdot \big(1 + 0.5\,e_i^2\big)}, \qquad w_i \in [0.35, 1.85]\bar{w}$$
   Down-weighting worm-gear velocity-spike exposures (`FWHM ~ 14.3 px`) in favor of turnaround exposures (`9.4–10.0 px`) while preserving Kish $N_{\text{eff}} = 17.194 / 18.0$ (`95.5%`) drove `median_fwhm_px` down to **`10.8941 px`**—sharper than the unweighted 18-exposure median (`10.9047 px`)—and reduced `median_eccentricity` from `0.6362` to `0.6114`.

5. **Temporal Per-Pixel Winsorized Sigma Clipping with Seeing-Variance Floor (`Iter 7`, `+0.1846`):**
   At each fixed pixel $(r, c)$ and channel $c$, [`winsorized_weighted_combine`](file:///home/julien/projects/astro_stacker/cpp_stacker/src/integrator.cpp#L276-L327) applies 2-iteration symmetric Huber/Winsorized clamping ($\kappa = 2.8$, consistency factor $1.038$) strictly along the 1D temporal exposure axis, incorporating an amplitude-proportional PSF-seeing tolerance floor:
   $$\sigma_{\text{tol}}(r, c) = \sqrt{\sigma_{\text{emp}}^2 + \Big(0.22 \max\!\big(0,\, \mu(r,c) - B_{\text{sky,ref}}^{(c)}\big)\Big)^2}$$
   On sky background ($\mu \approx B_{\text{sky,ref}}$), $\sigma_{\text{tol}} = \sigma_{\text{emp}}$ clamps every drifting warm pixel, negative MHC $r=2$ ring, and cosmic ray (`residual_warm_pixel_impulses: 32 → 0`) with zero spatial coupling (`block_4x4_ratio: 1.3906 → 1.3899`). On bright stellar cores ($\mu \gg B_{\text{sky,ref}}$), the $0.22(\mu - B)$ floor prevents `9.4 px` vs. `14.3 px` tracking-width differences from being falsely clipped (`flux_linearity_r = 0.99963`).

6. **Optical-Black Row Pedestal, Winsorized Master Dark & Thermal Dark Scaling (`Iter 8`, `+0.3722`):**
   [`calibration.cpp`](file:///home/julien/projects/astro_stacker/cpp_stacker/src/calibration.cpp) subtracts a 33-row running median of the 58-column left optical-black strip (`ob_row_medians`), integrates the 18 clean darks via $\kappa = 3.0$ temporal Winsorized clipping, and fits a scalar thermal dark factor $\alpha_{\text{thermal}} = \operatorname{median}_{k \in \text{hot}}(l_{\text{exc},k}/d_{\text{exc},k})$ across unsaturated hot dark sites (`250 < d_0 < 9000 ADU`), scaling elevated thermal dark current ($d > 25\text{ ADU}$) without amplifying cold-pixel read noise or applying any single-frame spatial interpolation. Both `median_fwhm_px` (`10.8520 px`) and `corner_fwhm_px` (`14.8577 px`) reached their sharpest 18-frame values.

7. **Stencil-Isolated Partial-Row Bitstream Salvage (`Iter 10`, `+0.3316` $\to$ Final `109.5265`):**
   [`raw_loader.cpp`](file:///home/julien/projects/astro_stacker/cpp_stacker/src/raw_loader.cpp) scans active rows top-down for ADC overflow (`> 16383`) or predictor underflow (`< 1750 ADU`). When $\ge 80\%$ (`>= 2,285` rows) of contiguous top rows pass full 14-bit integrity (`top_p1 >= 1850 ADU`), rows `[0, contiguous_clean_rows - 8)` are salvaged, and [`integrator.cpp`](file:///home/julien/projects/astro_stacker/cpp_stacker/src/integrator.cpp) excludes the frame for `fy >= valid_rows - 4` (guaranteeing a 10-row combined buffer so neither `±3 px` Lanczos-3 nor `±2 px` MHC stencils ever touch corrupted rows) while renormalizing active weights per pixel. Salvaging `IMG_6089.CR2`'s top `2,468` rows (`57` star inliers, `residual_rms = 0.420 px`) increased `effective_sample_size` from `17.206` to `18.094`, lowered `normalized_bg_rms` to **`0.109508`**, and recovered **`351`** $5\sigma$ stars.

### 4.3 Deep Postmortems on the Four Reverted Hypotheses

1. **The Non-Commutative Ordering Discovery (`Iter 1` `-0.7080` & `Iter 2` `-0.7879` vs. `Iter 7 + Iter 8` `+0.5569`):**
   Why did sensor-level calibration fail in Iterations 1 and 2 (`106.02 → 105.31 / 105.23`), yet succeed cleanly in Iteration 8 (`108.82 → 109.19`)?
   - First, comparing `Iter 1` against `Iter 2` isolated a subtle color-PSF bug in `Iter 1`'s cross-color warm-pixel guard (`cross_inner` at $r=1$ vs. `cross_outer` at $r=8.06$): for comatic corner stars (`FWHM ≈ 15 px`, $\sigma \approx 6.4\text{ px}$) peaking on a Green site, $r = 8.06\text{ px}$ still retains $\exp(-65 / 2\sigma^2) = 45\%$ of peak flux, while `(R+B)/2G ≈ 0.35`, yielding $0.55 \times 0.35 = 0.192 < 0.20$ and shaving bright Green star peaks (`12.61 → 12.84 px`).
   - Second, replacing that guard with a spectral-filter-matched same-color guard (`r=2` vs. `r=18` on the identical CFA sub-lattice in `Iter 2`, where optical stars have $>69\%$ halo ratio vs. $\sim 0\%$ for single-site defects) *still* yielded `corner_fwhm_px = 15.35 px`! Inspection of `transforms.json` revealed the deeper interaction: in `iter-0-baseline` (unweighted integer shift-and-add *without* temporal sigma clipping), five worm-gear turnaround exposures (`IMG_6077..6082`) cluster within `2–4 px`, smearing unrepaired stationary warm pixels into compact `~4 px` multi-pixel pseudo-stars in the sparse outer corners (`~25` corner stars), artificially depressing `iter-0-baseline`'s apparent `corner_fwhm_px` (`14.66 px` vs. true optical `15.23 px`), while lower sky noise simultaneously lifted faint comatic outer-corner stars (`FWHM ~ 16–18 px`) above $5\sigma$. Once **temporal Winsorized Sigma Clipping (`Iter 7`)** eliminated drifting warm-pixel artifacts orthogonally along the time axis (`32 → 0` warm impulses), applying **pure per-row OB pedestal + Winsorized master dark + thermal dark scaling (`Iter 8`, with zero spatial CFA interpolation)** improved *both* `median_fwhm_px` (`10.9057 → 10.8520 px`) and `corner_fwhm_px` (`15.0433 → 14.8577 px`) for a clean `+0.3722` gain.

2. **6-DOF Full Affine Overfitting Over Short Zenith Tracks (`Iter 5`, `-0.3241`):**
   Upgrading 4-DOF Similarity to 6-DOF Affine tightened interior `median_fwhm_px` (`11.1734 → 11.1018 px`) but degraded `corner_fwhm_px` (`15.3746 → 15.5491 px`) and lost 8 faint stars (`345 → 337`). Across a 10-minute imaging window near transit (`620 s`), physical differential atmospheric refraction shear across a $1.2^\circ$ field is $<0.04\text{ px}$; fitting two unconstrained shear/aspect parameters (`m00 - m11`, `m01 + m10`) to ~55 stars overfits radially asymmetric Newtonian coma centroids and extrapolates edge registration error.

3. **Directional Hamilton–Adams vs. Wiener-Optimal Linear MHC 5×5 on Oversampled PSFs (`Iter 9`, `-0.9509`):**
   Replacing linear MHC $5\times 5$ with two-pass directional gradient-weighted Hamilton–Adams (1997) degraded both sharpness (`10.8520 → 10.9212 px`) and background noise (`0.111904 → 0.113102`). Unlike natural scenes dominated by 1-pixel step edges, prime-focus stellar PSFs here are heavily oversampled ($\text{FWHM} = 9\text{–}15\text{ px}$); omitting MHC's second-derivative cross-channel Laplacian terms (`-1.5 * ax2`, `-h2 + 0.5 * v2`) in Pass 2 softens R/B stellar cores and elevates chrominance-to-luminance noise leakage.

---

## 5. Out-of-Distribution Holdout Generalization (`andromeda_set`, M31 Extended Galaxy)

### 5.1 Zero-Tuning Generalization Across Astrophysical Regimes

Whereas `ring_set` (M57) is a sparse star field containing a compact $1.4'$ planetary nebula at $24 \to 19\,^\circ\text{C}$ with minimal field rotation (`0.021°`), the holdout `andromeda_set` (M31) represents a dramatic domain shift:
1. **Field-Filling Extended Emission:** M31's bright bulge, spiral arms, and dust lanes span the entire $4290 \times 2856$ sensor, stress-testing background estimation and sky-plane equalization against diffuse flux erasure.
2. **Severe Outdoor Thermal Cooling (`34 °C → 21 °C`):** A $16\,^\circ\text{C}$ gap above the $18\,^\circ\text{C}$ master dark generates **`477` surviving warm-pixel impulses** in the unweighted baseline stack (`43×` more than `ring_set`).
3. **Mixed Exposure Durations & $2.6\times$ Larger Field Rotation:** Combines `2 × 29.0 s` + `19 × 28.0 s` exposures across a `56.83 px` drift trajectory with **`0.0561°` monotonic field rotation (`2.52 px` corner arc)**.

Both `iter-0-baseline` and `iter-best-final` were executed on `andromeda_set` + `dark_set` **with zero hyperparameter or algorithmic modifications** ([`6b38b6d2-6e68-4c70-bfe2-5ef7fc560daf`](conversation://6b38b6d2-6e68-4c70-bfe2-5ef7fc560daf) `AUDIT_PASS`):

| Pipeline Version on `andromeda_set` | Cheat? | Composite Score | $\Delta$ vs. `iter-0` | `med_fwhm` (px) | `cor_fwhm` (px) | `ecc` | `norm_bg_rms` | `stars` | `ncc_16x16` / `ncc_64x64` | `rms_ratio_64` | `flux_lin_r` | Warm Imp. | $N_{\text{int}}$ ($N_{\text{eff}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single Reference (`IMG_6044.CR2`)** | `False` | `50.0000` | — | `7.7490` | `12.0637` | `0.5520` | `0.452792` | `107` | `1.000000` / `1.000000` | `1.0000` | `1.00000` | — | `1 (1.00)` |
| **`iter-0-baseline` (Unmodified)** | `False` | `111.4353` | `0.0000` | `8.3600` | `12.9533` | `0.6243` | `0.100537` | `249` | `0.994483` / `0.999596` | `1.0021` | `0.99995` | `477` | `21 (21.00)` |
| **`iter-best-final` (Zero Tuning)** | `False` | **`114.8796`** | **`+3.4443`** | **`8.2000`** | **`12.5927`** | `0.6258` | **`0.096736`** | **`264`** | **`0.995794`** / **`0.999639`** | **`1.0016`** | **`0.99995`** | **`0`** | `21 (20.42)` |

### 5.2 Physical Analysis of Holdout Results & Coordinate-Origin Anchoring Note

- **Near-Exact Transfer of Optimization Gain (`+3.4443` Test vs. `+3.5098` Train):** Every physical improvement transferred cleanly out-of-distribution:
  - **Sub-Ensemble PSF Sharpness & Field-Rotation Correction:** `median_fwhm_px` improved to **`8.2000 px`** (beating the 21-exposure single-frame median `8.2791 px`), while 4-DOF Valdes triangle + Procrustes similarity warping reduced `corner_fwhm_px` by **`-0.3606 px`** (`12.9533 → 12.5927 px`) across M31's `2.52 px` corner rotation arc (`64–70` star inliers/frame, `0.301–0.419 px` astrometric RMS; measured `median_nonref_shift_px = 17.5556 px` vs. Phase 2 ground truth `17.5459 px`).
  - **Complete Thermal Defect Elimination (`477 → 0`):** Combining per-frame hot-site thermal dark scaling (`Iter 8`) with temporal Winsorized Sigma Clipping (`Iter 7`) eliminated $100\%$ of the `477` high-temperature (`34 °C`) warm-pixel impulses without spatial smoothing (`block_4x4_ratio = 1.3811`, `dark_clipping_triggered = false` at `2.3e-6`), lowering `normalized_bg_rms` by $4.68\times$ relative to a single frame (`0.452792 → 0.096736`).
  - **100.16% Diffuse Galactic Disk Fidelity:** Fitting the 1st-degree differential sky plane to $\frac{1}{F_i}I_i - I_{\text{ref}}$ (`Iter 6`) preserved M31's extended disk with `ncc_16x16 = 0.995794`, `ncc_64x64 = 0.999639`, `structure_rms_ratio_16x16 = 0.9975`, and `structure_rms_ratio_64x64 = 1.0016`.
- **Coordinate-Origin Gauge Anchoring (`--ref-frame`):** Because locked [`evaluate.py`](file:///home/julien/projects/astro_stacker/scripts/evaluate.py#L869) cross-matches stars within a fixed $\le 4.0\text{ px}$ sensor-coordinate radius around `--ref` (`IMG_6044.CR2`) without fitting an inter-image shift, an initial unanchored holdout invocation ([`eval_andromeda_best_unanchored.json`](file:///home/julien/projects/astro_stacker/logs/eval_andromeda_best_unanchored.json)) autonomously selected `IMG_6046.CR2` (`t = 200 s`), placing `IMG_6044.CR2` at `(dx = -1.5127, dy = +4.6577) px` ($\|\Delta\mathbf{x}\|_2 = 4.8972\text{ px} > 4.0\text{ px}$—just $0.90\text{ px}$ beyond the coordinate-chart gate despite clean 21-frame alignment; `iter-0-baseline` happened to select `IMG_6045.CR2` at $1.435\text{ px} < 4.0\text{ px}$). Adding a pure CLI coordinate-origin flag `--ref-frame IMG_6044.CR2` (`+17/-2` lines in [`main.cpp`](file:///home/julien/projects/astro_stacker/cpp_stacker/src/main.cpp#L124-L140), audited `AUDIT_PASS` with zero changes to any calibration, registration, weighting, or integration math) aligned the output canvas origin to `IMG_6044.CR2` for direct pixel-locked evaluation.

---

## 6. C++20/OpenMP Systems Architecture, Performance & Conclusions

### 6.1 Modular C++20 Architecture & Stage-by-Stage Throughput

The production engine in [`cpp_stacker/`](file:///home/julien/projects/astro_stacker/cpp_stacker) comprises `2,782` lines of strict `-Wall -Wextra -pedantic` C++20 compiled with `g++ 15.2.0` (`-O3 -march=native -fopenmp -lraw_r`) into a `262 KiB` standalone executable with native IEEE-754 big-endian FITS, 32-bit float TIFF, 16-bit `asinh`-preview TIFF, and NumPy `.npy` v1.0 writers ([`io_export.cpp`](file:///home/julien/projects/astro_stacker/cpp_stacker/src/io_export.cpp)):

| Pipeline Stage (`16` OpenMP Threads) | Key Algorithmic Complexity per Run | `ring_set` (`19L + 18D`) | `andromeda_set` (`21L + 18D`) |
| :--- | :--- | ---: | ---: |
| **1. `LibRaw` Unpack & Dual ADC Gate** | `41` `.CR2` lossless-JPEG decodes + 16-bit histograms (`502M` raw px) | `2.84 – 7.49 s` | `3.08 – 6.63 s` |
| **2. Temporal Winsorized Master Dark** | `18 × 12.25M` CFA sites, 33-row OB median + 3-iter $\kappa=3.0$ clip | `2.18 – 3.24 s` | `2.84 – 3.20 s` |
| **3. Calibrate, MHC 5×5 & 2D-LM Stars** | Hot-site $\alpha_{\text{th}}$ + `5×5` MHC + `64×64` SEP + `~600 × 7-DOF` LM fits/frame | `46.58 – 55.59 s` | `54.24 s` |
| **4. Valdes Asterism + 4-DOF IRLS** | `~5,000` chiral triangles/frame (`64×64` hash) + RANSAC + 5-pass IRLS | **`0.036 s`** | **`0.033 s`** |
| **5. Sky Plane, Lanczos-3 & Winsorized Clip** | `3-DOF` diff planes + `21 × 36.76M` `6×6` LUT Lanczos-3 + $\kappa=2.8$ clip | `15.34 – 21.62 s` | `22.55 s` |
| **6. Scientific Export (`.npy` + `.fits` + 2×`.tif`)** | `515 MiB` formatted scientific & stretched preview disk I/O | `1.89 – 3.54 s` | `2.78 s` |
| **Total End-to-End Wall-Clock Time** | Complete raw `.CR2` directory $\to$ calibrated scientific stack | **`69.94 – 90.84 s`** | **`89.07 s`** |

Notably, **registration across all 21 frames executes in `33 milliseconds`** (`1.6 ms/frame`), replacing the 2020 Python `fftconvolve` / `[-40, 40]^2` `DifferenceCorrelator` bottleneck by $>10^3\times$, while full `6×6` 2D Lanczos-3 warping + per-pixel iterative Winsorized Sigma Clipping over `7.72 × 10^8` interpolated channel samples completes in `22.55 s` via a `4,096`-subdivision L1-cached Lanczos lookup table and zero-allocation register insertion sorts.

### 6.2 Key Scientific & Methodological Conclusions

1. **Point-Source Geometric Hashing Defeats Coherent Sensor Systematics:** Extracting background-subtracted stellar centroids ($<0.2\%$ of pixels) and matching chirality-preserved Valdes triangle invariants completely decouples astrometry from zero-lag master-dark estimation variance, optical vignetting bowls, worm-gear velocity reversals, and celestial field rotation.
2. **Algorithmic Evolution Is Non-Commutative:** Single-frame spatial defect repair interacts nonlinearly with downstream multi-frame integration; deferring outlier rejection to orthogonal 1D temporal Winsorized Sigma Clipping unlocked pure physical sensor calibration (`+0.56` combined) where premature spatial repair had regressed (`-0.79`).
3. **Blind Physics Referees + Isolated Red-Team Auditors Prevent Goodhart's Law:** Pairing an immutable, torture-tested photometric referee (`Gates 0–5`) with per-iteration independent code/telemetry auditors enables fully autonomous scientific software engineering—achieving `+3.51` training gain and `+3.44` zero-shot out-of-distribution generalization without a single image ever entering a language model context window.

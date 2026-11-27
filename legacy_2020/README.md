# Legacy 2020 Prototype (`legacy_2020/`)

> **Archived / Unmaintained:** This directory preserves the original 2020 Python prototype of `astro_stacker`. It has been superseded by the C++20/OpenMP engine in [`cpp_stacker/`](../cpp_stacker/).

## Why This Code Is Kept

This codebase is retained intact as a historical reference for the accompanying research paper ([`paper/astro_stacker_paper.pdf`](../paper/astro_stacker_paper.pdf)), which performs a mathematical and empirical postmortem of this prototype's full-frame spatial cross-correlation registration approach ($C_{\bm{k}}$) and its failure modes (zero-lag master-dark estimation variance, stationary optical vignetting bowls, sparse stellar occupancy, non-uniform worm-gear drift, and celestial field rotation).

## Contents

- `astro_stacker/` — Original Python 3.8 package (`rawpy`, `numpy`, `scipy`, `scikit-image`, `lmfit`)
  - `image_readers/` — Canon raw file reader
  - `processing/` — Spatial cross-correlator, shift analyzer, difference correlator, variance analyzer
  - `curve_fitters/` — Gaussian PSF fitting utilities
  - `algorithms/` — Savitzky–Golay smoothing
  - `models/` — Image, ROI, and series data models
- `tests/` — Unit tests for the Gaussian curve fitter
- `pyproject.toml`, `poetry.lock`, `tasks.py` — Original Poetry 1.x / Python 3.8 environment definition

## Historical Usage (Reference Only)

This prototype targeted Python 3.8 and Poetry 1.x and is not maintained for modern Python environments. The original example from 2020 was:

```python
from astro_stacker.config import Config
from astro_stacker.models.cannon_raw_file_group import CannonRawFileGroup
from astro_stacker.greyscale_image_processor import CannonGreyScaleImageProcessor

image_filenames = CannonRawFileGroup.from_directory(Config.STACK_1_DIR)
darks = CannonRawFileGroup.from_directory(Config.DARK_DIR)

processor = CannonGreyScaleImageProcessor(darks)
average_image = processor.compute_average()

histogram = average_image.histogram()
```

For building and running the current stacking engine, see the [root README](../README.md).

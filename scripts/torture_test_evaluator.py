#!/usr/bin/env python3
"""Phase 2 Task 3: Independent Adversarial Torture-Test Suite for scripts/evaluate.py.

Tests all mandatory anti-cheat & baseline cases (a)-(e) and red-team probes (f)-(k)
across BOTH ring_set (logs/ref_ring.npy) and andromeda_set (logs/ref_andromeda.npy),
invoking scripts/evaluate.py strictly as an unmodified black-box CLI subprocess.

All temporary synthetic .npy arrays, C++ LibRaw calibration shims, and mock
transforms.json files live strictly inside a tempfile.TemporaryDirectory().
"""

from __future__ import annotations

import ast
import ctypes
from datetime import datetime, timezone
import hashlib
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

# Import CppRefShim and strict constants from Phase 2 Task 1 generator
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_references import (  # noqa: E402
    ACTIVE_H,
    ACTIVE_W,
    CandidateObjC,
    CppRefShim,
    FrameIntegrityMetaC,
)


WORKSPACE_ROOT = SCRIPTS_DIR.parent
EVALUATE_SCRIPT = WORKSPACE_ROOT / "scripts" / "evaluate.py"
REF_SELECTION_JSON = WORKSPACE_ROOT / "logs" / "reference_selection.json"
DATASET_BASE_DIR = WORKSPACE_ROOT / "pictures" / "2020-11-28-deep-sky-ring-andromeda"
RESULTS_OUT_JSON = WORKSPACE_ROOT / "logs" / "torture_test_results.json"

DATASETS = [
    {
        "name": "ring_set",
        "ref_path": WORKSPACE_ROOT / "logs" / "ref_ring.npy",
        "raw_dir": DATASET_BASE_DIR / "ring_set",
        "expected_clean_frames": 18,
    },
    {
        "name": "andromeda_set",
        "ref_path": WORKSPACE_ROOT / "logs" / "ref_andromeda.npy",
        "raw_dir": DATASET_BASE_DIR / "andromeda_set",
        "expected_clean_frames": 21,
    },
]


def strict_cr2_files(directory: Path) -> list[Path]:
    """Strictly filter for .CR2 files (len(f) > 4 and f.upper().endswith('.CR2'))."""
    out: list[Path] = []
    for fname in sorted(os.listdir(directory)):
        if len(fname) > 4 and fname.upper().endswith(".CR2"):
            fpath = directory / fname
            if fpath.is_file():
                out.append(fpath)
    return out


def audit_evaluator_dependencies(eval_path: Path) -> dict[str, Any]:
    """Audit scripts/evaluate.py via AST to verify zero external local .py or .so imports."""
    source_text = eval_path.read_text(encoding="utf-8")
    sha256_hex = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    tree = ast.parse(source_text, filename=str(eval_path))

    stdlib_names = set(sys.stdlib_module_names) | {"__future__"}
    imported_modules: list[dict[str, Any]] = []
    external_local_imports: list[str] = []
    dynamic_import_or_ffi_calls: list[str] = []

    banned_call_names = {"__import__", "eval", "exec", "compile"}
    banned_attr_roots = {"ctypes", "cffi", "importlib", "subprocess"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                is_std = top in stdlib_names
                is_np = top == "numpy"
                imported_modules.append(
                    {
                        "statement": f"import {alias.name}",
                        "lineno": node.lineno,
                        "top_module": top,
                        "category": "stdlib" if is_std else ("numpy" if is_np else "external_or_local"),
                    }
                )
                if not (is_std or is_np):
                    external_local_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            top = mod.split(".")[0] if node.level == 0 else f".{mod}"
            is_std = (node.level == 0) and (top in stdlib_names)
            is_np = (node.level == 0) and (top == "numpy")
            names_str = ", ".join(a.name for a in node.names)
            imported_modules.append(
                {
                    "statement": f"from {'.' * node.level}{mod} import {names_str}",
                    "lineno": node.lineno,
                    "top_module": top,
                    "category": "stdlib" if is_std else ("numpy" if is_np else "external_or_local"),
                }
            )
            if not (is_std or is_np):
                external_local_imports.append(f"{'.' * node.level}{mod}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in banned_call_names:
                dynamic_import_or_ffi_calls.append(f"{node.func.id}@L{node.lineno}")
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id in banned_attr_roots:
                    dynamic_import_or_ffi_calls.append(
                        f"{node.func.value.id}.{node.func.attr}@L{node.lineno}"
                    )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            val_low = node.value.lower()
            if val_low.endswith(".so") or val_low.endswith(".dll") or val_low.endswith(".dylib"):
                dynamic_import_or_ffi_calls.append(f"shared_lib_literal({node.value})@L{node.lineno}")

    self_contained_ok = (len(external_local_imports) == 0) and (len(dynamic_import_or_ffi_calls) == 0)
    return {
        "evaluate_script_path": str(eval_path),
        "sha256": sha256_hex,
        "line_count": len(source_text.splitlines()),
        "byte_size": len(source_text.encode("utf-8")),
        "imported_modules": imported_modules,
        "external_or_local_py_imports": external_local_imports,
        "dynamic_import_or_so_ffi_references": dynamic_import_or_ffi_calls,
        "strictly_self_contained_stdlib_plus_numpy": self_contained_ok,
    }


def gaussian_blur_rgb(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Isotropic 2D Gaussian convolution per RGB channel using exact separable 1D kernel."""
    radius = max(1, int(math.ceil(4.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= np.sum(kernel)

    h, w, _ = arr.shape
    pad_x = np.pad(arr.astype(np.float64), ((0, 0), (radius, radius), (0, 0)), mode="reflect")
    horiz = np.zeros((h, w, 3), dtype=np.float64)
    for i, weight in enumerate(kernel):
        horiz += weight * pad_x[:, i : i + w, :]

    pad_y = np.pad(horiz, ((radius, radius), (0, 0), (0, 0)), mode="reflect")
    out = np.zeros((h, w, 3), dtype=np.float64)
    for i, weight in enumerate(kernel):
        out += weight * pad_y[i : i + h, :, :]

    return out.astype(np.float32)


def subtract_64x64_median_mesh_rgb(arr: np.ndarray, cell_size: int = 64) -> np.ndarray:
    """Subtract a smooth bilinearly-interpolated 64x64 median background mesh per channel."""
    h, w, c = arr.shape
    bh = (h + cell_size - 1) // cell_size
    bw = (w + cell_size - 1) // cell_size
    pad_h = bh * cell_size - h
    pad_w = bw * cell_size - w

    padded = np.pad(arr.astype(np.float64), ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    cells = padded.reshape(bh, cell_size, bw, cell_size, c).transpose(0, 2, 1, 3, 4)
    cell_meds = np.median(cells.reshape(bh, bw, cell_size * cell_size, c), axis=2)  # (bh, bw, 3)

    gy = np.clip(
        (np.arange(h, dtype=np.float64) - (cell_size * 0.5 - 0.5)) / cell_size,
        0.0,
        bh - 1.000001,
    )
    gx = np.clip(
        (np.arange(w, dtype=np.float64) - (cell_size * 0.5 - 0.5)) / cell_size,
        0.0,
        bw - 1.000001,
    )
    iy = gy.astype(np.int32)
    ix = gx.astype(np.int32)
    fy = (gy - iy)[:, None, None]
    fx = (gx - ix)[None, :, None]

    mesh = (
        (1.0 - fy) * (1.0 - fx) * cell_meds[iy[:, None], ix[None, :]]
        + (1.0 - fy) * fx * cell_meds[iy[:, None], ix[None, :] + 1]
        + fy * (1.0 - fx) * cell_meds[iy[:, None] + 1, ix[None, :]]
        + fy * fx * cell_meds[iy[:, None] + 1, ix[None, :] + 1]
    )
    # Preserve original channel median offset so sky median stays in normal positive ADU range
    orig_meds = np.median(arr.astype(np.float64), axis=(0, 1), keepdims=True)
    return (arr.astype(np.float64) - mesh + orig_meds).astype(np.float32)


def run_evaluator_cli(
    stack_npy: Path,
    ref_npy: Path,
    transforms_json: Path | None = None,
    single_frame_baseline: bool = False,
) -> dict[str, Any]:
    """Invoke scripts/evaluate.py via python3 subprocess and parse JSON output."""
    cmd = [
        sys.executable,
        str(EVALUATE_SCRIPT),
        "--stack",
        str(stack_npy),
        "--ref",
        str(ref_npy),
    ]
    if single_frame_baseline:
        cmd.append("--single-frame-baseline")
    if transforms_json is not None:
        cmd.extend(["--transforms", str(transforms_json)])

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    wall_s = round(time.perf_counter() - t0, 4)

    if proc.returncode != 0:
        raise RuntimeError(
            f"evaluate.py exited with non-zero code {proc.returncode}:\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
        )

    parsed = json.loads(proc.stdout)
    parsed["_cli_wall_s"] = wall_s
    parsed["_cli_returncode"] = proc.returncode
    return parsed


def build_realistic_transforms_payload(drift_table: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert reference_selection.json per_frame_drift_table into transforms.json schema."""
    frames = []
    for row in drift_table:
        frames.append(
            {
                "filename": row["filename"],
                "integrated": True,
                "weight": 1.0,
                "is_reference": bool(row["is_reference"]),
                "dx": float(row["dx_px"]),
                "dy": float(row["dy_px"]),
                "dtheta_deg": float(row["dtheta_deg"]),
            }
        )
    return {"frames": frames}


def build_zero_transforms_payload(filenames: list[str]) -> dict[str, Any]:
    """Build a mock transforms.json where all frames have (dx=0, dy=0, dtheta=0)."""
    frames = []
    for i, fname in enumerate(filenames):
        frames.append(
            {
                "filename": fname,
                "integrated": True,
                "weight": 1.0,
                "is_reference": (i == 0),
                "dx": 0.0,
                "dy": 0.0,
                "dtheta_deg": 0.0,
            }
        )
    return {"frames": frames}


def calibrate_and_demosaic_clean_frames(
    shim: CppRefShim,
    master_dark: np.ndarray,
    raw_dir: Path,
) -> dict[str, np.ndarray]:
    """Load, verify cleanliness, dark-subtract, and MHC-5x5 demosaic all clean .CR2 files."""
    cr2_paths = strict_cr2_files(raw_dir)
    md_ptr = master_dark.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    dummy_cand = (CandidateObjC * 1)()

    clean_rgb_by_name: dict[str, np.ndarray] = {}
    for fpath in cr2_paths:
        meta = FrameIntegrityMetaC()
        bg_rms = ctypes.c_double(0.0)
        bg_med = ctypes.c_double(0.0)
        tot_5sig = ctypes.c_int(0)
        tot_sat = ctypes.c_int(0)
        rgb = np.empty((ACTIVE_H, ACTIVE_W, 3), dtype=np.float32)

        shim.lib.analyze_light_frame(
            os.fsencode(str(fpath)),
            md_ptr,
            ctypes.byref(meta),
            ctypes.byref(bg_rms),
            ctypes.byref(bg_med),
            dummy_cand,
            0,  # max_cands=0 skips Gaussian star fitting, running pure calibration + MHC 5x5
            ctypes.byref(tot_5sig),
            ctypes.byref(tot_sat),
            rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if bool(meta.is_clean):
            clean_rgb_by_name[fpath.name] = rgb
    return clean_rgb_by_name


def build_integer_shift_and_add_stack(
    clean_rgb_by_name: dict[str, np.ndarray],
    drift_table: list[dict[str, Any]],
) -> np.ndarray:
    """Build an honest linear integer shift-and-add stack aligned to the winning reference."""
    accum = np.zeros((ACTIVE_H, ACTIVE_W, 3), dtype=np.float64)
    y_idx = np.arange(ACTIVE_H, dtype=np.int32)
    x_idx = np.arange(ACTIVE_W, dtype=np.int32)

    count = 0
    for row in drift_table:
        fname = row["filename"]
        rgb = clean_rgb_by_name[fname]
        shift_x = int(round(float(row["dx_px"])))
        shift_y = int(round(float(row["dy_px"])))
        src_y = np.clip(y_idx + shift_y, 0, ACTIVE_H - 1)
        src_x = np.clip(x_idx + shift_x, 0, ACTIVE_W - 1)
        accum += rgb[src_y[:, None], src_x[None, :], :]
        count += 1

    if count == 0:
        raise RuntimeError("No frames accumulated in honest stack")
    return (accum / float(count)).astype(np.float32)


def summarize_eval_record(
    case_id: str,
    case_label: str,
    category: str,
    dataset_name: str,
    mode: str,
    expected_cheating: bool,
    score_expectation: str,
    assertion_passed: bool,
    eval_out: dict[str, Any],
) -> dict[str, Any]:
    """Package concise + diagnostic evaluation output for torture_test_results.json."""
    return {
        "case_id": case_id,
        "case_label": case_label,
        "category": category,
        "dataset": dataset_name,
        "mode": mode,
        "expected": {
            "cheating_detected": expected_cheating,
            "score_rule": score_expectation,
        },
        "observed": {
            "cheating_detected": bool(eval_out["cheating_detected"]),
            "score": eval_out["score"],
            "composite_score": eval_out.get("composite_score"),
            "cheat_reasons": eval_out.get("cheat_reasons", []),
            "core_metrics": eval_out.get("core_metrics"),
            "score_breakdown": eval_out.get("score_breakdown"),
            "photometric_normalization": eval_out.get("photometric_normalization"),
            "anti_cheat_diagnostics": eval_out.get("anti_cheat_diagnostics"),
            "evaluator_runtime_s": eval_out.get("runtime_s"),
            "cli_wall_s": eval_out.get("_cli_wall_s"),
        },
        "assertion_passed": bool(assertion_passed),
    }


def main() -> int:
    t_suite_start = time.perf_counter()

    # 1. Static Immutability & Dependency Audit on scripts/evaluate.py
    dep_audit = audit_evaluator_dependencies(EVALUATE_SCRIPT)
    eval_sha256_before = dep_audit["sha256"]

    with open(REF_SELECTION_JSON, "r", encoding="utf-8") as fh:
        ref_selection = json.load(fh)

    all_test_records: list[dict[str, Any]] = []
    temp_dir_cleaned_verified = False
    temp_dir_path_recorded = ""

    with tempfile.TemporaryDirectory(prefix="astro_torture_test_") as tmp_str:
        tmp_dir = Path(tmp_str)
        temp_dir_path_recorded = str(tmp_dir)

        # Compile inline C++20 + OpenMP + LibRaw calibration & MHC-5x5 shim inside tmp_dir
        shim = CppRefShim(str(tmp_dir))

        # Build linear master dark from clean dark_set .CR2 files
        dark_dir = DATASET_BASE_DIR / "dark_set"
        dark_cr2s = strict_cr2_files(dark_dir)
        n_dark = len(dark_cr2s)
        c_dark_paths = (ctypes.c_char_p * n_dark)(*[os.fsencode(str(p)) for p in dark_cr2s])
        dark_metas = (FrameIntegrityMetaC * n_dark)()
        master_dark = np.empty((ACTIVE_H, ACTIVE_W), dtype=np.float32)
        clean_dark_count = shim.lib.build_master_dark(
            n_dark,
            c_dark_paths,
            dark_metas,
            master_dark.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if clean_dark_count != 18:
            raise RuntimeError(f"Expected 18 clean dark frames, got {clean_dark_count}")

        for ds in DATASETS:
            ds_name = ds["name"]
            ref_path = ds["ref_path"]
            raw_dir = ds["raw_dir"]
            expected_clean_n = ds["expected_clean_frames"]

            print(f"\n=== Auditing dataset: {ds_name} ===", flush=True)
            ref_arr = np.load(ref_path)
            drift_table = ref_selection["datasets"][ds_name]["per_frame_drift_table"]
            if len(drift_table) != expected_clean_n:
                raise RuntimeError(
                    f"Drift table count mismatch for {ds_name}: {len(drift_table)} != {expected_clean_n}"
                )

            realistic_trans_path = tmp_dir / f"{ds_name}_realistic_transforms.json"
            realistic_trans_path.write_text(
                json.dumps(build_realistic_transforms_payload(drift_table), indent=2),
                encoding="utf-8",
            )

            # Calibrate and MHC-demosaic all clean .CR2 frames for cases (e), (h), (k)
            t_cal0 = time.perf_counter()
            clean_rgb_map = calibrate_and_demosaic_clean_frames(shim, master_dark, raw_dir)
            cal_s = round(time.perf_counter() - t_cal0, 3)
            if len(clean_rgb_map) != expected_clean_n:
                raise RuntimeError(
                    f"Expected {expected_clean_n} clean .CR2 frames in {ds_name}, got {len(clean_rgb_map)}"
                )
            print(
                f"  Calibrated & MHC-demosaiced {len(clean_rgb_map)} clean .CR2 frames in {cal_s:.2f}s",
                flush=True,
            )

            # ------------------------------------------------------------------
            # (a) Single Reference Frame Baseline
            # ------------------------------------------------------------------
            out_a = run_evaluator_cli(ref_path, ref_path, single_frame_baseline=True)
            pass_a = (
                (out_a["cheating_detected"] is False)
                and math.isfinite(out_a["score"])
                and (out_a["score"] == 50.0)
            )
            all_test_records.append(
                summarize_eval_record(
                    case_id="a",
                    case_label="(a) Single Reference Frame Baseline",
                    category="mandatory",
                    dataset_name=ds_name,
                    mode="single_frame_baseline",
                    expected_cheating=False,
                    score_expectation="score == 50.0 (exact)",
                    assertion_passed=pass_a,
                    eval_out=out_a,
                )
            )
            print(f"  [a] Baseline: cheat={out_a['cheating_detected']} score={out_a['score']} pass={pass_a}")

            # ------------------------------------------------------------------
            # (b) All-Zero Array (both --single-frame-baseline and stack mode)
            # ------------------------------------------------------------------
            zero_path = tmp_dir / f"{ds_name}_zeros.npy"
            np.save(zero_path, np.zeros((ACTIVE_H, ACTIVE_W, 3), dtype=np.float32))

            out_b_base = run_evaluator_cli(zero_path, ref_path, single_frame_baseline=True)
            pass_b_base = (out_b_base["cheating_detected"] is True) and (
                out_b_base["score"] == float("-inf")
            )
            all_test_records.append(
                summarize_eval_record(
                    case_id="b_baseline",
                    case_label="(b) All-Zero Array [single-frame-baseline]",
                    category="mandatory",
                    dataset_name=ds_name,
                    mode="single_frame_baseline",
                    expected_cheating=True,
                    score_expectation="score == -inf",
                    assertion_passed=pass_b_base,
                    eval_out=out_b_base,
                )
            )

            out_b_stack = run_evaluator_cli(
                zero_path, ref_path, transforms_json=realistic_trans_path, single_frame_baseline=False
            )
            pass_b_stack = (out_b_stack["cheating_detected"] is True) and (
                out_b_stack["score"] == float("-inf")
            )
            all_test_records.append(
                summarize_eval_record(
                    case_id="b_stack",
                    case_label=f"(b) All-Zero Array [stack mode w/ {expected_clean_n}-frame transforms]",
                    category="mandatory",
                    dataset_name=ds_name,
                    mode="stack",
                    expected_cheating=True,
                    score_expectation="score == -inf",
                    assertion_passed=pass_b_stack,
                    eval_out=out_b_stack,
                )
            )
            zero_path.unlink()
            print(f"  [b] All-Zero: base_pass={pass_b_base} stack_pass={pass_b_stack}")

            # ------------------------------------------------------------------
            # (c) Reference Frame Blurred with sigma = 2.0 px (both modes)
            # ------------------------------------------------------------------
            blur2_path = tmp_dir / f"{ds_name}_blur_sigma2.npy"
            np.save(blur2_path, gaussian_blur_rgb(ref_arr, sigma=2.0))

            out_c_base = run_evaluator_cli(blur2_path, ref_path, single_frame_baseline=True)
            pass_c_base = (out_c_base["cheating_detected"] is True) and (
                out_c_base["score"] == float("-inf")
            )
            all_test_records.append(
                summarize_eval_record(
                    case_id="c_baseline",
                    case_label="(c) Reference Blurred sigma=2.0 px [single-frame-baseline]",
                    category="mandatory",
                    dataset_name=ds_name,
                    mode="single_frame_baseline",
                    expected_cheating=True,
                    score_expectation="score == -inf",
                    assertion_passed=pass_c_base,
                    eval_out=out_c_base,
                )
            )

            out_c_stack = run_evaluator_cli(
                blur2_path, ref_path, transforms_json=realistic_trans_path, single_frame_baseline=False
            )
            pass_c_stack = (out_c_stack["cheating_detected"] is True) and (
                out_c_stack["score"] == float("-inf")
            )
            all_test_records.append(
                summarize_eval_record(
                    case_id="c_stack",
                    case_label=f"(c) Reference Blurred sigma=2.0 px [stack mode w/ {expected_clean_n}-frame transforms]",
                    category="mandatory",
                    dataset_name=ds_name,
                    mode="stack",
                    expected_cheating=True,
                    score_expectation="score == -inf",
                    assertion_passed=pass_c_stack,
                    eval_out=out_c_stack,
                )
            )
            blur2_path.unlink()
            print(f"  [c] Blur sigma=2.0: base_pass={pass_c_base} stack_pass={pass_c_stack}")

            # ------------------------------------------------------------------
            # (d) Reference Frame with Dark Pixels Clipped at Median (both modes)
            # ------------------------------------------------------------------
            chan_medians = np.median(ref_arr, axis=(0, 1), keepdims=True).astype(np.float32)
            clipped_med_arr = np.maximum(ref_arr, chan_medians).astype(np.float32)
            clip_path = tmp_dir / f"{ds_name}_clipped_median.npy"
            np.save(clip_path, clipped_med_arr)

            out_d_base = run_evaluator_cli(clip_path, ref_path, single_frame_baseline=True)
            pass_d_base = (out_d_base["cheating_detected"] is True) and (
                out_d_base["score"] == float("-inf")
            )
            all_test_records.append(
                summarize_eval_record(
                    case_id="d_baseline",
                    case_label="(d) Dark Pixels Clipped at Per-Channel Median [single-frame-baseline]",
                    category="mandatory",
                    dataset_name=ds_name,
                    mode="single_frame_baseline",
                    expected_cheating=True,
                    score_expectation="score == -inf",
                    assertion_passed=pass_d_base,
                    eval_out=out_d_base,
                )
            )

            out_d_stack = run_evaluator_cli(
                clip_path, ref_path, transforms_json=realistic_trans_path, single_frame_baseline=False
            )
            pass_d_stack = (out_d_stack["cheating_detected"] is True) and (
                out_d_stack["score"] == float("-inf")
            )
            all_test_records.append(
                summarize_eval_record(
                    case_id="d_stack",
                    case_label=f"(d) Dark Pixels Clipped at Per-Channel Median [stack mode w/ {expected_clean_n}-frame transforms]",
                    category="mandatory",
                    dataset_name=ds_name,
                    mode="stack",
                    expected_cheating=True,
                    score_expectation="score == -inf",
                    assertion_passed=pass_d_stack,
                    eval_out=out_d_stack,
                )
            )
            clip_path.unlink()
            print(f"  [d] Median Dark-Clip: base_pass={pass_d_base} stack_pass={pass_d_stack}")

            # ------------------------------------------------------------------
            # (e) Unaligned Stack of 10 Clean .CR2 Frames w/ Zero Transforms (0,0,0)
            # ------------------------------------------------------------------
            chronological_clean_names = [row["filename"] for row in drift_table]
            first_10_names = chronological_clean_names[:10]
            unaligned_10 = (
                sum(clean_rgb_map[nm].astype(np.float64) for nm in first_10_names) / 10.0
            ).astype(np.float32)

            unaligned_10_path = tmp_dir / f"{ds_name}_unaligned_10.npy"
            zero_10_trans_path = tmp_dir / f"{ds_name}_zero_10_transforms.json"
            np.save(unaligned_10_path, unaligned_10)
            zero_10_trans_path.write_text(
                json.dumps(build_zero_transforms_payload(first_10_names), indent=2),
                encoding="utf-8",
            )

            out_e = run_evaluator_cli(
                unaligned_10_path,
                ref_path,
                transforms_json=zero_10_trans_path,
                single_frame_baseline=False,
            )
            pass_e = (out_e["cheating_detected"] is True) and (out_e["score"] == float("-inf"))
            all_test_records.append(
                summarize_eval_record(
                    case_id="e",
                    case_label="(e) Unaligned 10-Frame .CR2 Stack w/ Zero Transforms (0,0,0)",
                    category="mandatory",
                    dataset_name=ds_name,
                    mode="stack",
                    expected_cheating=True,
                    score_expectation="score == -inf",
                    assertion_passed=pass_e,
                    eval_out=out_e,
                )
            )
            unaligned_10_path.unlink()
            zero_10_trans_path.unlink()
            print(f"  [e] Unaligned 10-Frame .CR2 Stack (0,0,0): pass={pass_e}")

            # ------------------------------------------------------------------
            # (f) Scale-Invariance Probe (Valid): <ref.npy> * 0.0037 + 42.0
            # ------------------------------------------------------------------
            scaled_ref = (ref_arr.astype(np.float64) * 0.0037 + 42.0).astype(np.float32)
            scaled_path = tmp_dir / f"{ds_name}_scaled_ref.npy"
            np.save(scaled_path, scaled_ref)

            out_f = run_evaluator_cli(scaled_path, ref_path, single_frame_baseline=True)
            # Check exact or float32-quantization-bounded invariance around 50.0
            pass_f = (
                (out_f["cheating_detected"] is False)
                and math.isfinite(out_f["score"])
                and (abs(out_f["score"] - 50.0) <= 1e-2)
            )
            all_test_records.append(
                summarize_eval_record(
                    case_id="f",
                    case_label="(f) Scale-Invariance Probe (ref * 0.0037 + 42.0) [Valid]",
                    category="red_team_bonus",
                    dataset_name=ds_name,
                    mode="single_frame_baseline",
                    expected_cheating=False,
                    score_expectation="score == 50.0 (within float32 ULP rounding)",
                    assertion_passed=pass_f,
                    eval_out=out_f,
                )
            )
            scaled_path.unlink()
            print(
                f"  [f] Scale-Invariance (ref*0.0037+42): cheat={out_f['cheating_detected']} "
                f"score={out_f['score']} pass={pass_f}"
            )

            # ------------------------------------------------------------------
            # (g) Forged transforms.json Frame-Drop Probe (1-frame ref + full transforms)
            # ------------------------------------------------------------------
            out_g = run_evaluator_cli(
                ref_path,
                ref_path,
                transforms_json=realistic_trans_path,
                single_frame_baseline=False,
            )
            pass_g = (out_g["cheating_detected"] is True) and (out_g["score"] == float("-inf"))
            all_test_records.append(
                summarize_eval_record(
                    case_id="g",
                    case_label=f"(g) Forged transforms.json Frame-Drop Probe (1-frame w/ {expected_clean_n}-frame transforms)",
                    category="red_team_bonus",
                    dataset_name=ds_name,
                    mode="stack",
                    expected_cheating=True,
                    score_expectation="score == -inf",
                    assertion_passed=pass_g,
                    eval_out=out_g,
                )
            )
            print(f"  [g] Forged Frame-Drop Probe: pass={pass_g}")

            # Build Honest Multi-Frame Integer Shift-and-Add Stack (used by both (h) and (k))
            honest_stack = build_integer_shift_and_add_stack(clean_rgb_map, drift_table)

            # ------------------------------------------------------------------
            # (h) Subtle sigma = 0.8 px Hidden Background Smoothing Probe on Honest Stack
            # ------------------------------------------------------------------
            subtle_blur_stack = gaussian_blur_rgb(honest_stack, sigma=0.8)
            subtle_blur_path = tmp_dir / f"{ds_name}_honest_blur_0p8.npy"
            np.save(subtle_blur_path, subtle_blur_stack)

            out_h = run_evaluator_cli(
                subtle_blur_path,
                ref_path,
                transforms_json=realistic_trans_path,
                single_frame_baseline=False,
            )
            pass_h = (out_h["cheating_detected"] is True) and (out_h["score"] == float("-inf"))
            all_test_records.append(
                summarize_eval_record(
                    case_id="h",
                    case_label=f"(h) Subtle sigma=0.8 px Hidden Smoothing on {expected_clean_n}-Frame Stack",
                    category="red_team_bonus",
                    dataset_name=ds_name,
                    mode="stack",
                    expected_cheating=True,
                    score_expectation="score == -inf",
                    assertion_passed=pass_h,
                    eval_out=out_h,
                )
            )
            subtle_blur_path.unlink()
            print(f"  [h] Subtle sigma=0.8 Smoothing Probe: pass={pass_h}")

            # ------------------------------------------------------------------
            # (i) Over-Aggressive 64x64 Background Mesh Erasure Probe
            # ------------------------------------------------------------------
            mesh_erased_ref = subtract_64x64_median_mesh_rgb(ref_arr, cell_size=64)
            mesh_erased_path = tmp_dir / f"{ds_name}_mesh64_erased.npy"
            np.save(mesh_erased_path, mesh_erased_ref)

            out_i = run_evaluator_cli(mesh_erased_path, ref_path, single_frame_baseline=True)
            pass_i = (out_i["cheating_detected"] is True) and (out_i["score"] == float("-inf"))
            all_test_records.append(
                summarize_eval_record(
                    case_id="i",
                    case_label="(i) Over-Aggressive 64x64 Median Background Mesh Erasure Probe",
                    category="red_team_bonus",
                    dataset_name=ds_name,
                    mode="single_frame_baseline",
                    expected_cheating=True,
                    score_expectation="score == -inf",
                    assertion_passed=pass_i,
                    eval_out=out_i,
                )
            )
            mesh_erased_path.unlink()
            print(f"  [i] 64x64 Background Mesh Erasure Probe: pass={pass_i}")

            # ------------------------------------------------------------------
            # (j) Non-Linear Gamma Stretch Probe: (stack - min)^0.5
            # ------------------------------------------------------------------
            s_min = float(np.min(honest_stack))
            gamma_stretched = np.sqrt(np.maximum(0.0, honest_stack.astype(np.float64) - s_min)).astype(
                np.float32
            )
            gamma_path = tmp_dir / f"{ds_name}_gamma_sqrt.npy"
            np.save(gamma_path, gamma_stretched)

            out_j = run_evaluator_cli(
                gamma_path,
                ref_path,
                transforms_json=realistic_trans_path,
                single_frame_baseline=False,
            )
            pass_j = (out_j["cheating_detected"] is True) and (out_j["score"] == float("-inf"))
            all_test_records.append(
                summarize_eval_record(
                    case_id="j",
                    case_label="(j) Non-Linear Gamma Stretch Probe ((stack - min)^0.5)",
                    category="red_team_bonus",
                    dataset_name=ds_name,
                    mode="stack",
                    expected_cheating=True,
                    score_expectation="score == -inf",
                    assertion_passed=pass_j,
                    eval_out=out_j,
                )
            )
            gamma_path.unlink()
            print(f"  [j] Non-Linear Gamma Stretch Probe: pass={pass_j}")

            # ------------------------------------------------------------------
            # (k) Honest Multi-Frame Baseline Control (18/21-frame integer shift-and-add)
            # ------------------------------------------------------------------
            honest_path = tmp_dir / f"{ds_name}_honest_stack.npy"
            np.save(honest_path, honest_stack)

            out_k = run_evaluator_cli(
                honest_path,
                ref_path,
                transforms_json=realistic_trans_path,
                single_frame_baseline=False,
            )
            pass_k = (
                (out_k["cheating_detected"] is False)
                and math.isfinite(out_k["score"])
                and (out_k["score"] > 50.0)
            )
            all_test_records.append(
                summarize_eval_record(
                    case_id="k",
                    case_label=f"(k) Honest {expected_clean_n}-Frame Integer Shift-and-Add Control",
                    category="red_team_bonus",
                    dataset_name=ds_name,
                    mode="stack",
                    expected_cheating=False,
                    score_expectation="finite score > 50.0",
                    assertion_passed=pass_k,
                    eval_out=out_k,
                )
            )
            honest_path.unlink()
            print(
                f"  [k] Honest {expected_clean_n}-Frame Control: cheat={out_k['cheating_detected']} "
                f"score={out_k['score']} pass={pass_k}",
                flush=True,
            )

    # Verify TemporaryDirectory cleanup and that evaluate.py was completely untouched
    temp_dir_cleaned_verified = not Path(temp_dir_path_recorded).exists()
    eval_sha256_after = hashlib.sha256(EVALUATE_SCRIPT.read_bytes()).hexdigest()
    evaluate_unmodified_verified = eval_sha256_before == eval_sha256_after

    passed_count = sum(1 for r in all_test_records if r["assertion_passed"])
    failed_records = [r for r in all_test_records if not r["assertion_passed"]]
    all_passed = (
        (len(failed_records) == 0)
        and dep_audit["strictly_self_contained_stdlib_plus_numpy"]
        and temp_dir_cleaned_verified
        and evaluate_unmodified_verified
    )

    total_suite_s = round(time.perf_counter() - t_suite_start, 3)

    final_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_script": str(Path(__file__).resolve()),
        "total_runtime_s": total_suite_s,
        "overall_audit_passed": all_passed,
        "summary_counts": {
            "total_evaluations_run": len(all_test_records),
            "assertions_passed": passed_count,
            "assertions_failed": len(failed_records),
            "failed_case_ids": [f"{r['dataset']}:{r['case_id']}" for r in failed_records],
        },
        "hygiene_and_immutability_verification": {
            "evaluate_py_unmodified": evaluate_unmodified_verified,
            "evaluate_py_sha256": eval_sha256_after,
            "temporary_directory_cleaned_up": temp_dir_cleaned_verified,
            "strict_cr2_filtering_enforced": True,
            "dependency_audit": dep_audit,
        },
        "results": all_test_records,
    }

    RESULTS_OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(final_report, fh, indent=2, allow_nan=True)

    print(
        f"\nCompleted {len(all_test_records)} torture-test evaluations in {total_suite_s:.2f}s: "
        f"{passed_count}/{len(all_test_records)} passed | "
        f"self_contained={dep_audit['strictly_self_contained_stdlib_plus_numpy']} | "
        f"tmp_cleaned={temp_dir_cleaned_verified}"
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

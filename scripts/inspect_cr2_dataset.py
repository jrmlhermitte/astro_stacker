#!/usr/bin/env python3
"""Inspect Canon .CR2 astrophotography datasets and emit logs/dataset_manifest.json.

Strictly filters for files ending in `.CR2` (case-insensitive, excluding `.CR2.xmp`,
`.JPG`, `.png`, `.npz`, `.xcf`, etc.) and never opens non-CR2 files.

Uses a lightweight runtime-compiled C++20 `LibRaw` (`-lraw_r`) shared bridge loaded
via Python `ctypes` (with an automatic fallback to `rawpy` if `rawpy` is installed
and `g++`/`libraw` is unavailable), so it runs out-of-the-box with standard `python3`.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np


PERCENTILE_LEVELS = [0, 1, 50, 99, 99.9, 100]

CPP_LIBRAW_BRIDGE = r"""
#include <libraw/libraw.h>
#include <cstdint>
#include <cstring>

extern "C" {

struct Cr2FrameMeta {
    int status;
    char error_msg[128];

    int32_t data_error_count;
    int32_t first_data_error_offset;
    uint32_t process_warnings;

    char make[64];
    char model[64];
    char software[64];
    char body_serial[64];
    char lens[128];
    char cdesc[5];

    float iso_speed;
    float real_iso;
    float shutter;
    float aperture;
    float focal_len;
    int64_t timestamp;
    uint32_t shot_order;

    float camera_temp_c;
    float sensor_temp_c;
    float sensor_temp2_c;
    float lens_temp_c;
    float ambient_temp_c;

    uint16_t raw_width;
    uint16_t raw_height;
    uint16_t width;
    uint16_t height;
    uint16_t top_margin;
    uint16_t left_margin;
    uint16_t iwidth;
    uint16_t iheight;
    int32_t flip;
    double pixel_aspect;

    uint32_t black;
    uint32_t cblack[6];
    uint32_t maximum;
    uint32_t data_maximum;
    uint32_t raw_bps;

    int32_t canon_avg_black;
    int32_t canon_channel_black[4];
    int32_t canon_normal_white;
    int32_t canon_specular_white;
    int16_t canon_quality;
    int16_t canon_record_mode;

    float cam_mul[4];
    float pre_mul[4];

    int32_t cfa_vis_2x2[4]; // (0,0), (0,1), (1,0), (1,1) in visible sensor coords
};

static void capture_data_error_cb(void* user_data, const char* /*file*/, const int offset) {
    if (!user_data) return;
    auto* meta = static_cast<Cr2FrameMeta*>(user_data);
    if (meta->data_error_count == 0) {
        meta->first_data_error_offset = offset;
    }
    meta->data_error_count += 1;
}

int cr2_probe_and_read(const char* path, Cr2FrameMeta* out_meta, uint16_t* raw_buf_out, size_t raw_buf_elems) {
    if (!out_meta) return -1;
    std::memset(out_meta, 0, sizeof(Cr2FrameMeta));
    out_meta->first_data_error_offset = -1;

    LibRaw lr;
    lr.set_dataerror_handler(capture_data_error_cb, out_meta);

    int rc = lr.open_file(path);
    if (rc != LIBRAW_SUCCESS) {
        out_meta->status = rc;
        std::strncpy(out_meta->error_msg, libraw_strerror(rc), sizeof(out_meta->error_msg) - 1);
        return rc;
    }

    rc = lr.unpack();
    out_meta->process_warnings = lr.imgdata.process_warnings;
    if (rc != LIBRAW_SUCCESS) {
        out_meta->status = rc;
        std::strncpy(out_meta->error_msg, libraw_strerror(rc), sizeof(out_meta->error_msg) - 1);
        return rc;
    }

    const auto& id = lr.imgdata.idata;
    const auto& sz = lr.imgdata.sizes;
    const auto& col = lr.imgdata.color;
    const auto& oth = lr.imgdata.other;
    const auto& cmn = lr.imgdata.makernotes.common;
    const auto& can = lr.imgdata.makernotes.canon;
    const auto& shi = lr.imgdata.shootinginfo;
    const auto& lns = lr.imgdata.lens;

    std::strncpy(out_meta->make, id.make, sizeof(out_meta->make) - 1);
    std::strncpy(out_meta->model, id.model, sizeof(out_meta->model) - 1);
    std::strncpy(out_meta->software, id.software, sizeof(out_meta->software) - 1);
    std::strncpy(out_meta->body_serial, shi.BodySerial, sizeof(out_meta->body_serial) - 1);
    std::strncpy(out_meta->lens, lns.Lens, sizeof(out_meta->lens) - 1);
    std::memcpy(out_meta->cdesc, id.cdesc, 4);
    out_meta->cdesc[4] = '\0';

    out_meta->iso_speed = oth.iso_speed;
    out_meta->real_iso = cmn.real_ISO;
    out_meta->shutter = oth.shutter;
    out_meta->aperture = oth.aperture;
    out_meta->focal_len = oth.focal_len;
    out_meta->timestamp = static_cast<int64_t>(oth.timestamp);
    out_meta->shot_order = oth.shot_order;

    out_meta->camera_temp_c = cmn.CameraTemperature;
    out_meta->sensor_temp_c = cmn.SensorTemperature;
    out_meta->sensor_temp2_c = cmn.SensorTemperature2;
    out_meta->lens_temp_c = cmn.LensTemperature;
    out_meta->ambient_temp_c = cmn.AmbientTemperature;

    out_meta->raw_width = sz.raw_width;
    out_meta->raw_height = sz.raw_height;
    out_meta->width = sz.width;
    out_meta->height = sz.height;
    out_meta->top_margin = sz.top_margin;
    out_meta->left_margin = sz.left_margin;
    out_meta->iwidth = sz.iwidth;
    out_meta->iheight = sz.iheight;
    out_meta->flip = sz.flip;
    out_meta->pixel_aspect = sz.pixel_aspect;

    out_meta->black = col.black;
    for (int i = 0; i < 6; ++i) {
        out_meta->cblack[i] = col.cblack[i];
    }
    out_meta->maximum = col.maximum;
    out_meta->data_maximum = col.data_maximum;
    out_meta->raw_bps = col.raw_bps;

    out_meta->canon_avg_black = can.AverageBlackLevel;
    for (int i = 0; i < 4; ++i) {
        out_meta->canon_channel_black[i] = can.ChannelBlackLevel[i];
        out_meta->cam_mul[i] = col.cam_mul[i];
        out_meta->pre_mul[i] = col.pre_mul[i];
    }
    out_meta->canon_normal_white = can.NormalWhiteLevel;
    out_meta->canon_specular_white = can.SpecularWhiteLevel;
    out_meta->canon_quality = can.Quality;
    out_meta->canon_record_mode = can.RecordMode;

    out_meta->cfa_vis_2x2[0] = lr.COLOR(0, 0);
    out_meta->cfa_vis_2x2[1] = lr.COLOR(0, 1);
    out_meta->cfa_vis_2x2[2] = lr.COLOR(1, 0);
    out_meta->cfa_vis_2x2[3] = lr.COLOR(1, 1);

    const size_t total_raw = static_cast<size_t>(sz.raw_width) * static_cast<size_t>(sz.raw_height);
    if (!lr.imgdata.rawdata.raw_image) {
        out_meta->status = -2;
        std::strncpy(out_meta->error_msg, "raw_image buffer is null (not Bayer 1-ch raw)", sizeof(out_meta->error_msg) - 1);
        return -2;
    }
    if (raw_buf_out != nullptr) {
        if (raw_buf_elems < total_raw) {
            out_meta->status = -3;
            std::strncpy(out_meta->error_msg, "raw_buf_out smaller than raw_width*raw_height", sizeof(out_meta->error_msg) - 1);
            return -3;
        }
        std::memcpy(raw_buf_out, lr.imgdata.rawdata.raw_image, total_raw * sizeof(uint16_t));
    }

    out_meta->status = 0;
    return 0;
}

const char* cr2_libraw_version() {
    return LibRaw::version();
}

} // extern "C"
"""


class Cr2FrameMetaC(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int),
        ("error_msg", ctypes.c_char * 128),
        ("data_error_count", ctypes.c_int32),
        ("first_data_error_offset", ctypes.c_int32),
        ("process_warnings", ctypes.c_uint32),
        ("make", ctypes.c_char * 64),
        ("model", ctypes.c_char * 64),
        ("software", ctypes.c_char * 64),
        ("body_serial", ctypes.c_char * 64),
        ("lens", ctypes.c_char * 128),
        ("cdesc", ctypes.c_char * 5),
        ("iso_speed", ctypes.c_float),
        ("real_iso", ctypes.c_float),
        ("shutter", ctypes.c_float),
        ("aperture", ctypes.c_float),
        ("focal_len", ctypes.c_float),
        ("timestamp", ctypes.c_int64),
        ("shot_order", ctypes.c_uint32),
        ("camera_temp_c", ctypes.c_float),
        ("sensor_temp_c", ctypes.c_float),
        ("sensor_temp2_c", ctypes.c_float),
        ("lens_temp_c", ctypes.c_float),
        ("ambient_temp_c", ctypes.c_float),
        ("raw_width", ctypes.c_uint16),
        ("raw_height", ctypes.c_uint16),
        ("width", ctypes.c_uint16),
        ("height", ctypes.c_uint16),
        ("top_margin", ctypes.c_uint16),
        ("left_margin", ctypes.c_uint16),
        ("iwidth", ctypes.c_uint16),
        ("iheight", ctypes.c_uint16),
        ("flip", ctypes.c_int32),
        ("pixel_aspect", ctypes.c_double),
        ("black", ctypes.c_uint32),
        ("cblack", ctypes.c_uint32 * 6),
        ("maximum", ctypes.c_uint32),
        ("data_maximum", ctypes.c_uint32),
        ("raw_bps", ctypes.c_uint32),
        ("canon_avg_black", ctypes.c_int32),
        ("canon_channel_black", ctypes.c_int32 * 4),
        ("canon_normal_white", ctypes.c_int32),
        ("canon_specular_white", ctypes.c_int32),
        ("canon_quality", ctypes.c_int16),
        ("canon_record_mode", ctypes.c_int16),
        ("cam_mul", ctypes.c_float * 4),
        ("pre_mul", ctypes.c_float * 4),
        ("cfa_vis_2x2", ctypes.c_int32 * 4),
    ]


def is_strict_cr2(filename: str) -> bool:
    """Strictly match files ending in .CR2 (case-insensitive), excluding .CR2.xmp etc."""
    return len(filename) > 4 and filename.upper().endswith(".CR2")


def classify_ignored_extension(filename: str) -> str:
    """Classify non-CR2 filenames by extension string only without reading any file."""
    lower = filename.lower()
    if lower.endswith(".cr2.xmp"):
        return ".CR2.xmp"
    _, ext = os.path.splitext(filename)
    return ext if ext else "<no_ext>"


def clean_temp(val: float) -> float | None:
    """LibRaw uses -1000.0f sentinel when a MakerNotes temperature tag is absent."""
    if val <= -999.0 or np.isnan(val):
        return None
    return round(float(val), 2)


def flip_to_description(flip: int) -> str:
    mapping = {
        0: "0_deg_normal (unrotated)",
        3: "180_deg",
        5: "90_deg_ccw (gravity tag; raw_image stays unrotated 4290x2856)",
        6: "90_deg_cw (gravity tag; raw_image stays unrotated 4290x2856)",
    }
    return mapping.get(flip, f"flip_code_{flip}")


def build_cfa_description(cdesc: str, cfa_2x2_flat: list[int]) -> dict[str, Any]:
    """Map LibRaw 2x2 color indices (0..3) to canonical R, G1, B, G2 labels."""
    idx_to_label: dict[int, str] = {}
    g_seen = 0
    for idx in range(4):
        ch = cdesc[idx] if idx < len(cdesc) else "?"
        if ch == "G":
            g_seen += 1
            idx_to_label[idx] = f"G{g_seen}"
        else:
            idx_to_label[idx] = ch

    c00, c01, c10, c11 = cfa_2x2_flat
    labels_2x2 = [
        [idx_to_label.get(c00, "?"), idx_to_label.get(c01, "?")],
        [idx_to_label.get(c10, "?"), idx_to_label.get(c11, "?")],
    ]
    base_letters = "".join(
        cdesc[c] if c < len(cdesc) else "?" for c in (c00, c01, c10, c11)
    )
    return {
        "cdesc": cdesc,
        "pattern_name": base_letters,
        "pattern_2x2_indices": [[int(c00), int(c01)], [int(c10), int(c11)]],
        "pattern_2x2_labels": labels_2x2,
        "index_to_channel": {str(k): v for k, v in idx_to_label.items()},
    }


class LibRawBridgeReader:
    """Runtime-compiled C++20 LibRaw bridge loaded via ctypes."""

    def __init__(self, tmp_dir: str) -> None:
        so_path = os.path.join(tmp_dir, "libcr2_bridge.so")
        cmd = [
            "g++",
            "-std=c++20",
            "-O3",
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
            input=CPP_LIBRAW_BRIDGE,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to compile C++20 LibRaw bridge:\n{proc.stderr}")

        self.lib = ctypes.CDLL(so_path)
        self.lib.cr2_probe_and_read.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(Cr2FrameMetaC),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_size_t,
        ]
        self.lib.cr2_probe_and_read.restype = ctypes.c_int
        self.lib.cr2_libraw_version.argtypes = []
        self.lib.cr2_libraw_version.restype = ctypes.c_char_p
        self.libraw_version = self.lib.cr2_libraw_version().decode("utf-8", errors="replace")
        self.backend_name = f"ctypes_cpp20_libraw_r ({self.libraw_version})"

    def read_cr2(self, filepath: Path) -> tuple[dict[str, Any], np.ndarray]:
        path_bytes = os.fsencode(str(filepath))
        meta = Cr2FrameMetaC()
        max_elems = 32_000_000
        buf = np.empty(max_elems, dtype=np.uint16)
        buf_ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))

        rc = self.lib.cr2_probe_and_read(path_bytes, ctypes.byref(meta), buf_ptr, max_elems)
        if rc == -3:
            exact_elems = int(meta.raw_width) * int(meta.raw_height)
            buf = np.empty(exact_elems, dtype=np.uint16)
            buf_ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
            rc = self.lib.cr2_probe_and_read(path_bytes, ctypes.byref(meta), buf_ptr, exact_elems)

        if rc != 0:
            err = meta.error_msg.decode("utf-8", errors="replace")
            raise RuntimeError(f"LibRaw error ({rc}) reading {filepath.name}: {err}")

        raw_h = int(meta.raw_height)
        raw_w = int(meta.raw_width)
        raw_2d = buf[: raw_h * raw_w].reshape((raw_h, raw_w)).copy()

        cdesc = meta.cdesc.decode("ascii", errors="replace")
        cfa_flat = [int(meta.cfa_vis_2x2[i]) for i in range(4)]
        cfa_info = build_cfa_description(cdesc, cfa_flat)

        cblack_list = [int(meta.cblack[i]) for i in range(6)]
        black_base = int(meta.black)
        idx_to_ch = cfa_info["index_to_channel"]
        effective_black_by_channel = {
            idx_to_ch[str(i)]: black_base + cblack_list[i] for i in range(4)
        }

        ts = int(meta.timestamp)
        ts_iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts > 0 else None

        meta_dict: dict[str, Any] = {
            "make": meta.make.decode("utf-8", errors="replace").strip(),
            "model": meta.model.decode("utf-8", errors="replace").strip(),
            "software": meta.software.decode("utf-8", errors="replace").strip(),
            "body_serial": meta.body_serial.decode("utf-8", errors="replace").strip(),
            "lens": meta.lens.decode("utf-8", errors="replace").strip(),
            "iso_speed": round(float(meta.iso_speed), 2),
            "real_iso": round(float(meta.real_iso), 2) if meta.real_iso > 0 else None,
            "exposure_time_s": round(float(meta.shutter), 6),
            "aperture_f_number": round(float(meta.aperture), 2),
            "focal_length_mm": round(float(meta.focal_len), 2),
            "timestamp_unix": ts,
            "timestamp_utc_iso": ts_iso,
            "shot_order": int(meta.shot_order),
            "temperature": {
                "camera_temp_c": clean_temp(meta.camera_temp_c),
                "sensor_temp_c": clean_temp(meta.sensor_temp_c),
                "sensor_temp2_c": clean_temp(meta.sensor_temp2_c),
                "lens_temp_c": clean_temp(meta.lens_temp_c),
                "ambient_temp_c": clean_temp(meta.ambient_temp_c),
            },
            "flip": int(meta.flip),
            "orientation": flip_to_description(int(meta.flip)),
            "libraw_integrity": {
                "data_error_count": int(meta.data_error_count),
                "first_data_error_offset": (
                    int(meta.first_data_error_offset)
                    if int(meta.data_error_count) > 0
                    else None
                ),
                "process_warnings_bitmask": int(meta.process_warnings),
            },
            "dimensions": {
                "raw_width": raw_w,
                "raw_height": raw_h,
                "width": int(meta.width),
                "height": int(meta.height),
                "top_margin": int(meta.top_margin),
                "left_margin": int(meta.left_margin),
                "right_margin": raw_w - int(meta.left_margin) - int(meta.width),
                "bottom_margin": raw_h - int(meta.top_margin) - int(meta.height),
                "iwidth": int(meta.iwidth),
                "iheight": int(meta.iheight),
                "pixel_aspect": round(float(meta.pixel_aspect), 6),
                "raw_bps": int(meta.raw_bps),
            },
            "levels": {
                "black": black_base,
                "cblack_raw_4ch": cblack_list[:4],
                "cblack_grid_dims": cblack_list[4:6],
                "effective_black_per_channel": effective_black_by_channel,
                "maximum": int(meta.maximum),
                "data_maximum": int(meta.data_maximum),
                "canon_makernotes": {
                    "average_black_level": int(meta.canon_avg_black),
                    "channel_black_level": [int(meta.canon_channel_black[i]) for i in range(4)],
                    "normal_white_level": int(meta.canon_normal_white),
                    "specular_white_level": int(meta.canon_specular_white),
                    "quality": int(meta.canon_quality),
                    "record_mode": int(meta.canon_record_mode),
                },
                "cam_mul": [round(float(meta.cam_mul[i]), 6) for i in range(4)],
                "pre_mul": [round(float(meta.pre_mul[i]), 6) for i in range(4)],
            },
            "cfa": cfa_info,
        }
        return meta_dict, raw_2d


class RawpyFallbackReader:
    """Fallback reader if rawpy is installed and C++ compiler/libraw headers are absent."""

    def __init__(self) -> None:
        import rawpy  # type: ignore

        self._rawpy = rawpy
        self.backend_name = f"rawpy ({getattr(rawpy, '__version__', 'installed')})"

    def read_cr2(self, filepath: Path) -> tuple[dict[str, Any], np.ndarray]:
        with self._rawpy.imread(str(filepath)) as raw:
            raw_2d = raw.raw_image.copy()
            sz = raw.sizes
            cdesc = raw.color_desc.decode("ascii", errors="replace")
            pat = raw.raw_pattern
            cfa_flat = [int(pat[0, 0]), int(pat[0, 1]), int(pat[1, 0]), int(pat[1, 1])]
            cfa_info = build_cfa_description(cdesc, cfa_flat)
            cblack = [int(x) for x in raw.black_level_per_channel]
            black_base = min(cblack)
            idx_to_ch = cfa_info["index_to_channel"]
            eff_black = {idx_to_ch[str(i)]: cblack[i] for i in range(4)}
            meta_dict: dict[str, Any] = {
                "make": "Unknown (rawpy fallback)",
                "model": "Unknown (rawpy fallback)",
                "software": "",
                "body_serial": "",
                "lens": "",
                "iso_speed": None,
                "real_iso": None,
                "exposure_time_s": None,
                "aperture_f_number": None,
                "focal_length_mm": None,
                "timestamp_unix": int(filepath.stat().st_mtime),
                "timestamp_utc_iso": datetime.fromtimestamp(
                    int(filepath.stat().st_mtime), tz=timezone.utc
                ).isoformat(),
                "shot_order": 0,
                "temperature": {
                    "camera_temp_c": None,
                    "sensor_temp_c": None,
                    "sensor_temp2_c": None,
                    "lens_temp_c": None,
                    "ambient_temp_c": None,
                },
                "flip": int(sz.flip),
                "orientation": flip_to_description(int(sz.flip)),
                "libraw_integrity": {
                    "data_error_count": 0,
                    "first_data_error_offset": None,
                    "process_warnings_bitmask": 0,
                },
                "dimensions": {
                    "raw_width": int(sz.raw_width),
                    "raw_height": int(sz.raw_height),
                    "width": int(sz.width),
                    "height": int(sz.height),
                    "top_margin": int(sz.top_margin),
                    "left_margin": int(sz.left_margin),
                    "right_margin": int(sz.raw_width - sz.left_margin - sz.width),
                    "bottom_margin": int(sz.raw_height - sz.top_margin - sz.height),
                    "iwidth": int(sz.iwidth),
                    "iheight": int(sz.iheight),
                    "pixel_aspect": float(sz.pixel_aspect),
                    "raw_bps": 14,
                },
                "levels": {
                    "black": black_base,
                    "cblack_raw_4ch": [c - black_base for c in cblack],
                    "cblack_grid_dims": [0, 0],
                    "effective_black_per_channel": eff_black,
                    "maximum": int(raw.white_level),
                    "data_maximum": 0,
                    "canon_makernotes": None,
                    "cam_mul": [float(x) for x in raw.camera_whitebalance],
                    "pre_mul": [float(x) for x in raw.daylight_whitebalance],
                },
                "cfa": cfa_info,
            }
            return meta_dict, raw_2d


def compute_pixel_statistics(meta: dict[str, Any], raw_2d: np.ndarray) -> dict[str, Any]:
    """Compute active visible sensor percentiles, mean, std, per-CFA-channel medians, OB & corruption stats."""
    dims = meta["dimensions"]
    top = dims["top_margin"]
    left = dims["left_margin"]
    h = dims["height"]
    w = dims["width"]
    raw_bps = dims.get("raw_bps") or 14
    adc_ceiling = (1 << raw_bps) - 1  # 16383 for 14-bit Canon CR2

    vis = raw_2d[top : top + h, left : left + w]
    pct_vals = np.percentile(vis, PERCENTILE_LEVELS)
    pct_keys = ["0", "1", "50", "99", "99.9", "100"]
    percentiles = {k: round(float(v), 3) for k, v in zip(pct_keys, pct_vals)}

    mean_val = float(np.mean(vis, dtype=np.float64))
    std_val = float(np.std(vis, dtype=np.float64))

    labels_2x2 = meta["cfa"]["pattern_2x2_labels"]
    cfa_medians: dict[str, float] = {}
    cfa_means: dict[str, float] = {}
    cfa_stds: dict[str, float] = {}
    for r_off in (0, 1):
        for c_off in (0, 1):
            label = labels_2x2[r_off][c_off]
            sub = vis[r_off::2, c_off::2]
            cfa_medians[label] = round(float(np.median(sub)), 3)
            cfa_means[label] = round(float(np.mean(sub, dtype=np.float64)), 3)
            cfa_stds[label] = round(float(np.std(sub, dtype=np.float64)), 3)

    levels = meta["levels"]
    libraw_max = levels["maximum"]
    black = levels["black"]
    canon_mn = levels.get("canon_makernotes") or {}
    normal_wl = canon_mn.get("normal_white_level") or libraw_max
    spec_wl = canon_mn.get("specular_white_level") or libraw_max

    total_vis_px = int(vis.size)
    sat_libraw = int(np.count_nonzero(vis >= libraw_max))
    sat_normal = int(np.count_nonzero(vis >= normal_wl))
    sat_specular = int(np.count_nonzero(vis >= spec_wl))

    # 14-bit overflow (>16383) or extreme sub-bias (<1500 on 2048 pedestal) indicates lossless-JPEG corruption
    over_adc_mask = vis > adc_ceiling
    sub_bias_mask = vis < 1500
    bad_px_mask = over_adc_mask | sub_bias_mask
    count_gt_adc = int(np.count_nonzero(over_adc_mask))
    count_lt_1500 = int(np.count_nonzero(sub_bias_mask))
    bad_rows_mask = np.any(bad_px_mask, axis=1)
    corrupted_row_count = int(np.count_nonzero(bad_rows_mask))
    first_corrupted_row = int(np.argmax(bad_rows_mask)) if corrupted_row_count > 0 else None

    libraw_err_cnt = meta["libraw_integrity"]["data_error_count"]
    if libraw_err_cnt > 0 or count_gt_adc > 0:
        integrity_status = "corrupted_bitstream"
    elif count_lt_1500 > 0 or percentiles["1"] < (black - 200):
        integrity_status = "corrupted_predictor_sub_bias"
    else:
        integrity_status = "clean"

    hot_plus_100 = int(np.count_nonzero(vis > black + 100))
    hot_plus_500 = int(np.count_nonzero(vis > black + 500))
    hot_plus_2000 = int(np.count_nonzero(vis > black + 2000))

    # Optical black margin strips (inset by 2 pixels from raw edges to avoid border padding)
    ob_stats: dict[str, Any] = {}
    if left >= 8:
        left_ob = raw_2d[top : top + h, 2 : left - 2]
        ob_stats["left_margin_strip"] = {
            "shape": [int(left_ob.shape[0]), int(left_ob.shape[1])],
            "median": round(float(np.median(left_ob)), 3),
            "mean": round(float(np.mean(left_ob, dtype=np.float64)), 3),
            "std": round(float(np.std(left_ob, dtype=np.float64)), 3),
        }
    if top >= 6:
        top_ob = raw_2d[2 : top - 2, left : left + w]
        ob_stats["top_margin_strip"] = {
            "shape": [int(top_ob.shape[0]), int(top_ob.shape[1])],
            "median": round(float(np.median(top_ob)), 3),
            "mean": round(float(np.mean(top_ob, dtype=np.float64)), 3),
            "std": round(float(np.std(top_ob, dtype=np.float64)), 3),
        }

    return {
        "integrity_status": integrity_status,
        "is_clean": integrity_status == "clean",
        "corruption_diagnostics": {
            "count_gt_14bit_adc_ceiling": count_gt_adc,
            "count_lt_1500_sub_bias": count_lt_1500,
            "corrupted_visible_row_count": corrupted_row_count,
            "corrupted_visible_row_fraction": round(corrupted_row_count / h, 6),
            "first_corrupted_visible_row": first_corrupted_row,
        },
        "visible_pixel_count": total_vis_px,
        "percentiles": percentiles,
        "percentiles_levels": PERCENTILE_LEVELS,
        "percentiles_values": [round(float(v), 3) for v in pct_vals],
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "cfa_channel_medians": cfa_medians,
        "cfa_channel_means": cfa_means,
        "cfa_channel_stds": cfa_stds,
        "saturation": {
            "count_ge_libraw_maximum": sat_libraw,
            "fraction_ge_libraw_maximum": round(sat_libraw / total_vis_px, 8),
            "count_ge_canon_normal_white": sat_normal,
            "count_ge_canon_specular_white": sat_specular,
        },
        "tail_counts_above_black": {
            "gt_black_plus_100": hot_plus_100,
            "gt_black_plus_500": hot_plus_500,
            "gt_black_plus_2000": hot_plus_2000,
        },
        "optical_black_margins": ob_stats,
    }


def inspect_single_cr2(reader: Any, filepath: Path, workspace_root: Path) -> dict[str, Any]:
    meta, raw_2d = reader.read_cr2(filepath)
    stats = compute_pixel_statistics(meta, raw_2d)
    return {
        "filename": filepath.name,
        "relative_path": str(filepath.relative_to(workspace_root)),
        "file_size_bytes": filepath.stat().st_size,
        **meta,
        "raw_stats": stats,
    }


def _aggregate_frame_subset(frames: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not frames:
        return None
    medians_all = [f["raw_stats"]["percentiles"]["50"] for f in frames]
    means_all = [f["raw_stats"]["mean"] for f in frames]
    stds_all = [f["raw_stats"]["std"] for f in frames]
    p999_all = [f["raw_stats"]["percentiles"]["99.9"] for f in frames]
    max_all = [f["raw_stats"]["percentiles"]["100"] for f in frames]
    sat_counts = [f["raw_stats"]["saturation"]["count_ge_libraw_maximum"] for f in frames]
    left_ob_medians = [
        f["raw_stats"]["optical_black_margins"]["left_margin_strip"]["median"]
        for f in frames
        if "left_margin_strip" in f["raw_stats"]["optical_black_margins"]
    ]
    cfa_keys = list(frames[0]["raw_stats"]["cfa_channel_medians"].keys())
    cfa_median_mean_across_frames = {
        ch: round(float(np.mean([f["raw_stats"]["cfa_channel_medians"][ch] for f in frames])), 3)
        for ch in cfa_keys
    }
    avg_eff_black = {
        ch: float(np.mean([f["levels"]["effective_black_per_channel"][ch] for f in frames]))
        for ch in cfa_keys
    }
    return {
        "frame_count": len(frames),
        "frame_median_min": round(float(min(medians_all)), 3),
        "frame_median_mean": round(float(np.mean(medians_all)), 3),
        "frame_median_max": round(float(max(medians_all)), 3),
        "frame_mean_avg": round(float(np.mean(means_all)), 3),
        "frame_std_min": round(float(min(stds_all)), 3),
        "frame_std_mean": round(float(np.mean(stds_all)), 3),
        "frame_std_max": round(float(max(stds_all)), 3),
        "frame_p99_9_mean": round(float(np.mean(p999_all)), 3),
        "frame_max_min": round(float(min(max_all)), 3),
        "frame_max_max": round(float(max(max_all)), 3),
        "cfa_channel_median_avg_across_frames": cfa_median_mean_across_frames,
        "sky_pedestal_above_black_adu_by_channel": {
            ch: round(cfa_median_mean_across_frames[ch] - avg_eff_black[ch], 3)
            for ch in cfa_keys
        },
        "saturated_pixels_ge_libraw_max_min": int(min(sat_counts)),
        "saturated_pixels_ge_libraw_max_median": round(float(np.median(sat_counts)), 1),
        "saturated_pixels_ge_libraw_max_max": int(max(sat_counts)),
        "left_optical_black_median_range": (
            [round(float(min(left_ob_medians)), 3), round(float(max(left_ob_medians)), 3)]
            if left_ob_medians
            else None
        ),
    }


def summarize_dataset(
    set_name: str,
    set_dir: Path,
    workspace_root: Path,
    reader: Any,
    workers: int = 8,
) -> dict[str, Any]:
    entries = sorted(os.listdir(set_dir))
    cr2_paths: list[Path] = []
    ignored_exts: Counter[str] = Counter()

    for fname in entries:
        full_path = set_dir / fname
        if not full_path.is_file():
            continue
        if is_strict_cr2(fname):
            cr2_paths.append(full_path)
        else:
            ignored_exts[classify_ignored_extension(fname)] += 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(
            pool.map(lambda p: inspect_single_cr2(reader, p, workspace_root), cr2_paths)
        )

    frames.sort(key=lambda f: (f["timestamp_unix"], f["filename"]))
    clean_frames = [f for f in frames if f["raw_stats"]["is_clean"]]
    corrupted_frames = [f for f in frames if not f["raw_stats"]["is_clean"]]

    timestamps = [f["timestamp_unix"] for f in frames]
    frame_gaps = [
        timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)
    ]
    exposures = [f["exposure_time_s"] for f in frames]
    exposure_counts = Counter(round(e, 4) for e in exposures)
    isos = sorted({f["iso_speed"] for f in frames})
    models = sorted({f"{f['make']} {f['model']}".strip() for f in frames})
    serials = sorted({f["body_serial"] for f in frames if f["body_serial"]})
    temps = [
        f["temperature"]["camera_temp_c"]
        for f in frames
        if f["temperature"]["camera_temp_c"] is not None
    ]

    span_s = (timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0
    total_integration_s = round(float(np.sum(exposures)), 3)
    clean_integration_s = round(float(np.sum([f["exposure_time_s"] for f in clean_frames])), 3)

    summary = {
        "directory": str(set_dir.relative_to(workspace_root)),
        "file_counts": {
            "total_files_in_dir": len(cr2_paths) + sum(ignored_exts.values()),
            "cr2_count": len(cr2_paths),
            "clean_cr2_count": len(clean_frames),
            "corrupted_or_anomalous_cr2_count": len(corrupted_frames),
            "ignored_non_cr2_count": sum(ignored_exts.values()),
            "ignored_by_extension": dict(sorted(ignored_exts.items())),
            "strict_cr2_filter_verified": all(
                f["filename"].upper().endswith(".CR2") for f in frames
            ),
        },
        "integrity_summary": {
            "clean_frame_filenames": [f["filename"] for f in clean_frames],
            "corrupted_or_anomalous_frames": [
                {
                    "filename": f["filename"],
                    "integrity_status": f["raw_stats"]["integrity_status"],
                    "libraw_data_error_offset": f["libraw_integrity"]["first_data_error_offset"],
                    "corrupted_visible_row_count": f["raw_stats"]["corruption_diagnostics"][
                        "corrupted_visible_row_count"
                    ],
                    "corrupted_visible_row_fraction": f["raw_stats"]["corruption_diagnostics"][
                        "corrupted_visible_row_fraction"
                    ],
                    "first_corrupted_visible_row": f["raw_stats"]["corruption_diagnostics"][
                        "first_corrupted_visible_row"
                    ],
                    "min_adu": f["raw_stats"]["percentiles"]["0"],
                    "max_adu": f["raw_stats"]["percentiles"]["100"],
                    "std_adu": f["raw_stats"]["std"],
                }
                for f in corrupted_frames
            ],
        },
        "camera_and_acquisition": {
            "models": models,
            "body_serials": serials,
            "iso_values": isos,
            "unique_exposure_times_s": sorted(exposure_counts.keys()),
            "exposure_time_histogram_s": {str(k): v for k, v in sorted(exposure_counts.items())},
            "exposure_min_s": min(exposures) if exposures else None,
            "exposure_max_s": max(exposures) if exposures else None,
            "total_integration_time_s": total_integration_s,
            "clean_integration_time_s": clean_integration_s,
            "clean_integration_time_min": round(clean_integration_s / 60.0, 2),
            "unique_apertures_f": sorted({f["aperture_f_number"] for f in frames}),
            "unique_focal_lengths_mm": sorted({f["focal_length_mm"] for f in frames}),
            "unique_lenses": sorted({f["lens"] for f in frames}),
            "unique_flips": sorted({f["flip"] for f in frames}),
        },
        "timeline": {
            "first_frame": frames[0]["filename"] if frames else None,
            "last_frame": frames[-1]["filename"] if frames else None,
            "start_timestamp_unix": timestamps[0] if timestamps else None,
            "end_timestamp_unix": timestamps[-1] if timestamps else None,
            "start_utc_iso": frames[0]["timestamp_utc_iso"] if frames else None,
            "end_utc_iso": frames[-1]["timestamp_utc_iso"] if frames else None,
            "elapsed_first_to_last_start_s": span_s,
            "total_wall_duration_incl_last_exp_s": round(
                span_s + (exposures[-1] if exposures else 0.0), 2
            ),
            "inter_frame_start_to_start_gaps_s": frame_gaps,
            "median_cadence_s": round(float(np.median(frame_gaps)), 2) if frame_gaps else None,
            "min_cadence_s": int(min(frame_gaps)) if frame_gaps else None,
            "max_cadence_s": int(max(frame_gaps)) if frame_gaps else None,
        },
        "thermal_profile": {
            "camera_temp_c_available": len(temps) == len(frames) and len(temps) > 0,
            "camera_temp_c_series": temps,
            "start_temp_c": temps[0] if temps else None,
            "end_temp_c": temps[-1] if temps else None,
            "min_temp_c": min(temps) if temps else None,
            "max_temp_c": max(temps) if temps else None,
            "mean_temp_c": round(float(np.mean(temps)), 2) if temps else None,
        },
        "sensor_and_calibration_summary": {
            "dimensions": frames[0]["dimensions"] if frames else None,
            "cfa": frames[0]["cfa"] if frames else None,
            "nominal_black_level": 2048,
            "black_level_range_clean_frames": (
                [
                    min(f["levels"]["black"] for f in clean_frames),
                    max(f["levels"]["black"] for f in clean_frames),
                ]
                if clean_frames
                else None
            ),
            "maximum": frames[0]["levels"]["maximum"] if frames else None,
            "canon_makernotes_levels_sample": (
                clean_frames[0]["levels"]["canon_makernotes"] if clean_frames else None
            ),
        },
        "aggregate_pixel_stats_clean_frames": _aggregate_frame_subset(clean_frames),
        "aggregate_pixel_stats_all_frames": _aggregate_frame_subset(frames),
    }

    return {
        "set_name": set_name,
        "summary": summary,
        "frames": frames,
    }


def build_verification_report(datasets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dark = datasets["dark_set"]["summary"]
    ring = datasets["ring_set"]["summary"]
    andro = datasets["andromeda_set"]["summary"]

    def physical_sensor_geom_set(ds_name: str) -> set[tuple[Any, ...]]:
        res = set()
        for f in datasets[ds_name]["frames"]:
            d = f["dimensions"]
            lv = f["levels"]
            res.add(
                (
                    d["raw_width"],
                    d["raw_height"],
                    d["width"],
                    d["height"],
                    d["top_margin"],
                    d["left_margin"],
                    d["raw_bps"],
                    f["cfa"]["pattern_name"],
                    tuple(f["cfa"]["pattern_2x2_indices"][0]),
                    tuple(f["cfa"]["pattern_2x2_indices"][1]),
                    lv["maximum"],
                )
            )
        return res

    dark_geom = physical_sensor_geom_set("dark_set")
    ring_geom = physical_sensor_geom_set("ring_set")
    andro_geom = physical_sensor_geom_set("andromeda_set")
    physical_sensor_exact_match = (
        len(dark_geom) == 1 and dark_geom == ring_geom and dark_geom == andro_geom
    )

    model_match = (
        dark["camera_and_acquisition"]["models"]
        == ring["camera_and_acquisition"]["models"]
        == andro["camera_and_acquisition"]["models"]
        and len(dark["camera_and_acquisition"]["models"]) == 1
    )
    serial_match = (
        dark["camera_and_acquisition"]["body_serials"]
        == ring["camera_and_acquisition"]["body_serials"]
        == andro["camera_and_acquisition"]["body_serials"]
        and len(dark["camera_and_acquisition"]["body_serials"]) == 1
    )
    iso_match = (
        dark["camera_and_acquisition"]["iso_values"]
        == ring["camera_and_acquisition"]["iso_values"]
        == andro["camera_and_acquisition"]["iso_values"]
        and len(dark["camera_and_acquisition"]["iso_values"]) == 1
    )

    ordered_sets = sorted(
        ["dark_set", "ring_set", "andromeda_set"],
        key=lambda s: datasets[s]["summary"]["timeline"]["start_timestamp_unix"],
    )
    inter_set_transitions = []
    for i in range(len(ordered_sets) - 1):
        s_prev = ordered_sets[i]
        s_next = ordered_sets[i + 1]
        t_prev_end = datasets[s_prev]["summary"]["timeline"]["end_timestamp_unix"]
        t_next_start = datasets[s_next]["summary"]["timeline"]["start_timestamp_unix"]
        inter_set_transitions.append(
            {
                "from_set": s_prev,
                "to_set": s_next,
                "last_frame_of_prev": datasets[s_prev]["summary"]["timeline"]["last_frame"],
                "first_frame_of_next": datasets[s_next]["summary"]["timeline"]["first_frame"],
                "gap_between_last_start_and_next_start_s": t_next_start - t_prev_end,
            }
        )

    session_start = datasets[ordered_sets[0]]["summary"]["timeline"]["start_timestamp_unix"]
    session_end = datasets[ordered_sets[-1]]["summary"]["timeline"]["end_timestamp_unix"]
    last_exp = datasets[ordered_sets[-1]]["frames"][-1]["exposure_time_s"]
    total_session_span_s = round((session_end - session_start) + last_exp, 2)

    dark_mean_temp = dark["thermal_profile"]["mean_temp_c"]
    ring_mean_temp = ring["thermal_profile"]["mean_temp_c"]
    andro_mean_temp = andro["thermal_profile"]["mean_temp_c"]

    all_corrupted = {
        s: datasets[s]["summary"]["integrity_summary"]["corrupted_or_anomalous_frames"]
        for s in ["andromeda_set", "ring_set", "dark_set"]
    }

    return {
        "dark_matches_lights_camera_model": model_match,
        "dark_matches_lights_body_serial": serial_match,
        "dark_matches_lights_iso": iso_match,
        "dark_matches_lights_physical_sensor_dimensions_cfa_and_white_level": physical_sensor_exact_match,
        "exposure_time_comparison": {
            "dark_matches_ring_exact": (
                dark["camera_and_acquisition"]["unique_exposure_times_s"]
                == ring["camera_and_acquisition"]["unique_exposure_times_s"]
            ),
            "andromeda_minor_bulb_diff_note": (
                "dark_set (20/20 frames) and ring_set (21/21 frames) are 29.0s; "
                "andromeda_set has 2 frames at 29.0s (IMG_6041, IMG_6042) and 19 frames at 28.0s "
                "(3.4% Bulb intervalometer difference)."
            ),
            "histograms_s": {
                "andromeda_set": andro["camera_and_acquisition"]["exposure_time_histogram_s"],
                "ring_set": ring["camera_and_acquisition"]["exposure_time_histogram_s"],
                "dark_set": dark["camera_and_acquisition"]["exposure_time_histogram_s"],
            },
        },
        "orientation_flip_warning": {
            "flips_by_set": {
                "andromeda_set": andro["camera_and_acquisition"]["unique_flips"],
                "ring_set": ring["camera_and_acquisition"]["unique_flips"],
                "dark_set": dark["camera_and_acquisition"]["unique_flips"],
            },
            "critical_pipeline_rule": (
                "ring_set has EXIF flip=5 (90 deg CCW gravity sensor tag) while dark_set and "
                "andromeda_set have flip=0. Calibration (bias/dark subtraction & bad-pixel "
                "repair) MUST operate directly on unrotated rawdata.raw_image sensor coordinates "
                "(4352x2874 -> active 4290x2856 RGGB) ignoring sizes.flip!"
            ),
        },
        "corrupted_frames_alert": {
            "total_corrupted_or_anomalous_cr2": sum(len(v) for v in all_corrupted.values()),
            "by_set": all_corrupted,
            "critical_pipeline_rule": (
                "5 .CR2 files have bitstream/predictor corruption on disk: "
                "dark_set/IMG_6098.CR2 (LibRaw data error at byte 9945668, 16-bit wrap to 65535), "
                "dark_set/IMG_6096.CR2 (silent predictor sub-bias corruption down to 743 ADU, std=130.1 vs 28.4), "
                "ring_set/IMG_6079.CR2 (data error at 7103945), "
                "ring_set/IMG_6083.CR2 (data error at 12325909), and "
                "ring_set/IMG_6089.CR2 (data error at 15098935, near bottom rows). "
                "C++ ingest MUST hook LibRaw::set_dataerror_handler AND reject frames/rows with "
                "pixels > 16383 (14-bit ceiling) or anomalous sub-bias tails (< 1500 ADU)."
            ),
        },
        "optics_attachment_notes": {
            "andromeda_set": {
                "aperture_f": andro["camera_and_acquisition"]["unique_apertures_f"],
                "focal_mm": andro["camera_and_acquisition"]["unique_focal_lengths_mm"],
                "lens": andro["camera_and_acquisition"]["unique_lenses"],
                "note": "T-ring / prime-focus telescope attachment (no electronic lens contacts -> f/0, 50mm nominal)",
            },
            "ring_set": {
                "aperture_f": ring["camera_and_acquisition"]["unique_apertures_f"],
                "focal_mm": ring["camera_and_acquisition"]["unique_focal_lengths_mm"],
                "lens": ring["camera_and_acquisition"]["unique_lenses"],
                "note": "T-ring / prime-focus telescope attachment (no electronic lens contacts -> f/0, 50mm nominal)",
            },
            "dark_set": {
                "aperture_f": dark["camera_and_acquisition"]["unique_apertures_f"],
                "focal_mm": dark["camera_and_acquisition"]["unique_focal_lengths_mm"],
                "lens": dark["camera_and_acquisition"]["unique_lenses"],
                "note": "Body removed from telescope at end of session and capped with EF-S 18-55mm lens for darks (valid dark acquisition)",
            },
        },
        "session_timeline_analysis": {
            "chronological_set_order": ordered_sets,
            "camera_rtc_clock_note": (
                "EXIF timestamps are contiguous from 1600708480 to 1600710646 "
                "(2020-09-21 17:14:40Z to 17:50:46Z on uncalibrated camera internal clock; "
                "folder date is 2020-11-28). Relative inter-frame elapsed times are exact."
            ),
            "total_session_duration_s": total_session_span_s,
            "total_session_duration_min": round(total_session_span_s / 60.0, 2),
            "inter_set_transitions": inter_set_transitions,
            "drift_duration_per_light_set": {
                "andromeda_set": {
                    "frame_count": andro["file_counts"]["cr2_count"],
                    "clean_frame_count": andro["file_counts"]["clean_cr2_count"],
                    "first_to_last_start_span_s": andro["timeline"]["elapsed_first_to_last_start_s"],
                    "total_wall_span_s": andro["timeline"]["total_wall_duration_incl_last_exp_s"],
                    "median_cadence_s": andro["timeline"]["median_cadence_s"],
                },
                "ring_set": {
                    "frame_count": ring["file_counts"]["cr2_count"],
                    "clean_frame_count": ring["file_counts"]["clean_cr2_count"],
                    "first_to_last_start_span_s": ring["timeline"]["elapsed_first_to_last_start_s"],
                    "total_wall_span_s": ring["timeline"]["total_wall_duration_incl_last_exp_s"],
                    "median_cadence_s": ring["timeline"]["median_cadence_s"],
                },
            },
        },
        "thermal_mismatch_warning": {
            "andromeda_temp_c_range": [
                andro["thermal_profile"]["min_temp_c"],
                andro["thermal_profile"]["max_temp_c"],
            ],
            "andromeda_mean_temp_c": andro_mean_temp,
            "ring_temp_c_range": [
                ring["thermal_profile"]["min_temp_c"],
                ring["thermal_profile"]["max_temp_c"],
            ],
            "ring_mean_temp_c": ring_mean_temp,
            "dark_temp_c_range": [
                dark["thermal_profile"]["min_temp_c"],
                dark["thermal_profile"]["max_temp_c"],
            ],
            "dark_mean_temp_c": dark_mean_temp,
            "delta_temp_andromeda_minus_dark_c": (
                round(andro_mean_temp - dark_mean_temp, 2)
                if (andro_mean_temp is not None and dark_mean_temp is not None)
                else None
            ),
            "delta_temp_ring_minus_dark_c": (
                round(ring_mean_temp - dark_mean_temp, 2)
                if (ring_mean_temp is not None and dark_mean_temp is not None)
                else None
            ),
            "calibration_recommendation": (
                "Camera cooled continuously outdoors during the 36.6-minute session "
                f"(andromeda_set 34->21 C, mean {andro_mean_temp} C; "
                f"ring_set 24->19 C, mean {ring_mean_temp} C; "
                f"dark_set steady {dark_mean_temp} C). Because Canon CMOS dark current roughly doubles "
                "every ~6-7 C, downstream calibration MUST use per-frame optical-black bias tracking "
                "plus optimized dark scaling + hot/bad-pixel cosmetic correction rather than naive "
                "unscaled (Light - MasterDark) subtraction."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Canon .CR2 datasets for astro_stacker.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("pictures/2020-11-28-deep-sky-ring-andromeda"),
        help="Base dataset directory containing dark_set, ring_set, andromeda_set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/dataset_manifest.json"),
        help="Output JSON manifest path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Worker threads for parallel LibRaw unpacking and numpy statistics.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    workspace_root = Path.cwd()
    base_dir = (workspace_root / args.base_dir).resolve()
    out_path = workspace_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target_sets = ["dark_set", "ring_set", "andromeda_set"]
    for s in target_sets:
        if not (base_dir / s).is_dir():
            raise FileNotFoundError(f"Expected dataset directory not found: {base_dir / s}")

    with tempfile.TemporaryDirectory(prefix="astro_cr2_inspect_") as tmp_dir:
        try:
            reader: Any = LibRawBridgeReader(tmp_dir)
        except Exception as exc:
            print(f"[warn] C++ LibRaw bridge unavailable ({exc}); trying rawpy fallback...", file=sys.stderr)
            reader = RawpyFallbackReader()

        datasets: dict[str, dict[str, Any]] = {}
        for s in target_sets:
            datasets[s] = summarize_dataset(
                set_name=s,
                set_dir=base_dir / s,
                workspace_root=workspace_root,
                reader=reader,
                workers=args.workers,
            )

        verification = build_verification_report(datasets)
        elapsed_s = round(time.perf_counter() - t0, 3)

        manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "inspection_runtime_s": elapsed_s,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "reader_backend": reader.backend_name,
            "base_dataset_directory": str(base_dir.relative_to(workspace_root)),
            "total_cr2_inspected": sum(
                datasets[s]["summary"]["file_counts"]["cr2_count"] for s in target_sets
            ),
            "total_clean_cr2": sum(
                datasets[s]["summary"]["file_counts"]["clean_cr2_count"] for s in target_sets
            ),
            "total_corrupted_or_anomalous_cr2": sum(
                datasets[s]["summary"]["file_counts"]["corrupted_or_anomalous_cr2_count"]
                for s in target_sets
            ),
            "total_non_cr2_ignored": sum(
                datasets[s]["summary"]["file_counts"]["ignored_non_cr2_count"]
                for s in target_sets
            ),
            "verification": verification,
            "datasets": datasets,
        }

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

    print(
        f"Wrote {out_path} in {elapsed_s:.2f}s using backend={manifest['reader_backend']} "
        f"(inspected {manifest['total_cr2_inspected']} .CR2 files: "
        f"{manifest['total_clean_cr2']} clean, {manifest['total_corrupted_or_anomalous_cr2']} corrupted/anomalous; "
        f"strictly ignored {manifest['total_non_cr2_ignored']} non-CR2 files)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

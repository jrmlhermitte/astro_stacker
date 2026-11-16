#include "raw_loader.hpp"

#include <libraw/libraw.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

namespace astro {

namespace {

void libraw_data_error_cb(void* user_data, const char* /*file*/, const int offset) {
    if (!user_data) return;
    auto* meta = static_cast<RawFrameMeta*>(user_data);
    if (meta->data_error_count == 0) {
        meta->first_data_error_offset = offset;
    }
    meta->data_error_count += 1;
}

double exact_u16_quantile_from_hist(const uint32_t* hist, size_t n_total, double q) {
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

bool has_strict_cr2_extension(const std::string& filename) {
    if (filename.size() <= 4) return false;
    const std::string tail = filename.substr(filename.size() - 4);
    return tail[0] == '.' &&
           (tail[1] == 'C' || tail[1] == 'c') &&
           (tail[2] == 'R' || tail[2] == 'r') &&
           tail[3] == '2';
}

} // namespace

std::vector<std::string> list_strict_cr2_files(const std::string& dir_path) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (!entry.is_regular_file()) continue;
        const std::string fname = entry.path().filename().string();
        if (has_strict_cr2_extension(fname)) {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

bool load_and_validate_cr2(const std::string& filepath, RawSensorFrame& out_frame) {
    RawFrameMeta& meta = out_frame.meta;
    meta = RawFrameMeta{};
    meta.filepath = filepath;
    meta.filename = fs::path(filepath).filename().string();

    LibRaw lr;
    lr.set_dataerror_handler(libraw_data_error_cb, &meta);

    int rc = lr.open_file(filepath.c_str());
    if (rc != LIBRAW_SUCCESS) {
        meta.rc = rc;
        meta.reject_reason = std::string("open_file: ") + libraw_strerror(rc);
        return false;
    }

    rc = lr.unpack();
    if (rc != LIBRAW_SUCCESS) {
        meta.rc = rc;
        meta.reject_reason = std::string("unpack: ") + libraw_strerror(rc);
        return false;
    }

    const uint16_t* raw = lr.imgdata.rawdata.raw_image;
    if (!raw) {
        meta.rc = -2;
        meta.reject_reason = "raw_image is null";
        return false;
    }

    const auto& sz = lr.imgdata.sizes;
    const auto& oth = lr.imgdata.other;
    const auto& cmn = lr.imgdata.makernotes.common;
    const auto& id = lr.imgdata.idata;

    meta.flip_ignored = sz.flip;
    meta.iso_speed = oth.iso_speed;
    meta.shutter_s = oth.shutter;
    meta.camera_temp_c = cmn.CameraTemperature;
    meta.timestamp = static_cast<int64_t>(oth.timestamp);

    if (sz.raw_width != RAW_W || sz.raw_height != RAW_H ||
        sz.top_margin != TOP_MARGIN || sz.left_margin != LEFT_MARGIN ||
        sz.width != ACTIVE_W || sz.height != ACTIVE_H) {
        meta.rc = -4;
        meta.reject_reason = "unexpected_sensor_geometry";
        return false;
    }

    const char c00 = id.cdesc[lr.FC(0, 0)];
    const char c01 = id.cdesc[lr.FC(0, 1)];
    const char c10 = id.cdesc[lr.FC(1, 0)];
    const char c11 = id.cdesc[lr.FC(1, 1)];
    meta.cfa_is_rggb = (c00 == 'R' && c01 == 'G' && c10 == 'G' && c11 == 'B');

    std::vector<uint32_t> hist_act(65536, 0);
    std::vector<uint32_t> hist_ob(65536, 0);

    out_frame.active_cfa.resize(ACTIVE_PIXELS);
    out_frame.ob_row_medians.resize(ACTIVE_H, 2048.0f);

    uint16_t vmin = 65535;
    uint16_t vmax = 0;
    const int ob_cols = OB_LEFT_END - OB_LEFT_START;
    std::vector<uint16_t> row_ob_buf(ob_cols);

    for (int r = 0; r < ACTIVE_H; ++r) {
        const uint16_t* raw_row = raw + static_cast<size_t>(r + TOP_MARGIN) * RAW_W;
        for (int c = OB_LEFT_START; c < OB_LEFT_END; ++c) {
            const uint16_t v = raw_row[c];
            hist_ob[v] += 1;
            row_ob_buf[c - OB_LEFT_START] = v;
        }
        std::nth_element(row_ob_buf.begin(), row_ob_buf.begin() + ob_cols / 2, row_ob_buf.end());
        const uint16_t hi_m = row_ob_buf[ob_cols / 2];
        std::nth_element(row_ob_buf.begin(), row_ob_buf.begin() + (ob_cols / 2 - 1), row_ob_buf.begin() + ob_cols / 2);
        const uint16_t lo_m = row_ob_buf[ob_cols / 2 - 1];
        out_frame.ob_row_medians[r] = 0.5f * static_cast<float>(lo_m + hi_m);

        uint16_t* dst_row = out_frame.active_cfa.data() + static_cast<size_t>(r) * ACTIVE_W;
        const uint16_t* act_row = raw_row + LEFT_MARGIN;
        for (int c = 0; c < ACTIVE_W; ++c) {
            const uint16_t v = act_row[c];
            dst_row[c] = v;
            hist_act[v] += 1;
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
    }

    meta.active_min = vmin;
    meta.active_max = vmax;
    meta.active_p1 = exact_u16_quantile_from_hist(hist_act.data(), ACTIVE_PIXELS, 0.01);
    meta.ob_left_median = exact_u16_quantile_from_hist(
        hist_ob.data(), static_cast<size_t>(ACTIVE_H) * ob_cols, 0.50
    );

    std::ostringstream oss;
    if (meta.data_error_count > 0) {
        oss << "libraw_data_error_count=" << meta.data_error_count
            << "(offset=" << meta.first_data_error_offset << ");";
    }
    if (meta.active_max > 16383) {
        oss << "active_max=" << meta.active_max << ">16383;";
    }
    if (meta.active_p1 < 1850.0) {
        oss << "active_p1=" << meta.active_p1 << "<1850;";
    }
    if (!meta.cfa_is_rggb) {
        oss << "non_rggb_cfa;";
    }

    meta.reject_reason = oss.str();
    meta.is_clean = meta.reject_reason.empty();
    if (!meta.is_clean) {
        out_frame.active_cfa.clear();
        out_frame.active_cfa.shrink_to_fit();
    }
    return meta.is_clean;
}

void load_cr2_directory(
    const std::string& dir_path,
    std::vector<RawSensorFrame>& clean_frames_out,
    std::vector<RawFrameMeta>& all_metas_out
) {
    const std::vector<std::string> paths = list_strict_cr2_files(dir_path);
    const int n = static_cast<int>(paths.size());
    std::vector<RawSensorFrame> loaded(n);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < n; ++i) {
        load_and_validate_cr2(paths[i], loaded[i]);
    }

    clean_frames_out.clear();
    all_metas_out.clear();
    all_metas_out.reserve(n);
    for (int i = 0; i < n; ++i) {
        all_metas_out.push_back(loaded[i].meta);
        if (loaded[i].meta.is_clean) {
            clean_frames_out.push_back(std::move(loaded[i]));
        }
    }
}

} // namespace astro

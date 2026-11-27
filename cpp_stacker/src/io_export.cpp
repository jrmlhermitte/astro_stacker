#include "io_export.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace astro {

namespace {

void ensure_parent_dir(const std::string& filepath) {
    const fs::path p(filepath);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }
}

std::string make_fits_card(const std::string& key, const std::string& value_expr, const std::string& comment = "") {
    std::ostringstream oss;
    oss << std::left << std::setw(8) << key.substr(0, 8) << "= "
        << std::right << std::setw(20) << value_expr;
    if (!comment.empty()) {
        oss << " / " << comment;
    }
    std::string card = oss.str();
    if (card.size() > 80) card.resize(80);
    if (card.size() < 80) card.append(80 - card.size(), ' ');
    return card;
}

struct TiffIfdEntry {
    uint16_t tag;
    uint16_t type;
    uint32_t count;
    uint32_t value_or_offset;
};

} // namespace

void write_npy_f32_rgb(
    const std::string& path,
    const std::vector<float>& rgb,
    int height,
    int width
) {
    const size_t expected = static_cast<size_t>(height) * width * 3;
    if (rgb.size() != expected) {
        throw std::runtime_error("RGB buffer size mismatch in write_npy_f32_rgb");
    }
    ensure_parent_dir(path);

    std::ostringstream dict_ss;
    dict_ss << "{'descr': '<f4', 'fortran_order': False, 'shape': ("
            << height << ", " << width << ", 3), }";
    std::string dict_str = dict_ss.str();

    const size_t prefix_len = 10;
    size_t total_hdr = prefix_len + dict_str.size() + 1;
    const size_t pad = (64 - (total_hdr % 64)) % 64;
    dict_str.append(pad, ' ');
    dict_str.push_back('\n');

    const uint16_t header_len = static_cast<uint16_t>(dict_str.size());
    const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00};

    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error("Failed to open output .npy file: " + path);
    }
    ofs.write(reinterpret_cast<const char*>(magic), 8);
    const unsigned char hlen_le[2] = {
        static_cast<unsigned char>(header_len & 0xFF),
        static_cast<unsigned char>((header_len >> 8) & 0xFF)
    };
    ofs.write(reinterpret_cast<const char*>(hlen_le), 2);
    ofs.write(dict_str.data(), static_cast<std::streamsize>(dict_str.size()));
    ofs.write(
        reinterpret_cast<const char*>(rgb.data()),
        static_cast<std::streamsize>(rgb.size() * sizeof(float))
    );
    if (!ofs) {
        throw std::runtime_error("Failed while writing .npy payload: " + path);
    }
}

std::vector<float> read_npy_f32_rgb(
    const std::string& path,
    int height,
    int width
) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open input .npy file: " + path);
    }
    unsigned char magic[8] = {};
    ifs.read(reinterpret_cast<char*>(magic), 8);
    if (!ifs || magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' ||
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
        throw std::runtime_error("Invalid .npy magic header in: " + path);
    }
    const int major = magic[6];
    uint32_t header_len = 0;
    if (major == 1) {
        unsigned char hl[2] = {};
        ifs.read(reinterpret_cast<char*>(hl), 2);
        header_len = static_cast<uint32_t>(hl[0]) | (static_cast<uint32_t>(hl[1]) << 8);
    } else if (major == 2) {
        unsigned char hl[4] = {};
        ifs.read(reinterpret_cast<char*>(hl), 4);
        header_len = static_cast<uint32_t>(hl[0]) | (static_cast<uint32_t>(hl[1]) << 8) |
                     (static_cast<uint32_t>(hl[2]) << 16) | (static_cast<uint32_t>(hl[3]) << 24);
    } else {
        throw std::runtime_error("Unsupported .npy version in: " + path);
    }
    std::string header(header_len, '\0');
    ifs.read(header.data(), static_cast<std::streamsize>(header_len));
    if (!ifs) {
        throw std::runtime_error("Truncated .npy header in: " + path);
    }
    if (header.find("<f4") == std::string::npos && header.find("float32") == std::string::npos) {
        throw std::runtime_error("Input .npy is not little-endian float32: " + path);
    }

    const size_t expected = static_cast<size_t>(height) * width * 3;
    std::vector<float> out(expected, 1.0f);
    ifs.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(expected * sizeof(float)));
    if (!ifs) {
        throw std::runtime_error("Truncated .npy float32 payload in: " + path);
    }
    return out;
}

void write_transforms_json(
    const std::string& path,
    const std::string& ref_filename,
    const std::vector<FrameTransform>& transforms
) {
    ensure_parent_dir(path);
    std::ofstream ofs(path, std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error("Failed to open transforms.json for writing: " + path);
    }

    int integrated_count = 0;
    for (const auto& t : transforms) {
        if (t.integrated && t.weight > 0.0) ++integrated_count;
    }

    ofs << std::fixed << std::setprecision(6);
    ofs << "{\n";
    ofs << "  \"reference_filename\": \"" << ref_filename << "\",\n";
    ofs << "  \"integrated_frame_count\": " << integrated_count << ",\n";
    ofs << "  \"frames\": [\n";
    for (size_t i = 0; i < transforms.size(); ++i) {
        const auto& t = transforms[i];
        ofs << "    {\n";
        ofs << "      \"filename\": \"" << t.filename << "\",\n";
        ofs << "      \"integrated\": " << (t.integrated ? "true" : "false") << ",\n";
        ofs << "      \"is_reference\": " << (t.is_reference ? "true" : "false") << ",\n";
        const double area_frac = static_cast<double>(std::clamp(t.valid_rows, 0, ACTIVE_H)) / ACTIVE_H;
        ofs << "      \"weight\": " << (t.weight * area_frac) << ",\n";
        ofs << "      \"dx\": " << t.dx << ",\n";
        ofs << "      \"dy\": " << t.dy << ",\n";
        ofs << "      \"dtheta_deg\": " << t.dtheta_deg << ",\n";
        ofs << "      \"scale\": " << t.scale << ",\n";
        ofs << "      \"inlier_matches\": " << t.inlier_matches << ",\n";
        ofs << "      \"residual_rms_px\": " << t.residual_rms_px << "\n";
        ofs << "    }" << (i + 1 < transforms.size() ? "," : "") << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";
}

void write_fits_f32_rgb(
    const std::string& path,
    const std::vector<float>& rgb,
    int height,
    int width
) {
    const size_t n_pix = static_cast<size_t>(height) * width;
    if (rgb.size() != n_pix * 3) {
        throw std::runtime_error("RGB size mismatch in write_fits_f32_rgb");
    }
    ensure_parent_dir(path);

    std::string hdr;
    hdr += make_fits_card("SIMPLE", "T", "Standard FITS format");
    hdr += make_fits_card("BITPIX", "-32", "IEEE 754 32-bit floating point");
    hdr += make_fits_card("NAXIS", "3", "3D RGB image cube");
    hdr += make_fits_card("NAXIS1", std::to_string(width), "Active sensor width");
    hdr += make_fits_card("NAXIS2", std::to_string(height), "Active sensor height");
    hdr += make_fits_card("NAXIS3", "3", "RGB channels");
    hdr += make_fits_card("EXTEND", "T", "Extensions permitted");
    std::string end_card = "END";
    end_card.append(80 - end_card.size(), ' ');
    hdr += end_card;
    while (hdr.size() % 2880 != 0) {
        hdr.append(80, ' ');
    }

    std::vector<uint32_t> be_planes(n_pix * 3);
    for (int ch = 0; ch < 3; ++ch) {
        uint32_t* dst = be_planes.data() + static_cast<size_t>(ch) * n_pix;
        for (size_t i = 0; i < n_pix; ++i) {
            uint32_t bits = 0;
            std::memcpy(&bits, &rgb[3 * i + ch], sizeof(float));
            dst[i] = __builtin_bswap32(bits);
        }
    }

    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error("Failed to open FITS output: " + path);
    }
    ofs.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
    const size_t data_bytes = be_planes.size() * sizeof(uint32_t);
    ofs.write(reinterpret_cast<const char*>(be_planes.data()), static_cast<std::streamsize>(data_bytes));

    const size_t rem = data_bytes % 2880;
    if (rem != 0) {
        std::vector<char> pad(2880 - rem, 0);
        ofs.write(pad.data(), static_cast<std::streamsize>(pad.size()));
    }
}

void write_tiff_f32_rgb(
    const std::string& path,
    const std::vector<float>& rgb,
    int height,
    int width
) {
    const size_t expected = static_cast<size_t>(height) * width * 3;
    if (rgb.size() != expected) {
        throw std::runtime_error("RGB size mismatch in write_tiff_f32_rgb");
    }
    ensure_parent_dir(path);

    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error("Failed to open 32-bit float TIFF output: " + path);
    }

    const uint32_t data_offset = 174;
    const uint32_t byte_count = static_cast<uint32_t>(expected * sizeof(float));
    const unsigned char tiff_hdr[8] = {'I', 'I', 42, 0, 8, 0, 0, 0};
    ofs.write(reinterpret_cast<const char*>(tiff_hdr), 8);

    const uint16_t num_entries = 11;
    ofs.write(reinterpret_cast<const char*>(&num_entries), 2);

    const TiffIfdEntry entries[11] = {
        {256, 4, 1, static_cast<uint32_t>(width)},
        {257, 4, 1, static_cast<uint32_t>(height)},
        {258, 3, 3, 146},
        {259, 3, 1, 1},
        {262, 3, 1, 2},
        {273, 4, 1, data_offset},
        {277, 3, 1, 3},
        {278, 4, 1, static_cast<uint32_t>(height)},
        {279, 4, 1, byte_count},
        {282, 5, 1, 158},
        {339, 3, 3, 152}
    };
    ofs.write(reinterpret_cast<const char*>(entries), sizeof(entries));

    const uint32_t next_ifd = 0;
    ofs.write(reinterpret_cast<const char*>(&next_ifd), 4);

    const uint16_t bps[3] = {32, 32, 32};
    const uint16_t sfmt[3] = {3, 3, 3};
    const uint32_t res[4] = {72, 1, 72, 1};
    ofs.write(reinterpret_cast<const char*>(bps), sizeof(bps));
    ofs.write(reinterpret_cast<const char*>(sfmt), sizeof(sfmt));
    ofs.write(reinterpret_cast<const char*>(res), sizeof(res));

    ofs.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(byte_count));
}

void write_tiff_u16_preview_rgb(
    const std::string& path,
    const std::vector<float>& rgb,
    int height,
    int width
) {
    const size_t n_pix = static_cast<size_t>(height) * width;
    if (rgb.size() != n_pix * 3) {
        throw std::runtime_error("RGB size mismatch in write_tiff_u16_preview_rgb");
    }
    ensure_parent_dir(path);

    double ch_med[3] = {0.0, 0.0, 0.0};
    double ch_mad[3] = {1.0, 1.0, 1.0};
    for (int c = 0; c < 3; ++c) {
        std::vector<double> samp;
        samp.reserve((height / 8) * (width / 8));
        for (int r = ROI_MARGIN; r < height - ROI_MARGIN; r += 8) {
            for (int x = ROI_MARGIN; x < width - ROI_MARGIN; x += 8) {
                samp.push_back(static_cast<double>(rgb[(static_cast<size_t>(r) * width + x) * 3 + c]));
            }
        }
        std::nth_element(samp.begin(), samp.begin() + samp.size() / 2, samp.end());
        ch_med[c] = samp[samp.size() / 2];
        std::vector<double> dev(samp.size());
        for (size_t i = 0; i < samp.size(); ++i) dev[i] = std::abs(samp[i] - ch_med[c]);
        std::nth_element(dev.begin(), dev.begin() + dev.size() / 2, dev.end());
        ch_mad[c] = std::max(1e-6, dev[dev.size() / 2] * 1.4826);
    }

    const double sigma_avg = (ch_mad[0] + ch_mad[1] + ch_mad[2]) / 3.0;
    std::vector<double> lum_norm;
    lum_norm.reserve((height / 4) * (width / 4));
    for (int r = ROI_MARGIN; r < height - ROI_MARGIN; r += 4) {
        for (int x = ROI_MARGIN; x < width - ROI_MARGIN; x += 4) {
            const size_t idx = (static_cast<size_t>(r) * width + x) * 3;
            const double v = (
                (rgb[idx + 0] - ch_med[0]) +
                (rgb[idx + 1] - ch_med[1]) +
                (rgb[idx + 2] - ch_med[2])
            ) / 3.0;
            lum_norm.push_back(v);
        }
    }
    const size_t p9995_idx = static_cast<size_t>(0.9995 * static_cast<double>(lum_norm.size() - 1));
    std::nth_element(lum_norm.begin(), lum_norm.begin() + p9995_idx, lum_norm.end());
    const double white_pt = std::max(25.0 * sigma_avg, lum_norm[p9995_idx]);
    const double black_pt = -2.2 * sigma_avg;
    const double span = std::max(1e-6, white_pt - black_pt);
    const double stretch_beta = 35.0;
    const double inv_asinh = 1.0 / std::asinh(stretch_beta);

    std::vector<uint16_t> u16_rgb(n_pix * 3);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_pix; ++i) {
        for (int c = 0; c < 3; ++c) {
            const double x = std::clamp((static_cast<double>(rgb[3 * i + c]) - ch_med[c] - black_pt) / span, 0.0, 1.0);
            const double y = std::asinh(stretch_beta * x) * inv_asinh;
            u16_rgb[3 * i + c] = static_cast<uint16_t>(std::lround(std::clamp(y, 0.0, 1.0) * 65535.0));
        }
    }

    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error("Failed to open 16-bit preview TIFF output: " + path);
    }

    const uint32_t data_offset = 174;
    const uint32_t byte_count = static_cast<uint32_t>(u16_rgb.size() * sizeof(uint16_t));
    const unsigned char tiff_hdr[8] = {'I', 'I', 42, 0, 8, 0, 0, 0};
    ofs.write(reinterpret_cast<const char*>(tiff_hdr), 8);

    const uint16_t num_entries = 11;
    ofs.write(reinterpret_cast<const char*>(&num_entries), 2);

    const TiffIfdEntry entries[11] = {
        {256, 4, 1, static_cast<uint32_t>(width)},
        {257, 4, 1, static_cast<uint32_t>(height)},
        {258, 3, 3, 146},
        {259, 3, 1, 1},
        {262, 3, 1, 2},
        {273, 4, 1, data_offset},
        {277, 3, 1, 3},
        {278, 4, 1, static_cast<uint32_t>(height)},
        {279, 4, 1, byte_count},
        {282, 5, 1, 158},
        {339, 3, 3, 152}
    };
    ofs.write(reinterpret_cast<const char*>(entries), sizeof(entries));

    const uint32_t next_ifd = 0;
    ofs.write(reinterpret_cast<const char*>(&next_ifd), 4);

    const uint16_t bps[3] = {16, 16, 16};
    const uint16_t sfmt[3] = {1, 1, 1};
    const uint32_t res[4] = {72, 1, 72, 1};
    ofs.write(reinterpret_cast<const char*>(bps), sizeof(bps));
    ofs.write(reinterpret_cast<const char*>(sfmt), sizeof(sfmt));
    ofs.write(reinterpret_cast<const char*>(res), sizeof(res));

    ofs.write(reinterpret_cast<const char*>(u16_rgb.data()), static_cast<std::streamsize>(byte_count));
}

} // namespace astro

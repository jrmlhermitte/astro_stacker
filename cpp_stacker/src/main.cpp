#include "calibration.hpp"
#include "demosaic.hpp"
#include "integrator.hpp"
#include "io_export.hpp"
#include "raw_loader.hpp"
#include "registration.hpp"
#include "star_detector.hpp"
#include "types.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliArgs {
    std::string lights_dir;
    std::string darks_dir;
    std::string out_npy;
    std::string out_transforms;
    std::string out_fits;
    std::string out_tif;
    std::string out_preview_tif;
};

CliArgs parse_cli(int argc, char** argv) {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next_val = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for argument: " + key);
            }
            return argv[++i];
        };
        if (key == "--lights") args.lights_dir = next_val();
        else if (key == "--darks") args.darks_dir = next_val();
        else if (key == "--out-npy") args.out_npy = next_val();
        else if (key == "--out-transforms") args.out_transforms = next_val();
        else if (key == "--out-fits") args.out_fits = next_val();
        else if (key == "--out-tif") args.out_tif = next_val();
        else if (key == "--out-preview-tif") args.out_preview_tif = next_val();
        else {
            throw std::runtime_error("Unknown CLI argument: " + key);
        }
    }
    if (args.lights_dir.empty() || args.darks_dir.empty() ||
        args.out_npy.empty() || args.out_transforms.empty()) {
        throw std::runtime_error(
            "Usage: astro_stacker --lights <dir> --darks <dir> --out-npy <file.npy> "
            "--out-transforms <transforms.json> [--out-fits <f.fits>] [--out-tif <f.tif>] "
            "[--out-preview-tif <preview16.tif>]"
        );
    }
    return args;
}

double elapsed_sec(
    const std::chrono::steady_clock::time_point& t0,
    const std::chrono::steady_clock::time_point& t1
) {
    return std::chrono::duration<double>(t1 - t0).count();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto t_total0 = std::chrono::steady_clock::now();
        const CliArgs args = parse_cli(argc, argv);

        // 1. Load & validate darks and lights (.CR2 strictly)
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<astro::RawSensorFrame> clean_darks;
        std::vector<astro::RawFrameMeta> dark_metas;
        astro::load_cr2_directory(args.darks_dir, clean_darks, dark_metas);

        std::vector<astro::RawSensorFrame> clean_lights;
        std::vector<astro::RawFrameMeta> light_metas;
        astro::load_cr2_directory(args.lights_dir, clean_lights, light_metas);
        const auto t1 = std::chrono::steady_clock::now();

        if (clean_lights.empty()) {
            throw std::runtime_error("Zero clean light frames found in " + args.lights_dir);
        }

        // 2. Build master dark & calibrate + demosaic all clean light frames
        const std::vector<float> master_dark = astro::build_master_dark(clean_darks);
        const int n_lights = static_cast<int>(clean_lights.size());
        std::vector<std::vector<float>> rgb_frames(n_lights);
        std::vector<astro::FrameQuality> qualities(n_lights);

        const auto t2 = std::chrono::steady_clock::now();
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < n_lights; ++i) {
            std::vector<float> cfa_cal;
            astro::calibrate_light_cfa(clean_lights[i], master_dark, cfa_cal);
            rgb_frames[i].resize(astro::ACTIVE_PIXELS * 3);
            astro::demosaic_mhc_rggb(cfa_cal.data(), astro::ACTIVE_H, astro::ACTIVE_W, rgb_frames[i].data());

            std::vector<float> lum;
            astro::compute_luminance_from_rgb(rgb_frames[i].data(), astro::ACTIVE_H, astro::ACTIVE_W, lum);
            qualities[i] = astro::analyze_frame_stars(
                lum,
                i,
                clean_lights[i].meta.filename,
                clean_lights[i].meta.valid_active_rows
            );
        }
        const auto t3 = std::chrono::steady_clock::now();

        // Free raw uint16 buffers now that calibrated float32 RGB frames are ready
        clean_darks.clear();
        clean_darks.shrink_to_fit();
        clean_lights.clear();
        clean_lights.shrink_to_fit();

        // 3. Select sharpest reference frame & register all frames
        const int ref_idx = astro::select_sharpest_reference_frame(qualities);
        const std::vector<astro::FrameTransform> transforms = astro::register_all_frames(qualities, ref_idx);
        const auto t4 = std::chrono::steady_clock::now();

        // 4. Integrate stack
        const std::vector<float> stacked_rgb = astro::integrate_frames(rgb_frames, transforms);
        const auto t5 = std::chrono::steady_clock::now();

        // 5. Export outputs
        astro::write_npy_f32_rgb(args.out_npy, stacked_rgb);
        astro::write_transforms_json(args.out_transforms, qualities[ref_idx].filename, transforms);
        if (!args.out_fits.empty()) {
            astro::write_fits_f32_rgb(args.out_fits, stacked_rgb);
        }
        if (!args.out_tif.empty()) {
            astro::write_tiff_f32_rgb(args.out_tif, stacked_rgb);
        }
        if (!args.out_preview_tif.empty()) {
            astro::write_tiff_u16_preview_rgb(args.out_preview_tif, stacked_rgb);
        }
        const auto t6 = std::chrono::steady_clock::now();

        std::cout << std::fixed << std::setprecision(4)
                  << "{\n"
                  << "  \"omp_max_threads\": " << omp_get_max_threads() << ",\n"
                  << "  \"darks_inspected\": " << dark_metas.size() << ",\n"
                  << "  \"darks_clean\": " << (dark_metas.size() - [&]() {
                         size_t bad = 0; for (const auto& m : dark_metas) if (!m.is_clean) ++bad; return bad;
                     }()) << ",\n"
                  << "  \"lights_inspected\": " << light_metas.size() << ",\n"
                  << "  \"lights_clean\": " << n_lights << ",\n"
                  << "  \"selected_reference\": \"" << qualities[ref_idx].filename << "\",\n"
                  << "  \"ref_detected_stars\": " << qualities[ref_idx].star_count << ",\n"
                  << "  \"ref_moment_fwhm_px\": " << qualities[ref_idx].median_fwhm_px << ",\n"
                  << "  \"timing_s\": {\n"
                  << "    \"raw_unpack_validate\": " << elapsed_sec(t0, t1) << ",\n"
                  << "    \"master_dark\": " << elapsed_sec(t1, t2) << ",\n"
                  << "    \"calibrate_demosaic_detect\": " << elapsed_sec(t2, t3) << ",\n"
                  << "    \"registration\": " << elapsed_sec(t3, t4) << ",\n"
                  << "    \"integration\": " << elapsed_sec(t4, t5) << ",\n"
                  << "    \"export_io\": " << elapsed_sec(t5, t6) << ",\n"
                  << "    \"total_wall_s\": " << elapsed_sec(t_total0, t6) << "\n"
                  << "  }\n"
                  << "}\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}

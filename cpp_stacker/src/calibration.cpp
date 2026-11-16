#include "calibration.hpp"

#include <stdexcept>

namespace astro {

std::vector<float> build_master_dark(const std::vector<RawSensorFrame>& clean_darks) {
    if (clean_darks.empty()) {
        throw std::runtime_error("No clean dark frames available to build master dark");
    }

    std::vector<float> master_dark(ACTIVE_PIXELS, 0.0f);
    const int n_darks = static_cast<int>(clean_darks.size());
    const double inv_n = 1.0 / static_cast<double>(n_darks);

    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < ACTIVE_PIXELS; ++idx) {
        double sum = 0.0;
        for (int i = 0; i < n_darks; ++i) {
            sum += static_cast<double>(clean_darks[i].active_cfa[idx]) - 2048.0;
        }
        master_dark[idx] = static_cast<float>(sum * inv_n);
    }

    return master_dark;
}

void calibrate_light_cfa(
    const RawSensorFrame& light,
    const std::vector<float>& master_dark,
    std::vector<float>& cfa_out,
    bool use_omp
) {
    cfa_out.resize(ACTIVE_PIXELS);

    #pragma omp parallel for schedule(static) if(use_omp)
    for (size_t idx = 0; idx < ACTIVE_PIXELS; ++idx) {
        const float raw_sub_bias = static_cast<float>(light.active_cfa[idx]) - 2048.0f;
        cfa_out[idx] = raw_sub_bias - master_dark[idx];
    }
}

} // namespace astro

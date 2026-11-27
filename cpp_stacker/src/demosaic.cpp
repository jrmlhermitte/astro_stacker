#include "demosaic.hpp"

#include <cstddef>

namespace astro {

namespace {

inline int reflect101(int idx, int n) {
    if (idx < 0) return -idx;
    if (idx >= n) return 2 * (n - 1) - idx;
    return idx;
}

} // namespace

// CAMERA-SPECIFIC (RGGB Bayer CFA):
// Assumes (r_even, c_even)=R, (r_even, c_odd)=G, (r_odd, c_even)=G, (r_odd, c_odd)=B.
// Supporting BGGR / GRBG / GBRG Bayer sensors requires passing the 2x2 CFA phase offset
// from LibRaw::FC(r, c); non-Bayer sensors (e.g. Fujifilm X-Trans) require a separate demosaicer.
void demosaic_mhc_rggb(
    const float* cfa,
    int height,
    int width,
    float* rgb_out,
    bool use_omp
) {
    auto at_border = [&](int r, int c) -> float {
        const int rr = reflect101(r, height);
        const int cc = reflect101(c, width);
        return cfa[static_cast<size_t>(rr) * width + cc];
    };

    #pragma omp parallel for schedule(static) if(use_omp)
    for (int r = 0; r < height; ++r) {
        const bool interior_r = (r >= 2 && r < height - 2);
        const bool r_even = ((r & 1) == 0);
        float* out_row = rgb_out + static_cast<size_t>(r) * width * 3;

        for (int c = 0; c < width; ++c) {
            float c0, n1, s1, w1, e1, n2, s2, w2, e2, nw, ne, sw, se;
            if (interior_r && c >= 2 && c < width - 2) {
                const float* p = cfa + static_cast<size_t>(r) * width + c;
                c0 = p[0];
                n1 = p[-width];       s1 = p[width];
                w1 = p[-1];           e1 = p[1];
                n2 = p[-2 * width];   s2 = p[2 * width];
                w2 = p[-2];           e2 = p[2];
                nw = p[-width - 1];   ne = p[-width + 1];
                sw = p[width - 1];    se = p[width + 1];
            } else {
                c0 = at_border(r, c);
                n1 = at_border(r - 1, c); s1 = at_border(r + 1, c);
                w1 = at_border(r, c - 1); e1 = at_border(r, c + 1);
                n2 = at_border(r - 2, c); s2 = at_border(r + 2, c);
                w2 = at_border(r, c - 2); e2 = at_border(r, c + 2);
                nw = at_border(r - 1, c - 1); ne = at_border(r - 1, c + 1);
                sw = at_border(r + 1, c - 1); se = at_border(r + 1, c + 1);
            }

            const float h1 = w1 + e1;
            const float v1 = n1 + s1;
            const float h2 = w2 + e2;
            const float v2 = n2 + s2;
            const float ax1 = h1 + v1;
            const float ax2 = h2 + v2;
            const float dg1 = nw + ne + sw + se;

            const bool c_even = ((c & 1) == 0);
            float R, G, B;
            if (r_even) {
                if (c_even) {
                    R = c0;
                    G = 0.125f * (4.0f * c0 + 2.0f * ax1 - ax2);
                    B = 0.125f * (6.0f * c0 + 2.0f * dg1 - 1.5f * ax2);
                } else {
                    G = c0;
                    R = 0.125f * (5.0f * c0 + 4.0f * h1 - dg1 - h2 + 0.5f * v2);
                    B = 0.125f * (5.0f * c0 + 4.0f * v1 - dg1 + 0.5f * h2 - v2);
                }
            } else {
                if (c_even) {
                    G = c0;
                    R = 0.125f * (5.0f * c0 + 4.0f * v1 - dg1 + 0.5f * h2 - v2);
                    B = 0.125f * (5.0f * c0 + 4.0f * h1 - dg1 - h2 + 0.5f * v2);
                } else {
                    B = c0;
                    G = 0.125f * (4.0f * c0 + 2.0f * ax1 - ax2);
                    R = 0.125f * (6.0f * c0 + 2.0f * dg1 - 1.5f * ax2);
                }
            }

            out_row[3 * c + 0] = R;
            out_row[3 * c + 1] = G;
            out_row[3 * c + 2] = B;
        }
    }
}

void compute_luminance_from_rgb(
    const float* rgb,
    int height,
    int width,
    std::vector<float>& lum_out,
    bool use_omp
) {
    const size_t n = static_cast<size_t>(height) * width;
    lum_out.resize(n);

    #pragma omp parallel for schedule(static) if(use_omp)
    for (size_t i = 0; i < n; ++i) {
        lum_out[i] = 0.25f * rgb[3 * i + 0] + 0.50f * rgb[3 * i + 1] + 0.25f * rgb[3 * i + 2];
    }
}

} // namespace astro

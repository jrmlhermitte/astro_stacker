#pragma once

#include "types.hpp"
#include <string>
#include <vector>

namespace astro {

// Strictly enumerate .CR2 files in sorted order (ignoring .JPG, .png, .xmp, .npz, .xcf)
std::vector<std::string> list_strict_cr2_files(const std::string& dir_path);

// Unpack a single Canon .CR2 file via thread-safe LibRaw and apply bitstream + 14-bit ADC gates
bool load_and_validate_cr2(const std::string& filepath, RawSensorFrame& out_frame);

// Load all .CR2 files in a directory in parallel, separating clean frames and rejected metadata
void load_cr2_directory(
    const std::string& dir_path,
    std::vector<RawSensorFrame>& clean_frames_out,
    std::vector<RawFrameMeta>& all_metas_out
);

} // namespace astro

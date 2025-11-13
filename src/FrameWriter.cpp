#include "FrameWriter.h"

#include <fstream>
#include <stdexcept>

void FrameWriter::writeFrameAsPPM(const std::string& path, uint32_t width, uint32_t height,
                                  const std::vector<uint8_t>& rgbaPixels) {
    if (rgbaPixels.size() < static_cast<size_t>(width) * height * 4) {
        throw std::runtime_error("Not enough data to write frame.");
    }

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    file << "P6\n" << width << " " << height << "\n255\n";

    std::vector<uint8_t> rgb;
    rgb.reserve(static_cast<size_t>(width) * height * 3);
    for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i) {
        rgb.push_back(rgbaPixels[i * 4 + 0]);
        rgb.push_back(rgbaPixels[i * 4 + 1]);
        rgb.push_back(rgbaPixels[i * 4 + 2]);
    }

    file.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    file.close();
}

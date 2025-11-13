#pragma once

#include <cstdint>
#include <string>
#include <vector>

class FrameWriter {
public:
    static void writeFrameAsPPM(const std::string& path, uint32_t width, uint32_t height,
                                const std::vector<uint8_t>& rgbaPixels);
};

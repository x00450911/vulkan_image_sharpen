#include "FrameProvider.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

namespace {
constexpr uint32_t kChannels = 4;
constexpr float kPi = 3.14159265358979323846f;

uint8_t clampToByte(float value) {
    value = std::clamp(value, 0.0f, 255.0f);
    return static_cast<uint8_t>(value + 0.5f);
}
}  // namespace

SyntheticFrameProvider::SyntheticFrameProvider(FrameDimensions dims, uint32_t frameCount, float noiseAmplitude)
    : dims_(dims),
      totalFrames_(frameCount),
      noiseAmplitude_(noiseAmplitude),
      rng_(1337u) {}

std::optional<FrameDimensions> SyntheticFrameProvider::getFrameDimensions() const {
    return dims_;
}

bool SyntheticFrameProvider::getNextFrame(std::vector<uint8_t>& outRgbaPixels) {
    if (nextFrameIndex_ >= totalFrames_) {
        return false;
    }

    outRgbaPixels.resize(static_cast<size_t>(dims_.width) * dims_.height * kChannels);

    std::uniform_real_distribution<float> noiseDist(-noiseAmplitude_, noiseAmplitude_);

    const float time = static_cast<float>(nextFrameIndex_) / static_cast<float>(std::max(totalFrames_, 1u));
    const float centerX = 0.5f + 0.2f * std::sin(2.0f * kPi * time);
    const float centerY = 0.5f + 0.2f * std::cos(2.0f * kPi * time);
    const float radius = 0.15f;

    for (uint32_t y = 0; y < dims_.height; ++y) {
        for (uint32_t x = 0; x < dims_.width; ++x) {
            const float nx = static_cast<float>(x) / static_cast<float>(dims_.width);
            const float ny = static_cast<float>(y) / static_cast<float>(dims_.height);

            const float dx = nx - centerX;
            const float dy = ny - centerY;
            const float dist = std::sqrt(dx * dx + dy * dy);

            float base = 120.0f + 80.0f * std::sin(4.0f * kPi * nx) * std::cos(4.0f * kPi * ny);
            float spot = 0.0f;
            if (dist < radius) {
                float falloff = (radius - dist) / radius;
                spot = 80.0f * std::pow(falloff, 1.5f);
            }

            const float noise = noiseDist(rng_) * 255.0f;

            const uint8_t value = clampToByte(base + spot + noise);

            const size_t index = (static_cast<size_t>(y) * dims_.width + x) * kChannels;
            outRgbaPixels[index + 0] = value;
            outRgbaPixels[index + 1] = value;
            outRgbaPixels[index + 2] = value;
            outRgbaPixels[index + 3] = 255;
        }
    }

    ++nextFrameIndex_;
    return true;
}

void SyntheticFrameProvider::reset() {
    nextFrameIndex_ = 0;
}

#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <vector>

struct FrameDimensions {
    uint32_t width;
    uint32_t height;
};

class IFrameProvider {
public:
    virtual ~IFrameProvider() = default;

    virtual std::optional<FrameDimensions> getFrameDimensions() const = 0;
    virtual bool getNextFrame(std::vector<uint8_t>& outRgbaPixels) = 0;
    virtual void reset() {}
};

class SyntheticFrameProvider final : public IFrameProvider {
public:
    SyntheticFrameProvider(FrameDimensions dims, uint32_t frameCount, float noiseAmplitude);

    std::optional<FrameDimensions> getFrameDimensions() const override;
    bool getNextFrame(std::vector<uint8_t>& outRgbaPixels) override;
    void reset() override;

private:
    FrameDimensions dims_;
    uint32_t totalFrames_;
    float noiseAmplitude_;
    uint32_t nextFrameIndex_{0};
    std::mt19937 rng_;
};

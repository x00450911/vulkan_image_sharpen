#pragma once

#include "accelerator/accelerator.h"
#include "image.h"

struct DisplayEnhancerConfig {
    float targetLuminanceNits = 350.f;
    float toneCurveMid = 0.4f;
    float toneCurveSlope = 1.35f;
    float localContrastStrength = 0.85f;
    int localContrastRadius = 3;
    float sharpness = 0.12f;
    float saturationBoost = 1.05f;
};

struct Adreno735PerfHints {
    int waveSize = 64;
    int tileSize = 32;
    bool enableAsyncCompute = true;
    bool enableCooperativeMatrix = true;
};

class DisplayEnhancer {
public:
    DisplayEnhancer(DisplayEnhancerConfig config, Adreno735PerfHints hints = {});

    Image enhance(const Image &input, AcceleratorRuntime &runtime) const;

private:
    void applyToneCurve(Image &image, AcceleratorRuntime &runtime) const;
    void applyLocalContrast(Image &image, AcceleratorRuntime &runtime) const;
    void applySharpen(Image &image) const;
    void applySaturation(Image &image) const;

    DisplayEnhancerConfig config_;
    Adreno735PerfHints hints_;
};

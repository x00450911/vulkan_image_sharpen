#pragma once

#include <memory>

#include "accelerator/accelerator.h"
#include "image.h"

struct SuperResolutionConfig {
    int upscaleFactor = 2;
    float detailBoost = 1.1f;
    float refinementStrength = 0.5f;
};

class SuperResolution {
public:
    explicit SuperResolution(SuperResolutionConfig config);

    Image upscale(const Image &input, AcceleratorRuntime &runtime) const;

private:
    SuperResolutionConfig config_;
    Image runSpatialUpScale(const Image &input) const;
    void runDetailEnhancement(Image &image, AcceleratorRuntime &runtime) const;
};

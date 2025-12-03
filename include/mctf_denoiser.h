#pragma once

#include <vector>

#include "accelerator/accelerator.h"
#include "image.h"

struct MCTFConfig {
    int temporalRadius = 2;
    int searchWindow = 2;
    float temporalStrength = 0.8f;
};

struct MotionVector {
    int dx = 0;
    int dy = 0;
};

class MCTFDenoiser {
public:
    explicit MCTFDenoiser(MCTFConfig config);

    Image denoise(const std::vector<Image> &frames, AcceleratorRuntime &runtime) const;

private:
    using MotionField = std::vector<MotionVector>;

    MCTFConfig config_;

    MotionField estimateField(const Image &reference, const Image &neighbor) const;
    Image fuseFrames(const std::vector<Image> &frames, size_t referenceIndex,
                     const std::vector<MotionField> &fields) const;
};

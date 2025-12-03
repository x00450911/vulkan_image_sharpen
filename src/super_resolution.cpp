#include "super_resolution.h"

#include <algorithm>
#include <cmath>
#include <memory>

namespace {
float clamp01(float v) {
    return std::max(0.0f, std::min(1.0f, v));
}
}

SuperResolution::SuperResolution(SuperResolutionConfig config) : config_(config) {}

Image SuperResolution::upscale(const Image &input, AcceleratorRuntime &runtime) const {
    if (input.empty()) {
        return input;
    }
    Image spatial = runSpatialUpScale(input);
    runDetailEnhancement(spatial, runtime);
    return spatial;
}

Image SuperResolution::runSpatialUpScale(const Image &input) const {
    const int targetWidth = input.width() * config_.upscaleFactor;
    const int targetHeight = input.height() * config_.upscaleFactor;
    return input.resizedBilinear(targetWidth, targetHeight);
}

void SuperResolution::runDetailEnhancement(Image &image, AcceleratorRuntime &runtime) const {
    const int w = image.width();
    const int h = image.height();
    const int c = image.channels();

    auto detailMap = std::make_shared<Image>(w, h, c);

    auto edgeFuture = runtime.dispatch(Workload{
        "EdgeLift", AcceleratorKind::GPU, [detailMap, &image, w, h, c, boost = config_.detailBoost]() {
            for (int y = 1; y < h - 1; ++y) {
                for (int x = 1; x < w - 1; ++x) {
                    for (int ch = 0; ch < c; ++ch) {
                        const float center = image.at(x, y, ch);
                        const float laplace = 4.f * center - image.at(x - 1, y, ch) - image.at(x + 1, y, ch) -
                                              image.at(x, y - 1, ch) - image.at(x, y + 1, ch);
                        detailMap->at(x, y, ch) = laplace * boost;
                    }
                }
            }
        }});

    edgeFuture.get();

    auto refineFuture = runtime.dispatch(Workload{
        "NeuralRefine", AcceleratorKind::NPU,
        [detailMap, &image, strength = config_.refinementStrength, w, h, c]() {
            const int kernelRadius = 1;
            const float weights[3][3] = {
                {0.0625f, 0.125f, 0.0625f},
                {0.125f, 0.25f, 0.125f},
                {0.0625f, 0.125f, 0.0625f},
            };
            for (int y = 1; y < h - 1; ++y) {
                for (int x = 1; x < w - 1; ++x) {
                    for (int ch = 0; ch < c; ++ch) {
                        float accum = 0.f;
                        for (int ky = -kernelRadius; ky <= kernelRadius; ++ky) {
                            for (int kx = -kernelRadius; kx <= kernelRadius; ++kx) {
                                accum += detailMap->at(x + kx, y + ky, ch) *
                                         weights[ky + kernelRadius][kx + kernelRadius];
                            }
                        }
                        const float refined = image.at(x, y, ch) + accum * strength;
                        image.at(x, y, ch) = clamp01(refined);
                    }
                }
            }
        }});

    refineFuture.get();
}

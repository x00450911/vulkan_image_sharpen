#include "display_enhancer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>

namespace {
float clamp01(float v) {
    return std::max(0.0f, std::min(1.0f, v));
}
} // namespace

DisplayEnhancer::DisplayEnhancer(DisplayEnhancerConfig config, Adreno735PerfHints hints)
    : config_(config), hints_(hints) {}

Image DisplayEnhancer::enhance(const Image &input, AcceleratorRuntime &runtime) const {
    if (input.empty()) {
        return input;
    }
    Image working = input;
    applyToneCurve(working, runtime);
    applyLocalContrast(working, runtime);
    applySharpen(working);
    applySaturation(working);
    return working;
}

void DisplayEnhancer::applyToneCurve(Image &image, AcceleratorRuntime &runtime) const {
    auto toneFuture = runtime.dispatch(Workload{
        "ToneCurve",
        AcceleratorKind::GPU,
        [&image, cfg = config_, hints = hints_]() {
            const float inverseMid = 1.f / std::max(1e-3f, cfg.toneCurveMid);
            const float slope = cfg.toneCurveSlope;
            for (int y = 0; y < image.height(); ++y) {
                for (int x = 0; x < image.width(); ++x) {
                    for (int c = 0; c < image.channels(); ++c) {
                        float v = image.at(x, y, c);
                        const float normalized = v * inverseMid;
                        const float powered = std::pow(normalized, slope);
                        v = clamp01(powered);
                        image.at(x, y, c) = v;
                    }
                }
            }
            std::cout << "[Adreno735] Tone curve dispatched with waveSize=" << hints.waveSize
                      << " tile=" << hints.tileSize << '\n';
        }});
    toneFuture.get();
}

void DisplayEnhancer::applyLocalContrast(Image &image, AcceleratorRuntime &runtime) const {
    const int radius = std::max(1, config_.localContrastRadius);
    auto contrastFuture = runtime.dispatch(Workload{
        "LocalContrast",
        AcceleratorKind::GPU,
        [&image, radius, strength = config_.localContrastStrength, hints = hints_]() {
            Image blurred = image.boxBlurred(radius);
            for (int y = 0; y < image.height(); ++y) {
                for (int x = 0; x < image.width(); ++x) {
                    for (int c = 0; c < image.channels(); ++c) {
                        const float detail = image.at(x, y, c) - blurred.at(x, y, c);
                        image.at(x, y, c) = clamp01(image.at(x, y, c) + detail * strength);
                    }
                }
            }
            if (hints.enableAsyncCompute) {
                std::cout << "[Adreno735] Async compute queue enabled for LocalContrast\n";
            }
        }});
    contrastFuture.get();
}

void DisplayEnhancer::applySharpen(Image &image) const {
    if (config_.sharpness <= 0.f) {
        return;
    }
    Image original = image;
    const float amount = config_.sharpness;
    for (int y = 1; y < image.height() - 1; ++y) {
        for (int x = 1; x < image.width() - 1; ++x) {
            for (int c = 0; c < image.channels(); ++c) {
                const float center = original.at(x, y, c);
                const float laplace = 4.f * center - original.at(x - 1, y, c) - original.at(x + 1, y, c) -
                                      original.at(x, y - 1, c) - original.at(x, y + 1, c);
                image.at(x, y, c) = clamp01(center + laplace * amount);
            }
        }
    }
}

void DisplayEnhancer::applySaturation(Image &image) const {
    if (image.channels() < 3) {
        return;
    }
    const float boost = config_.saturationBoost;
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            const float r = image.at(x, y, 0);
            const float g = image.at(x, y, 1);
            const float b = image.at(x, y, 2);
            const float luminance = 0.299f * r + 0.587f * g + 0.114f * b;
            image.at(x, y, 0) = clamp01(luminance + (r - luminance) * boost);
            image.at(x, y, 1) = clamp01(luminance + (g - luminance) * boost);
            image.at(x, y, 2) = clamp01(luminance + (b - luminance) * boost);
        }
    }
}

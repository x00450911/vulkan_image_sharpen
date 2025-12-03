#include "mctf_denoiser.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {
int clampInt(int v, int lo, int hi) {
    return std::max(lo, std::min(hi, v));
}

size_t idx(int x, int y, int width) {
    return static_cast<size_t>(y * width + x);
}
}

MCTFDenoiser::MCTFDenoiser(MCTFConfig config) : config_(config) {}

Image MCTFDenoiser::denoise(const std::vector<Image> &frames, AcceleratorRuntime &runtime) const {
    if (frames.empty()) {
        throw std::invalid_argument("MCTF requires at least one frame");
    }
    const size_t referenceIndex = frames.size() / 2;
    const Image &reference = frames.at(referenceIndex);

    auto fields = std::make_shared<std::vector<MotionField>>(frames.size());

    auto motionFuture = runtime.dispatch(Workload{
        "MotionField", AcceleratorKind::DSP,
        [this, fields, &frames, referenceIndex]() {
            const Image &ref = frames.at(referenceIndex);
            for (size_t i = 0; i < frames.size(); ++i) {
                if (i == referenceIndex) {
                    continue;
                }
                (*fields)[i] = estimateField(ref, frames.at(i));
            }
        }});

    motionFuture.get();

    auto fusedImage = std::make_shared<Image>();
    auto fusionFuture = runtime.dispatch(Workload{
        "TemporalFuse", AcceleratorKind::CPU,
        [this, fusedImage, fields, &frames, referenceIndex]() {
            *fusedImage = fuseFrames(frames, referenceIndex, *fields);
        }});

    fusionFuture.get();

    return *fusedImage;
}

MCTFDenoiser::MotionField MCTFDenoiser::estimateField(const Image &reference, const Image &neighbor) const {
    if (reference.width() != neighbor.width() || reference.height() != neighbor.height() ||
        reference.channels() != neighbor.channels()) {
        throw std::invalid_argument("Frames must match dimensions");
    }
    const int w = reference.width();
    const int h = reference.height();
    const int c = reference.channels();
    MotionField field(static_cast<size_t>(w) * h, MotionVector{});

    const int searchRadius = std::max(1, config_.searchWindow);

    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            float bestScore = std::numeric_limits<float>::max();
            MotionVector best{0, 0};
            for (int dy = -searchRadius; dy <= searchRadius; ++dy) {
                for (int dx = -searchRadius; dx <= searchRadius; ++dx) {
                    float score = 0.f;
                    for (int ch = 0; ch < c; ++ch) {
                        const float refPx = reference.at(x, y, ch);
                        const float nbrPx = neighbor.at(clampInt(x + dx, 0, w - 1), clampInt(y + dy, 0, h - 1), ch);
                        const float diff = refPx - nbrPx;
                        score += diff * diff;
                    }
                    if (score < bestScore) {
                        bestScore = score;
                        best = {dx, dy};
                    }
                }
            }
            field[idx(x, y, w)] = best;
        }
    }
    return field;
}

Image MCTFDenoiser::fuseFrames(const std::vector<Image> &frames, size_t referenceIndex,
                               const std::vector<MotionField> &fields) const {
    const Image &reference = frames.at(referenceIndex);
    Image output(reference.width(), reference.height(), reference.channels());

    for (int y = 0; y < reference.height(); ++y) {
        for (int x = 0; x < reference.width(); ++x) {
            for (int ch = 0; ch < reference.channels(); ++ch) {
                float accum = reference.at(x, y, ch);
                float weightSum = 1.f;
                for (size_t idxFrame = 0; idxFrame < frames.size(); ++idxFrame) {
                    if (idxFrame == referenceIndex || idxFrame >= fields.size() || fields[idxFrame].empty()) {
                        continue;
                    }
                    const MotionVector mv = fields[idxFrame][idx(x, y, reference.width())];
                    const int sampleX = clampInt(x + mv.dx, 0, reference.width() - 1);
                    const int sampleY = clampInt(y + mv.dy, 0, reference.height() - 1);
                    const float sample = frames[idxFrame].at(sampleX, sampleY, ch);
                    const float blendWeight = config_.temporalStrength / static_cast<float>(frames.size());
                    accum += sample * blendWeight;
                    weightSum += blendWeight;
                }
                output.at(x, y, ch) = accum / weightSum;
            }
        }
    }
    return output;
}

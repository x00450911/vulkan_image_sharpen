#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "accelerator/accelerator.h"
#include "display_enhancer.h"
#include "image.h"
#include "mctf_denoiser.h"
#include "super_resolution.h"

Image AddNoise(const Image &input, float amount, int seed) {
    Image noisy = input;
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, amount);
    for (int y = 0; y < noisy.height(); ++y) {
        for (int x = 0; x < noisy.width(); ++x) {
            for (int c = 0; c < noisy.channels(); ++c) {
                noisy.at(x, y, c) = std::clamp(noisy.at(x, y, c) + dist(rng), 0.f, 1.f);
            }
        }
    }
    return noisy;
}

int main() {
    QualcommGen2Runtime runtime;
    runtime.logTopology();
    runtime.setPowerMode("burst");

    SuperResolution sr({2, 1.25f, 0.65f});
    MCTFDenoiser denoiser({2, 2, 0.9f});
    DisplayEnhancer enhancer({350.f, 0.42f, 1.4f, 0.9f, 3, 0.15f, 1.08f}, {64, 32, true, true});

    std::vector<Image> frames;
    const int frameCount = 5;
    for (int i = 0; i < frameCount; ++i) {
        Image base = LoadSyntheticGradient(128, 72, 3);
        frames.push_back(AddNoise(base, 0.05f + 0.01f * i, 42 + i));
    }

    auto denoised = denoiser.denoise(frames, runtime);
    auto enhanced = sr.upscale(denoised, runtime);
    auto displayReady = enhancer.enhance(enhanced, runtime);

    SaveAsPGM(displayReady, "display_enhanced.pgm");

    std::cout << "Pipeline complete. Output written to display_enhanced.pgm\n";

    for (auto kind : {AcceleratorKind::CPU, AcceleratorKind::DSP, AcceleratorKind::GPU, AcceleratorKind::NPU}) {
        const auto metrics = runtime.latestMetrics(kind);
        std::cout << "  Metrics " << static_cast<int>(metrics.utilization * 100) << "% util, "
                  << metrics.latencyMs << " ms latency" << '\n';
    }
    return 0;
}

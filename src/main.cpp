// Example program demonstrating block-based motion vector estimation for denoising.

#include "motion_vectors.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace denoise;

namespace {

std::vector<uint8_t> make_reference_pattern(int width, int height) {
    std::vector<uint8_t> frame(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const double gx = std::sin(static_cast<double>(x) * 0.12);
            const double gy = std::cos(static_cast<double>(y) * 0.15);
            const double pattern = 0.55 + 0.30 * gx + 0.15 * gy;
            const double value = std::clamp(pattern, 0.0, 1.0) * 255.0;
            frame[static_cast<std::size_t>(y) * width + static_cast<std::size_t>(x)] =
                static_cast<uint8_t>(value);
        }
    }
    return frame;
}

std::vector<uint8_t> apply_shift_with_noise(const std::vector<uint8_t>& source,
                                            int width,
                                            int height,
                                            int shift_x,
                                            int shift_y,
                                            double noise_sigma,
                                            std::mt19937& rng) {
    std::vector<uint8_t> result(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);
    std::normal_distribution<double> gaussian(0.0, noise_sigma);

    for (int y = 0; y < height; ++y) {
        const int src_y = y - shift_y;
        if (src_y < 0 || src_y >= height) {
            continue;
        }
        for (int x = 0; x < width; ++x) {
            const int src_x = x - shift_x;
            if (src_x < 0 || src_x >= width) {
                continue;
            }
            const double noisy =
                static_cast<double>(source[static_cast<std::size_t>(src_y) * width +
                                           static_cast<std::size_t>(src_x)]) +
                gaussian(rng);
            const double clamped = std::clamp(noisy, 0.0, 255.0);
            result[static_cast<std::size_t>(y) * width + static_cast<std::size_t>(x)] =
                static_cast<uint8_t>(clamped);
        }
    }

    return result;
}

}  // namespace

int main() {
    constexpr int width = 96;
    constexpr int height = 64;
    constexpr int shift_x = 3;
    constexpr int shift_y = 2;

    std::mt19937 rng(42);

    const auto reference_frame = make_reference_pattern(width, height);
    const auto target_frame = apply_shift_with_noise(reference_frame,
                                                     width,
                                                     height,
                                                     shift_x,
                                                     shift_y,
                                                     /*noise_sigma=*/7.5,
                                                     rng);

    MotionEstimationParams params;
    params.block_size = 8;
    params.search_range = 6;
    params.step = 8;
    params.cost_type = MatchingCostType::SAD;
    params.penalty_lambda = 0.05;

    const auto field = compute_motion_vectors(make_frame_view(reference_frame, width, height),
                                              make_frame_view(target_frame, width, height),
                                              params);

    double mean_dx = 0.0;
    double mean_dy = 0.0;
    for (const auto& mv : field) {
        mean_dx += static_cast<double>(mv.dx);
        mean_dy += static_cast<double>(mv.dy);
    }
    if (!field.empty()) {
        mean_dx /= static_cast<double>(field.size());
        mean_dy /= static_cast<double>(field.size());
    }

    std::cout << "Estimated motion vectors (sample of first 10 blocks):\n";
    std::cout << " block(x,y) -> (dx, dy) | cost\n";
    std::cout << "--------------------------------\n";
    const std::size_t sample_count = std::min<std::size_t>(10, field.size());
    for (std::size_t i = 0; i < sample_count; ++i) {
        const auto& mv = field[i];
        std::cout << " (" << std::setw(2) << mv.block_x << "," << std::setw(2) << mv.block_y
                  << ") -> (" << std::setw(2) << mv.dx << "," << std::setw(2) << mv.dy
                  << ") | " << std::fixed << std::setprecision(2) << mv.cost << '\n';
    }

    std::cout << "\nMean motion: dx=" << std::fixed << std::setprecision(2) << mean_dx
              << ", dy=" << mean_dy << '\n';
    std::cout << "Ground-truth shift: dx=" << shift_x << ", dy=" << shift_y << '\n';
    std::cout << "\nUse the motion vectors to warp neighbouring frames prior to temporal denoising.\n";

    return 0;
}


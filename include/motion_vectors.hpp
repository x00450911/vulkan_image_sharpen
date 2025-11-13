// Motion vector estimation utilities for video denoising workflows.
// The implementation provides a block-matching motion estimator that can be
// used to align neighbouring frames prior to temporal denoising.

#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace denoise {

/// Cost metric used for block matching.
enum class MatchingCostType {
    /// Sum of absolute differences (L1).
    SAD,
    /// Sum of squared differences (L2).
    SSD
};

/// Lightweight view over a planar 8-bit grayscale frame.
struct FrameView {
    const uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;  // Bytes between two consecutive rows.

    constexpr bool is_valid() const noexcept {
        return data != nullptr && width > 0 && height > 0 && stride >= width;
    }

    constexpr bool contains(int x, int y) const noexcept {
        return x >= 0 && x < width && y >= 0 && y < height;
    }

    uint8_t at(int x, int y) const noexcept {
        return data[y * stride + x];
    }
};

/// Convenience helper to wrap a contiguous grayscale buffer.
inline FrameView make_frame_view(const std::vector<uint8_t>& buffer, int width, int height) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("Frame dimensions must be positive.");
    }
    const std::size_t required = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (buffer.size() < required) {
        throw std::invalid_argument("Buffer too small for requested frame dimensions.");
    }
    return FrameView{buffer.data(), width, height, width};
}

/// Block motion vector information.
struct BlockMotionVector {
    int block_x = 0;
    int block_y = 0;
    int dx = 0;
    int dy = 0;
    double cost = 0.0;  // Normalised matching cost per pixel.
};

/// Parameters that control block matching.
struct MotionEstimationParams {
    int block_size = 8;
    int search_range = 8;
    int step = 0;  // Defaults to block_size when zero or negative.
    MatchingCostType cost_type = MatchingCostType::SAD;
    double penalty_lambda = 0.0;  // Optional regularisation for large motion.
};

/// Computes block-based motion vectors between reference and target frames.
///
/// @param reference  Frame at time t.
/// @param target     Frame at time t + 1 (or t - 1).
/// @param params     Motion estimation parameters.
/// @return A vector of block motion vectors covering the reference frame.
std::vector<BlockMotionVector> compute_motion_vectors(
    const FrameView& reference,
    const FrameView& target,
    const MotionEstimationParams& params);

}  // namespace denoise


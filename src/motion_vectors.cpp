// Motion vector estimation implementation.

#include "motion_vectors.hpp"

#include <cmath>
#include <cstdlib>
#include <limits>

namespace denoise {
namespace {

double accumulate_cost(const FrameView& reference,
                       const FrameView& target,
                       int ref_x,
                       int ref_y,
                       int target_x,
                       int target_y,
                       int block_size,
                       MatchingCostType cost_type) {
    double cost = 0.0;
    for (int by = 0; by < block_size; ++by) {
        const int ref_row = ref_y + by;
        const int tgt_row = target_y + by;
        for (int bx = 0; bx < block_size; ++bx) {
            const int ref_col = ref_x + bx;
            const int tgt_col = target_x + bx;
            const int diff = static_cast<int>(reference.at(ref_col, ref_row)) -
                             static_cast<int>(target.at(tgt_col, tgt_row));
            if (cost_type == MatchingCostType::SAD) {
                cost += std::abs(diff);
            } else {
                cost += static_cast<double>(diff * diff);
            }
        }
    }
    return cost;
}

}  // namespace

std::vector<BlockMotionVector> compute_motion_vectors(const FrameView& reference,
                                                      const FrameView& target,
                                                      const MotionEstimationParams& params) {
    if (!reference.is_valid() || !target.is_valid()) {
        throw std::invalid_argument("Invalid frame view provided to compute_motion_vectors.");
    }
    if (reference.width != target.width || reference.height != target.height ||
        reference.stride != target.stride) {
        throw std::invalid_argument("Reference and target frames must have identical geometry.");
    }
    if (params.block_size <= 0) {
        throw std::invalid_argument("Block size must be positive.");
    }
    if (params.search_range < 0) {
        throw std::invalid_argument("Search range cannot be negative.");
    }

    const int step = (params.step > 0) ? params.step : params.block_size;
    const int width = reference.width;
    const int height = reference.height;
    const int block_size = params.block_size;
    const double penalty_lambda = params.penalty_lambda;

    if (block_size > width || block_size > height) {
        throw std::invalid_argument("Block size cannot exceed frame dimensions.");
    }

    std::vector<BlockMotionVector> motion_field;
    const int blocks_x = 1 + (width - block_size) / step;
    const int blocks_y = 1 + (height - block_size) / step;
    motion_field.reserve(static_cast<std::size_t>(blocks_x) * static_cast<std::size_t>(blocks_y));

    for (int by = 0; by + block_size <= height; by += step) {
        for (int bx = 0; bx + block_size <= width; bx += step) {
            double best_cost = std::numeric_limits<double>::infinity();
            int best_dx = 0;
            int best_dy = 0;

            for (int dy = -params.search_range; dy <= params.search_range; ++dy) {
                const int candidate_y = by + dy;
                if (candidate_y < 0 || candidate_y + block_size > height) {
                    continue;
                }

                for (int dx = -params.search_range; dx <= params.search_range; ++dx) {
                    const int candidate_x = bx + dx;
                    if (candidate_x < 0 || candidate_x + block_size > width) {
                        continue;
                    }

                    double cost = accumulate_cost(reference,
                                                  target,
                                                  bx,
                                                  by,
                                                  candidate_x,
                                                  candidate_y,
                                                  block_size,
                                                  params.cost_type);

                    if (penalty_lambda > 0.0) {
                        cost += penalty_lambda * static_cast<double>(std::abs(dx) + std::abs(dy));
                    }

                    if (cost < best_cost) {
                        best_cost = cost;
                        best_dx = dx;
                        best_dy = dy;
                    }
                }
            }

            const double normalised_cost =
                best_cost / static_cast<double>(block_size * block_size);

            motion_field.push_back(BlockMotionVector{
                .block_x = bx,
                .block_y = by,
                .dx = best_dx,
                .dy = best_dy,
                .cost = normalised_cost,
            });
        }
    }

    return motion_field;
}

}  // namespace denoise


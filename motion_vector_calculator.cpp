#include "motion_vector_calculator.h"
#include <cstring>

MotionVectorCalculator::MotionVectorCalculator(int block_size, int search_range, int step_size)
    : block_size_(block_size), search_range_(search_range), step_size_(step_size) {
}

std::vector<MotionVector> MotionVectorCalculator::calculateMotionVectors(
    const uint8_t* prev_frame,
    const uint8_t* curr_frame,
    int width,
    int height) {
    
    std::vector<Block> blocks = calculateMotionVectorsWithBlocks(
        prev_frame, curr_frame, width, height);
    
    std::vector<MotionVector> motion_vectors;
    motion_vectors.reserve(blocks.size());
    
    for (const auto& block : blocks) {
        motion_vectors.push_back(block.mv);
    }
    
    return motion_vectors;
}

std::vector<Block> MotionVectorCalculator::calculateMotionVectorsWithBlocks(
    const uint8_t* prev_frame,
    const uint8_t* curr_frame,
    int width,
    int height) {
    
    std::vector<Block> blocks;
    
    // Calculate number of blocks
    int blocks_x = (width + block_size_ - 1) / block_size_;
    int blocks_y = (height + block_size_ - 1) / block_size_;
    
    // Allocate temporary buffer for current block
    int block_pixels = block_size_ * block_size_;
    uint8_t* curr_block = new uint8_t[block_pixels];
    
    // Process each block
    for (int by = 0; by < blocks_y; ++by) {
        for (int bx = 0; bx < blocks_x; ++bx) {
            int block_x = bx * block_size_;
            int block_y = by * block_size_;
            
            // Actual block dimensions (may be smaller at edges)
            int actual_width = std::min(block_size_, width - block_x);
            int actual_height = std::min(block_size_, height - block_y);
            
            // Extract current block
            extractBlock(curr_frame, block_x, block_y, width, height,
                        curr_block, actual_width, actual_height);
            
            MotionVector best_mv;
            float best_cost = std::numeric_limits<float>::max();
            
            // Search in previous frame
            for (int dy = -search_range_; dy <= search_range_; dy += step_size_) {
                for (int dx = -search_range_; dx <= search_range_; dx += step_size_) {
                    int ref_x = block_x + dx;
                    int ref_y = block_y + dy;
                    
                    // Check if reference block is within bounds
                    if (ref_x < 0 || ref_y < 0 ||
                        ref_x + actual_width > width ||
                        ref_y + actual_height > height) {
                        continue;
                    }
                    
                    // Calculate SAD
                    float cost = calculateSAD(
                        prev_frame + ref_y * width + ref_x,
                        curr_block,
                        width,
                        actual_width,
                        actual_height
                    );
                    
                    // Update best match
                    if (cost < best_cost) {
                        best_cost = cost;
                        best_mv = MotionVector(dx, dy, cost);
                    }
                }
            }
            
            Block block;
            block.x = block_x;
            block.y = block_y;
            block.width = actual_width;
            block.height = actual_height;
            block.mv = best_mv;
            
            blocks.push_back(block);
        }
    }
    
    delete[] curr_block;
    return blocks;
}

std::vector<MotionVector> MotionVectorCalculator::refineMotionVectors(
    const uint8_t* prev_frame,
    const uint8_t* curr_frame,
    int width,
    int height,
    const std::vector<Block>& blocks) {
    
    std::vector<MotionVector> refined_vectors;
    refined_vectors.reserve(blocks.size());
    
    int block_pixels = block_size_ * block_size_;
    uint8_t* curr_block = new uint8_t[block_pixels];
    
    for (const auto& block : blocks) {
        // Extract current block
        extractBlock(curr_frame, block.x, block.y, width, height,
                    curr_block, block.width, block.height);
        
        MotionVector best_mv = block.mv;
        float best_cost = block.mv.cost;
        
        // Refine around the initial motion vector with half-pixel accuracy
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                
                int ref_x = block.x + block.mv.dx + dx;
                int ref_y = block.y + block.mv.dy + dy;
                
                // Check bounds
                if (ref_x < 0 || ref_y < 0 ||
                    ref_x + block.width > width ||
                    ref_y + block.height > height) {
                    continue;
                }
                
                float cost = calculateSAD(
                    prev_frame + ref_y * width + ref_x,
                    curr_block,
                    width,
                    block.width,
                    block.height
                );
                
                if (cost < best_cost) {
                    best_cost = cost;
                    best_mv = MotionVector(block.mv.dx + dx, block.mv.dy + dy, cost);
                }
            }
        }
        
        refined_vectors.push_back(best_mv);
    }
    
    delete[] curr_block;
    return refined_vectors;
}

float MotionVectorCalculator::calculateSAD(
    const uint8_t* block1,
    const uint8_t* block2,
    int width,
    int block_width,
    int block_height) const {
    
    float sad = 0.0f;
    
    for (int y = 0; y < block_height; ++y) {
        for (int x = 0; x < block_width; ++x) {
            int diff = static_cast<int>(block1[y * width + x]) -
                      static_cast<int>(block2[y * block_width + x]);
            sad += std::abs(diff);
        }
    }
    
    // Normalize by block size
    return sad / (block_width * block_height);
}

uint8_t MotionVectorCalculator::getPixel(
    const uint8_t* frame,
    int x, int y,
    int width, int height) const {
    
    // Clamp coordinates to valid range
    x = std::max(0, std::min(x, width - 1));
    y = std::max(0, std::min(y, height - 1));
    
    return frame[y * width + x];
}

void MotionVectorCalculator::extractBlock(
    const uint8_t* frame,
    int x, int y,
    int width, int height,
    uint8_t* block,
    int block_width,
    int block_height) const {
    
    for (int by = 0; by < block_height; ++by) {
        for (int bx = 0; bx < block_width; ++bx) {
            int px = x + bx;
            int py = y + by;
            
            uint8_t pixel = getPixel(frame, px, py, width, height);
            block[by * block_width + bx] = pixel;
        }
    }
}

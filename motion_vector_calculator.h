#ifndef MOTION_VECTOR_CALCULATOR_H
#define MOTION_VECTOR_CALCULATOR_H

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>

/**
 * Structure to represent a motion vector
 */
struct MotionVector {
    int dx;  // Horizontal displacement
    int dy;  // Vertical displacement
    float cost;  // Matching cost (lower is better)
    
    MotionVector() : dx(0), dy(0), cost(std::numeric_limits<float>::max()) {}
    MotionVector(int x, int y, float c = 0.0f) : dx(x), dy(y), cost(c) {}
};

/**
 * Structure to represent a block
 */
struct Block {
    int x, y;  // Top-left corner position
    int width, height;  // Block dimensions
    MotionVector mv;  // Associated motion vector
};

/**
 * Class for calculating motion vectors using block matching algorithm
 * Used for video denoising applications
 */
class MotionVectorCalculator {
public:
    /**
     * Constructor
     * @param block_size Size of blocks for matching (typically 8, 16, or 32)
     * @param search_range Maximum search range in pixels
     * @param step_size Step size for search (1 = full search, 2 = half pixel, etc.)
     */
    MotionVectorCalculator(int block_size = 16, int search_range = 16, int step_size = 1);
    
    /**
     * Calculate motion vectors between two frames
     * @param prev_frame Previous frame (grayscale, row-major order)
     * @param curr_frame Current frame (grayscale, row-major order)
     * @param width Frame width
     * @param height Frame height
     * @return Vector of motion vectors, one per block
     */
    std::vector<MotionVector> calculateMotionVectors(
        const uint8_t* prev_frame,
        const uint8_t* curr_frame,
        int width,
        int height
    );
    
    /**
     * Calculate motion vectors and return block information
     * @param prev_frame Previous frame
     * @param curr_frame Current frame
     * @param width Frame width
     * @param height Frame height
     * @return Vector of blocks with associated motion vectors
     */
    std::vector<Block> calculateMotionVectorsWithBlocks(
        const uint8_t* prev_frame,
        const uint8_t* curr_frame,
        int width,
        int height
    );
    
    /**
     * Refine motion vectors using sub-pixel accuracy
     * @param prev_frame Previous frame
     * @param curr_frame Current frame
     * @param width Frame width
     * @param height Frame height
     * @param blocks Blocks with initial motion vectors
     * @return Refined motion vectors with sub-pixel accuracy
     */
    std::vector<MotionVector> refineMotionVectors(
        const uint8_t* prev_frame,
        const uint8_t* curr_frame,
        int width,
        int height,
        const std::vector<Block>& blocks
    );
    
    // Getters and setters
    int getBlockSize() const { return block_size_; }
    void setBlockSize(int size) { block_size_ = size; }
    
    int getSearchRange() const { return search_range_; }
    void setSearchRange(int range) { search_range_ = range; }
    
    int getStepSize() const { return step_size_; }
    void setStepSize(int step) { step_size_ = step; }

private:
    /**
     * Calculate SAD (Sum of Absolute Differences) between two blocks
     */
    float calculateSAD(
        const uint8_t* block1,
        const uint8_t* block2,
        int width,
        int block_width,
        int block_height
    ) const;
    
    /**
     * Get pixel value with boundary checking
     */
    uint8_t getPixel(const uint8_t* frame, int x, int y, int width, int height) const;
    
    /**
     * Extract a block from a frame
     */
    void extractBlock(
        const uint8_t* frame,
        int x, int y,
        int width, int height,
        uint8_t* block,
        int block_width,
        int block_height
    ) const;
    
    int block_size_;
    int search_range_;
    int step_size_;
};

#endif // MOTION_VECTOR_CALCULATOR_H

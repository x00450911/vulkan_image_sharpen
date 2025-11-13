/**
 * Simple test program for motion vector calculation
 * This demonstrates basic usage without OpenCV
 */

#include "motion_vector_calculator.h"
#include <iostream>
#include <vector>
#include <cstring>

int main() {
    std::cout << "Simple Motion Vector Test\n" << std::endl;
    
    // Create simple test frames
    const int width = 320;
    const int height = 240;
    
    std::vector<uint8_t> prev_frame(width * height);
    std::vector<uint8_t> curr_frame(width * height);
    
    // Fill with a simple pattern
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            prev_frame[y * width + x] = (x + y) % 256;
        }
    }
    
    // Shift pattern by 3 pixels to the right
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int src_x = (x - 3 + width) % width;
            curr_frame[y * width + x] = prev_frame[y * width + src_x];
        }
    }
    
    // Create calculator
    MotionVectorCalculator calculator(16, 8, 1);
    
    std::cout << "Calculating motion vectors..." << std::endl;
    std::cout << "Block size: " << calculator.getBlockSize() << std::endl;
    std::cout << "Search range: " << calculator.getSearchRange() << std::endl;
    
    // Calculate motion vectors
    std::vector<MotionVector> mvs = calculator.calculateMotionVectors(
        prev_frame.data(),
        curr_frame.data(),
        width,
        height
    );
    
    std::cout << "\nCalculated " << mvs.size() << " motion vectors" << std::endl;
    
    // Analyze results
    int correct_detections = 0;
    int total_blocks = mvs.size();
    
    for (const auto& mv : mvs) {
        // Check if motion vector is close to expected (dx ≈ 3, dy ≈ 0)
        if (mv.dx >= 2 && mv.dx <= 4 && mv.dy >= -1 && mv.dy <= 1) {
            correct_detections++;
        }
    }
    
    std::cout << "Correct detections: " << correct_detections 
              << " / " << total_blocks 
              << " (" << (100.0 * correct_detections / total_blocks) << "%)" << std::endl;
    
    // Show sample vectors
    std::cout << "\nSample motion vectors:" << std::endl;
    for (size_t i = 0; i < std::min(static_cast<size_t>(5), mvs.size()); ++i) {
        std::cout << "  Block " << i << ": dx=" << mvs[i].dx 
                  << ", dy=" << mvs[i].dy 
                  << ", cost=" << mvs[i].cost << std::endl;
    }
    
    std::cout << "\nTest completed successfully!" << std::endl;
    return 0;
}

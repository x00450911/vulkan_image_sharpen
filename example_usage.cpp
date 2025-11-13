#include "motion_vector_calculator.h"
#include "motion_vector_opencv.h"
#include <iostream>
#include <vector>
#include <chrono>

#ifdef OPENCV_AVAILABLE
#include <opencv2/opencv.hpp>
#endif

/**
 * Example: Calculate motion vectors using block matching
 */
void exampleBlockMatching() {
    std::cout << "=== Block Matching Example ===" << std::endl;
    
    // Create synthetic test frames (simple gradient)
    const int width = 640;
    const int height = 480;
    std::vector<uint8_t> prev_frame(width * height);
    std::vector<uint8_t> curr_frame(width * height);
    
    // Create a simple pattern that shifts
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            prev_frame[y * width + x] = ((x + y) % 256);
            // Shift by 5 pixels to the right
            int shifted_x = (x - 5 + width) % width;
            curr_frame[y * width + shifted_x] = ((x + y) % 256);
        }
    }
    
    // Create calculator
    MotionVectorCalculator calculator(16, 16, 1);
    
    // Calculate motion vectors
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<MotionVector> mvs = calculator.calculateMotionVectors(
        prev_frame.data(),
        curr_frame.data(),
        width,
        height
    );
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Calculated " << mvs.size() << " motion vectors in "
              << duration.count() << " ms" << std::endl;
    
    // Print statistics
    int blocks_x = (width + 15) / 16;
    int blocks_y = (height + 15) / 16;
    
    std::cout << "Block grid: " << blocks_x << " x " << blocks_y << std::endl;
    
    // Show some motion vectors
    std::cout << "\nSample motion vectors:" << std::endl;
    for (size_t i = 0; i < std::min(static_cast<size_t>(10), mvs.size()); ++i) {
        std::cout << "  Block " << i << ": dx=" << mvs[i].dx
                  << ", dy=" << mvs[i].dy
                  << ", cost=" << mvs[i].cost << std::endl;
    }
}

#ifdef OPENCV_AVAILABLE
/**
 * Example: Calculate motion vectors using OpenCV optical flow
 */
void exampleOpenCVFlow() {
    std::cout << "\n=== OpenCV Optical Flow Example ===" << std::endl;
    
    // Create synthetic frames using OpenCV
    cv::Mat prev_frame = cv::Mat::zeros(480, 640, CV_8UC1);
    cv::Mat curr_frame = cv::Mat::zeros(480, 640, CV_8UC1);
    
    // Draw a simple pattern
    cv::circle(prev_frame, cv::Point(320, 240), 50, cv::Scalar(255), -1);
    cv::circle(curr_frame, cv::Point(330, 240), 50, cv::Scalar(255), -1);  // Shifted
    
    // Calculate dense optical flow
    MotionVectorOpenCV flow_calc(MotionVectorOpenCV::FlowMethod::FARNEBACK);
    
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat flow = flow_calc.calculateDenseFlow(prev_frame, curr_frame);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Calculated dense optical flow in " << duration.count() << " ms" << std::endl;
    std::cout << "Flow size: " << flow.cols << " x " << flow.rows << std::endl;
    
    // Convert to block vectors
    std::vector<cv::Point2f> block_vectors = flow_calc.flowToBlockVectors(flow, 16);
    std::cout << "Block vectors: " << block_vectors.size() << std::endl;
    
    // Visualize
    cv::Mat vis = flow_calc.visualizeFlow(curr_frame, flow, 16);
    cv::imwrite("motion_vectors_visualization.png", vis);
    std::cout << "Saved visualization to motion_vectors_visualization.png" << std::endl;
}

/**
 * Example: Sparse optical flow with feature detection
 */
void exampleSparseFlow() {
    std::cout << "\n=== Sparse Optical Flow Example ===" << std::endl;
    
    cv::Mat prev_frame = cv::Mat::zeros(480, 640, CV_8UC1);
    cv::Mat curr_frame = cv::Mat::zeros(480, 640, CV_8UC1);
    
    // Create pattern with corners
    cv::rectangle(prev_frame, cv::Rect(100, 100, 200, 200), cv::Scalar(255), -1);
    cv::rectangle(curr_frame, cv::Rect(110, 100, 200, 200), cv::Scalar(255), -1);
    
    // Detect features
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(prev_frame, corners, 100, 0.01, 10);
    
    std::cout << "Detected " << corners.size() << " feature points" << std::endl;
    
    // Track features
    MotionVectorOpenCV flow_calc;
    std::vector<cv::Point2f> points = corners;
    std::vector<cv::Point2f> mvs = flow_calc.calculateSparseFlow(
        prev_frame, curr_frame, points);
    
    std::cout << "Tracked " << mvs.size() << " points" << std::endl;
    
    // Show some motion vectors
    for (size_t i = 0; i < std::min(static_cast<size_t>(5), mvs.size()); ++i) {
        std::cout << "  Point " << i << ": (" << corners[i].x << ", " << corners[i].y
                  << ") -> (" << mvs[i].x << ", " << mvs[i].y << ")" << std::endl;
    }
}
#endif

int main() {
    std::cout << "Motion Vector Calculation for Denoising - Examples\n" << std::endl;
    
    // Block matching example (no dependencies)
    exampleBlockMatching();
    
#ifdef OPENCV_AVAILABLE
    // OpenCV examples
    exampleOpenCVFlow();
    exampleSparseFlow();
#else
    std::cout << "\nOpenCV examples skipped (OpenCV not available)" << std::endl;
    std::cout << "To enable OpenCV examples, compile with -DOPENCV_AVAILABLE" << std::endl;
#endif
    
    return 0;
}

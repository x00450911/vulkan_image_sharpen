# Motion Vector Calculation for Denoising

A C++ library for calculating motion vectors between video frames, designed for video denoising applications. The library provides both a lightweight block matching implementation and an OpenCV-based optical flow implementation.

## Features

- **Block Matching Algorithm**: Fast, lightweight motion vector calculation using Sum of Absolute Differences (SAD)
- **OpenCV Optical Flow**: Support for multiple optical flow methods (Farneback, Lucas-Kanade, Dual TV-L1, DIS)
- **Sub-pixel Refinement**: Optional refinement for improved accuracy
- **Flexible Configuration**: Adjustable block sizes, search ranges, and step sizes
- **No External Dependencies**: Block matching works standalone (OpenCV is optional)

## Building

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.10 or higher
- OpenCV 3.4+ (optional, for advanced optical flow)

### Build Instructions

```bash
# Create build directory
mkdir build
cd build

# Configure (with OpenCV)
cmake .. -DENABLE_OPENCV=ON

# Or without OpenCV
cmake .. -DENABLE_OPENCV=OFF

# Build
cmake --build .

# Run example
./example_usage
```

### Installing OpenCV (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install libopencv-dev
```

## Usage

### Block Matching (Standalone)

```cpp
#include "motion_vector_calculator.h"

// Create calculator
MotionVectorCalculator calculator(16, 16, 1);  // block_size, search_range, step_size

// Calculate motion vectors
std::vector<MotionVector> mvs = calculator.calculateMotionVectors(
    prev_frame_data,  // uint8_t* previous frame (grayscale)
    curr_frame_data,  // uint8_t* current frame (grayscale)
    width,            // int frame width
    height            // int frame height
);

// Access motion vectors
for (const auto& mv : mvs) {
    int dx = mv.dx;      // horizontal displacement
    int dy = mv.dy;      // vertical displacement
    float cost = mv.cost; // matching cost
}
```

### OpenCV Optical Flow

```cpp
#include "motion_vector_opencv.h"
#include <opencv2/opencv.hpp>

// Create calculator
MotionVectorOpenCV flow_calc(MotionVectorOpenCV::FlowMethod::FARNEBACK);

// Calculate dense optical flow
cv::Mat flow = flow_calc.calculateDenseFlow(prev_frame, curr_frame);

// Convert to block-based vectors
std::vector<cv::Point2f> block_vectors = flow_calc.flowToBlockVectors(flow, 16);

// Visualize
cv::Mat visualization = flow_calc.visualizeFlow(curr_frame, flow, 16);
cv::imwrite("motion_vectors.png", visualization);
```

## API Reference

### MotionVectorCalculator

#### Constructor
```cpp
MotionVectorCalculator(int block_size = 16, int search_range = 16, int step_size = 1)
```

- `block_size`: Size of blocks for matching (typically 8, 16, or 32 pixels)
- `search_range`: Maximum search range in pixels (typically 8-32)
- `step_size`: Step size for search (1 = full search, 2 = half pixel, etc.)

#### Methods

- `calculateMotionVectors()`: Calculate motion vectors for all blocks
- `calculateMotionVectorsWithBlocks()`: Calculate motion vectors with block information
- `refineMotionVectors()`: Refine motion vectors with sub-pixel accuracy

### MotionVectorOpenCV

#### Flow Methods

- `FARNEBACK`: Dense optical flow using Farneback algorithm (fast, good quality)
- `LUCAS_KANADE`: Sparse optical flow at feature points (very fast)
- `DUAL_TVL1`: Dense optical flow using Dual TV-L1 (high quality, slower)
- `DIS`: Dense inverse search (balanced quality/speed)

#### Methods

- `calculateDenseFlow()`: Calculate dense motion vectors for all pixels
- `calculateSparseFlow()`: Calculate sparse motion vectors at feature points
- `flowToBlockVectors()`: Convert dense flow to block-based representation
- `visualizeFlow()`: Create visualization of motion vectors

## Applications in Denoising

Motion vectors are essential for temporal denoising algorithms:

1. **Temporal Filtering**: Align frames using motion vectors before averaging
2. **Motion-Compensated Filtering**: Apply filters along motion trajectories
3. **Outlier Detection**: Identify inconsistent motion for artifact reduction
4. **Multi-frame Denoising**: Combine information from multiple frames

### Example Denoising Workflow

```cpp
// 1. Calculate motion vectors
MotionVectorCalculator calc(16, 16, 1);
auto mvs = calc.calculateMotionVectors(prev_frame, curr_frame, w, h);

// 2. Warp previous frame using motion vectors
// (implementation depends on your denoising algorithm)

// 3. Apply temporal filter
// (average or median filter along motion trajectories)
```

## Performance Considerations

- **Block Size**: Smaller blocks (8x8) provide more detail but are slower. Larger blocks (32x32) are faster but less accurate.
- **Search Range**: Larger ranges handle faster motion but increase computation time quadratically.
- **Step Size**: Step size of 1 provides best quality. Step size of 2 or 4 speeds up computation with minimal quality loss.
- **OpenCV Methods**: Farneback is fastest for dense flow. Dual TV-L1 provides best quality but is slower.

## License

This code is provided as-is for educational and research purposes.

## References

- Block Matching: Used in video codecs (H.264, H.265)
- Farneback Optical Flow: "Two-Frame Motion Estimation Based on Polynomial Expansion"
- Dual TV-L1: "Dual TV-L1 Optical Flow Estimation"

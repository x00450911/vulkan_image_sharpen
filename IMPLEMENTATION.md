# Implementation Details

## Architecture Overview

This Vulkan-based multi-frame video denoiser implements a temporal-spatial denoising algorithm using GPU compute shaders. The implementation consists of several key components:

### Core Components

1. **VulkanDenoiser Class** (`vulkan_denoiser.h/cpp`)
   - Manages Vulkan instance, device, and resources
   - Handles frame buffer management
   - Coordinates compute shader execution
   - Provides API for processing frames

2. **Compute Shader** (`denoise.comp`)
   - Implements temporal denoising algorithm
   - Performs spatial bilateral filtering
   - Runs on GPU for maximum performance

3. **Main Application** (`main.cpp`)
   - Example usage of the denoiser
   - Generates test frames with noise
   - Demonstrates frame processing workflow

4. **Video I/O** (`video_io.h/cpp`)
   - Interface for video file reading/writing
   - PPM image format support
   - Extensible for other formats (FFmpeg, OpenCV)

## Algorithm Details

### Temporal Denoising

The temporal denoising algorithm accumulates information from multiple frames:

1. **Frame Accumulation**:
   - Current frame is given weight 1.0
   - Previous frames receive decreasing weights (0.8, 0.64, 0.512, ...)
   - Weights are multiplied by motion factor

2. **Motion Detection**:
   - Computes luminance difference between current and previous frames
   - Uses smoothstep function for soft thresholding
   - Motion factor ranges from 0.0 (high motion) to 1.0 (static)

3. **Color Similarity**:
   - Additional weight based on color distance
   - Prevents ghosting artifacts in moving regions
   - Uses squared color distance metric

4. **Final Blending**:
   - Accumulated color divided by total weight
   - Blended with original based on denoising strength parameter

### Spatial Filtering

A 3x3 bilateral filter is applied after temporal denoising:

1. **Bilateral Filtering**:
   - Spatial weight: Gaussian based on pixel distance
   - Color weight: Gaussian based on color difference
   - Preserves edges while reducing noise

2. **Parameters**:
   - Spatial sigma: 1.5 pixels
   - Color sigma: 0.1 (normalized)
   - Applied with reduced strength (50% of denoising strength)

## Memory Management

### Frame Buffers

Each frame buffer contains:
- **Input Image**: Stores the current frame (read-only in shader)
- **Output Image**: Stores the denoised result (write-only in shader)
- **Staging Buffer**: For CPU-GPU data transfers

### Memory Layout

```
FrameBuffer[0..N-1]:
  - Input Image (Device Local)
  - Output Image (Device Local)
  - Staging Buffer (Host Visible)
```

### Resource Lifecycle

1. **Initialization**:
   - All frame buffers created at startup
   - Images allocated in device-local memory
   - Staging buffers in host-visible memory

2. **Frame Processing**:
   - Input data copied to staging buffer
   - Staging buffer copied to input image
   - Compute shader processes frame
   - Output image copied to staging buffer
   - Result read back to CPU

3. **Cleanup**:
   - All resources destroyed in reverse order
   - Device idle before cleanup

## Vulkan Pipeline

### Descriptor Set Layout

```
Binding 0: Current frame (storage image, read-only)
Binding 1: Previous frames (storage image array, read-only)
Binding 2: Output frame (storage image, write-only)
Binding 3: Uniform buffer (denoising parameters)
```

### Compute Shader Execution

1. **Workgroup Size**: 16x16 threads (256 pixels per workgroup)
2. **Dispatch**: 
   - X: (width + 15) / 16 workgroups
   - Y: (height + 15) / 16 workgroups
   - Z: 1

3. **Synchronization**:
   - Fences for frame completion
   - Semaphores for queue synchronization
   - Image layout transitions for proper access

## Performance Considerations

### Optimization Opportunities

1. **Memory Transfers**:
   - Current: Synchronous CPU-GPU transfers
   - Optimization: Use async transfers with multiple staging buffers
   - Benefit: Overlap computation and transfer

2. **Uniform Buffer**:
   - Current: Created per frame
   - Optimization: Pre-allocate and reuse
   - Benefit: Reduce allocation overhead

3. **Image Layout Transitions**:
   - Current: Multiple transitions per frame
   - Optimization: Batch transitions
   - Benefit: Reduce pipeline stalls

4. **Command Buffer Recording**:
   - Current: Recorded per frame
   - Optimization: Pre-record and reuse
   - Benefit: Reduce CPU overhead

### Expected Performance

On modern GPUs (RTX 3060, RX 6600):
- 1080p: 30-60 FPS
- 4K: 10-20 FPS

Bottlenecks:
- Memory bandwidth (CPU-GPU transfers)
- Compute shader execution
- Image layout transitions

## Extending the Implementation

### Adding Real Video Support

1. **FFmpeg Integration**:
```cpp
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

// Decode frame to RGBA
// Pass to denoiser
// Encode result
```

2. **OpenCV Integration**:
```cpp
#include <opencv2/opencv.hpp>

cv::VideoCapture cap("input.mp4");
cv::Mat frame;
// Convert to RGBA
// Process with denoiser
cv::VideoWriter writer("output.mp4", ...);
```

### Custom Denoising Algorithms

Modify `denoise.comp` to implement:

1. **Non-Local Means**:
   - Search for similar patches
   - Weighted average based on patch similarity

2. **BM3D (Block-Matching 3D)**:
   - Group similar blocks
   - 3D transform and thresholding
   - Inverse transform and aggregation

3. **Deep Learning**:
   - Load pre-trained model weights
   - Implement neural network in compute shader
   - Or use Vulkan ML extensions

### Motion Estimation

Add optical flow for better motion detection:

1. **Lucas-Kanade**:
   - Compute optical flow vectors
   - Use for motion compensation

2. **Block Matching**:
   - Search for best match in previous frame
   - Use motion vectors for alignment

## Error Handling

The implementation includes basic error handling:

1. **Vulkan Errors**:
   - Check return codes
   - Print error messages
   - Cleanup on failure

2. **Validation Layers**:
   - Enabled in debug builds
   - Catches common Vulkan errors
   - Provides detailed diagnostics

3. **Resource Management**:
   - RAII-style cleanup
   - Null handle checks
   - Proper resource destruction order

## Known Limitations

1. **Fixed Frame Count**:
   - Number of frames must be set at initialization
   - Cannot change during runtime

2. **Synchronous Processing**:
   - Each frame waits for previous to complete
   - No pipelining of multiple frames

3. **No Motion Compensation**:
   - Simple motion detection only
   - No pixel-level alignment

4. **Limited Format Support**:
   - Only RGBA8 format
   - No HDR or other formats

5. **CPU-GPU Transfer Overhead**:
   - Synchronous transfers
   - No async processing

## Future Improvements

1. **Async Processing**:
   - Multiple frames in flight
   - Overlap transfers and computation

2. **Motion Compensation**:
   - Optical flow computation
   - Pixel-level alignment

3. **Adaptive Parameters**:
   - Scene-based parameter adjustment
   - Automatic strength tuning

4. **Multi-Resolution**:
   - Pyramid-based processing
   - Coarse-to-fine denoising

5. **Format Support**:
   - HDR formats (FP16, FP32)
   - YUV color spaces
   - Different bit depths

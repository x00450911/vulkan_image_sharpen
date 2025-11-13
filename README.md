# Vulkan Multi-Frame Video Denoising

A high-performance GPU-accelerated video denoising implementation using Vulkan compute shaders. This project implements temporal denoising by combining multiple video frames to reduce noise while preserving detail.

## Features

- **GPU-Accelerated**: Uses Vulkan compute shaders for maximum performance
- **Temporal Denoising**: Combines information from multiple frames for better noise reduction
- **Adaptive Filtering**: Automatically adjusts blending based on motion and detail
- **Bilateral Filtering**: Spatial denoising to reduce noise within individual frames
- **Real-time Processing**: Efficient frame buffer management for video processing

## Algorithm

The denoising algorithm uses a two-stage approach:

1. **Spatial Filtering**: Applies a bilateral filter to each frame to reduce noise while preserving edges
2. **Temporal Blending**: Uses exponential moving average to combine current and previous frames

The temporal blending factor (`alpha`) controls the balance:
- **Low alpha (0.1-0.3)**: More temporal smoothing, better noise reduction but more motion blur
- **High alpha (0.5-0.9)**: Less temporal smoothing, preserves motion but retains more noise

Adaptive filtering detects high-variance regions (likely motion or detail) and reduces temporal smoothing to preserve these areas.

## Requirements

- **Vulkan SDK** (1.2 or later)
- **CMake** (3.15 or later)
- **C++17** compatible compiler
- **OpenCV** (for video I/O)
- **GLFW** (optional, for windowing if needed)
- **glslc** (Vulkan shader compiler, included with Vulkan SDK)

### Installing Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install vulkan-sdk libvulkan-dev libopencv-dev cmake build-essential
```

#### macOS
```bash
brew install vulkan-headers vulkan-loader opencv cmake
```

#### Windows
- Install [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
- Install [OpenCV](https://opencv.org/releases/)
- Install [CMake](https://cmake.org/download/)

## Building

```bash
mkdir build
cd build
cmake ..
make
```

The compiled shaders will be placed in `build/shaders/` and copied to the build directory.

## Usage

```bash
./VulkanMultiFrameDenoising <input_video> <output_video> [alpha]
```

### Parameters

- `input_video`: Path to input video file (supports formats supported by OpenCV)
- `output_video`: Path to output video file
- `alpha`: (Optional) Temporal blending factor (0.0-1.0, default: 0.2)
  - Lower values = more temporal smoothing (less noise, more motion blur)
  - Higher values = less temporal smoothing (more noise, less motion blur)

### Examples

```bash
# Basic usage with default alpha (0.2)
./VulkanMultiFrameDenoising input.mp4 output.mp4

# More aggressive denoising (lower alpha)
./VulkanMultiFrameDenoising input.mp4 output.mp4 0.1

# Preserve more motion detail (higher alpha)
./VulkanMultiFrameDenoising input.mp4 output.mp4 0.5
```

## Project Structure

```
.
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── include/                # Header files
│   ├── vulkan_context.h   # Vulkan initialization and resource management
│   ├── frame_manager.h    # Frame buffer management
│   └── denoiser.h         # Denoising pipeline
├── src/                    # Source files
│   ├── main.cpp           # Main application
│   ├── vulkan_context.cpp # Vulkan implementation
│   ├── frame_manager.cpp  # Frame management implementation
│   └── denoiser.cpp       # Denoising implementation
└── shaders/                # Compute shaders
    ├── denoise.comp       # Main denoising shader
    └── accumulate.comp    # Multi-frame accumulation shader
```

## Technical Details

### Compute Shaders

#### denoise.comp
- Implements bilateral filtering for spatial denoising
- Performs temporal blending using exponential moving average
- Adaptive filtering based on variance detection
- Workgroup size: 8x8 threads

#### accumulate.comp
- Combines multiple frames with weighted averaging
- Supports up to 4 frames
- Useful for more advanced multi-frame techniques

### Memory Management

- Frames are stored in GPU memory as `VkImage` objects
- Staging buffers handle CPU-GPU transfers
- Frame buffers are reused in a circular buffer pattern
- Images use `VK_IMAGE_LAYOUT_GENERAL` for compute shader access

### Performance Considerations

- Frame upload/download is synchronous (can be optimized with async transfers)
- Command buffers are allocated per operation (can be pre-allocated)
- Consider using multiple command buffers and semaphores for pipelining
- For very high-resolution videos, consider tiling the processing

## Limitations

- Currently processes frames sequentially (no pipelining)
- Fixed bilateral filter radius (can be made configurable)
- Supports RGBA8 format only
- Frame buffer count is fixed at initialization

## Future Improvements

- [ ] Asynchronous frame transfers
- [ ] Configurable filter parameters
- [ ] Support for different pixel formats
- [ ] Motion estimation for better temporal blending
- [ ] Multi-pass denoising
- [ ] Real-time preview mode

## License

This code is provided as-is for educational and research purposes.

## References

- [Vulkan Specification](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html)
- [Vulkan Tutorial](https://vulkan-tutorial.com/)
- Bilateral Filtering: Tomasi, C., & Manduchi, R. (1998). Bilateral filtering for gray and color images.

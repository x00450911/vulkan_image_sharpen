# Vulkan Multi-Frame Video Denoiser

A high-performance GPU-accelerated video denoising implementation using Vulkan compute shaders. This implementation performs temporal denoising by accumulating information across multiple frames, reducing noise while preserving detail and motion.

## Features

- **Temporal Denoising**: Uses multiple frames (configurable, default 5) to reduce noise
- **Motion-Aware Blending**: Automatically detects motion and adjusts temporal weights
- **Spatial Filtering**: Additional bilateral filtering for remaining noise
- **GPU Accelerated**: Fully implemented using Vulkan compute shaders
- **Configurable Parameters**: Adjustable denoising strength and temporal weights

## Algorithm Overview

The denoising algorithm combines two techniques:

1. **Temporal Accumulation**: 
   - Accumulates color information from previous frames
   - Uses motion detection based on luminance differences
   - Applies decreasing weights for older frames
   - Blends based on color similarity

2. **Spatial Bilateral Filtering**:
   - Applies a 3x3 bilateral filter to reduce remaining noise
   - Preserves edges using color distance weighting
   - Only applied lightly to avoid over-smoothing

## Requirements

### Build Dependencies
- CMake 3.10 or higher
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- Vulkan SDK (1.0 or higher)
- glslangValidator (for shader compilation)

### Runtime Dependencies
- Vulkan-capable GPU and drivers
- Linux: Mesa drivers or proprietary drivers (NVIDIA/AMD)
- Windows: Latest GPU drivers
- macOS: MoltenVK (Vulkan on Metal)

### Installing Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake vulkan-tools libvulkan-dev glslang-tools
```

**Fedora:**
```bash
sudo dnf install gcc-c++ cmake vulkan-tools vulkan-loader-devel glslang
```

**macOS:**
```bash
brew install cmake vulkan-headers vulkan-loader glslang
```

**Windows:**
- Install [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
- Install CMake from [cmake.org](https://cmake.org/download/)

## Building

1. Clone or download this repository

2. Compile the compute shader:
```bash
chmod +x compile_shader.sh
./compile_shader.sh
```

Or manually:
```bash
glslangValidator -V denoise.comp -o denoise.comp.spv
```

3. Build the project:
```bash
mkdir build
cd build
cmake ..
make
```

On Windows with Visual Studio:
```cmd
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019"
cmake --build . --config Release
```

## Usage

### Basic Usage

Run the denoiser with default settings (generates test frames):
```bash
./vulkan_denoiser
```

This will:
- Generate 30 test frames with noise
- Process them through the denoiser
- Save sample input/output frames as PPM images

### Programmatic Usage

```cpp
#include "vulkan_denoiser.h"

// Initialize denoiser
VulkanDenoiser denoiser(width, height, numFrames);
denoiser.Initialize();

// Set parameters
denoiser.SetDenoisingStrength(0.7f);  // 0.0 = no denoising, 1.0 = maximum
denoiser.SetTemporalWeight(0.8f);     // Weight for temporal accumulation

// Process frames
for (uint32_t i = 0; i < numFrames; i++) {
    uint8_t* inputData;   // RGBA8 format, width * height * 4 bytes
    uint8_t* outputData;  // RGBA8 format, width * height * 4 bytes
    
    // Load your frame data into inputData...
    
    denoiser.ProcessFrame(inputData, outputData, i);
    
    // Use outputData...
}

// Cleanup
denoiser.Cleanup();
```

## Parameters

### Denoising Strength
- Range: 0.0 to 1.0
- Default: 0.7
- Controls how much denoising is applied
- Lower values preserve more detail but reduce noise less
- Higher values reduce more noise but may over-smooth

### Temporal Weight
- Range: 0.0 to 1.0
- Default: 0.8
- Controls how much previous frames contribute
- Higher values use more temporal information
- Lower values rely more on the current frame

### Number of Frames
- Default: 5
- Number of frames to use for temporal accumulation
- More frames = better denoising but higher memory usage
- Recommended: 3-7 frames

## Performance

Performance depends on:
- GPU compute capability
- Frame resolution
- Number of frames used
- Denoising parameters

Typical performance on modern GPUs:
- 1080p (1920x1080): ~30-60 FPS
- 4K (3840x2160): ~10-20 FPS

## File Structure

```
.
├── vulkan_denoiser.h          # Main denoiser class header
├── vulkan_denoiser.cpp        # Vulkan implementation
├── denoise.comp               # GLSL compute shader source
├── denoise.comp.spv           # Compiled SPIR-V shader (generated)
├── main.cpp                   # Example application
├── video_io.h                 # Video I/O interface
├── video_io.cpp               # Video I/O implementation
├── CMakeLists.txt             # Build configuration
├── compile_shader.sh          # Shader compilation script
└── README.md                  # This file
```

## Shader Details

The compute shader (`denoise.comp`) implements:

1. **Temporal Denoising**:
   - Loads current frame and previous frames
   - Computes motion factor based on luminance difference
   - Accumulates weighted colors from previous frames
   - Blends with current frame based on denoising strength

2. **Spatial Filtering**:
   - Applies 3x3 bilateral filter
   - Uses spatial and color distance for weighting
   - Preserves edges while reducing noise

3. **Workgroup Size**: 16x16 threads per workgroup
   - Processes 256 pixels per workgroup
   - Automatically dispatches across entire image

## Extending the Implementation

### Adding Real Video Support

To process real video files, you can integrate FFmpeg or OpenCV:

```cpp
// Example with OpenCV
#include <opencv2/opencv.hpp>

cv::VideoCapture cap("input.mp4");
cv::Mat frame;
std::vector<uint8_t> frameData;

while (cap.read(frame)) {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGBA);
    frameData.assign(frame.data, frame.data + frame.total() * 4);
    
    std::vector<uint8_t> outputData(frameData.size());
    denoiser.ProcessFrame(frameData.data(), outputData.data(), frameIndex++);
    
    // Save or display output...
}
```

### Custom Denoising Algorithms

Modify `denoise.comp` to implement different algorithms:
- Non-local means
- BM3D (Block-Matching 3D)
- Deep learning-based denoising
- Custom temporal filters

## Troubleshooting

### "Failed to find GPUs with Vulkan support"
- Ensure Vulkan drivers are installed
- Check with `vulkaninfo` command
- Verify GPU supports Vulkan

### "Failed to load compute shader"
- Ensure `denoise.comp.spv` exists in the working directory
- Recompile shader: `./compile_shader.sh`
- Check shader compilation errors

### Poor Performance
- Reduce number of frames used
- Lower resolution
- Check GPU utilization with profiling tools
- Ensure using dedicated GPU (not integrated)

### Validation Layer Errors
- These are warnings/errors from Vulkan validation layers
- Can be disabled by building in Release mode
- Check error messages for specific issues

## License

This code is provided as-is for educational and research purposes.

## References

- [Vulkan Specification](https://www.khronos.org/vulkan/)
- [GLSL Compute Shaders](https://www.khronos.org/opengl/wiki/Compute_Shader)
- [Temporal Denoising Techniques](https://en.wikipedia.org/wiki/Video_denoising)

## Contributing

Contributions are welcome! Areas for improvement:
- Real video file support (FFmpeg integration)
- Additional denoising algorithms
- Performance optimizations
- Better motion estimation
- Adaptive parameter tuning

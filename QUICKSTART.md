# Quick Start Guide

Get up and running with Vulkan Video Denoiser in 5 minutes!

## Prerequisites

**Required:**
- Vulkan-capable GPU (NVIDIA, AMD, Intel, or Apple Silicon)
- Vulkan SDK installed ([Download here](https://vulkan.lunarg.com/))
- C++17 compiler (GCC 7+, Clang 5+, MSVC 2019+)
- CMake 3.15+

**Verify Installation:**
```bash
# Check Vulkan is installed
vulkaninfo --summary

# Check shader compiler is available
glslc --version

# Check CMake version
cmake --version
```

## Build in 3 Commands

### Linux/macOS
```bash
./build.sh
cd build
./denoiser_example
```

### Windows
```cmd
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019"
cmake --build . --config Release
Release\denoiser_example.exe
```

## Your First Program

Create `my_denoiser.cpp`:

```cpp
#include "vulkan_video_denoiser.hpp"
#include <vector>
#include <iostream>

int main() {
    // 1. Create denoiser
    vk_denoise::VulkanVideoDenoiser denoiser;
    
    // 2. Configure parameters
    vk_denoise::DenoiseParams params;
    params.temporalWeight = 0.7f;    // Temporal strength
    params.spatialSigma = 1.5f;      // Spatial smoothing
    params.colorSigma = 0.15f;       // Edge preservation
    params.frameCount = 3;           // Use 3 previous frames
    params.kernelSize = 5;           // 5x5 spatial kernel
    
    // 3. Initialize (1920x1080 resolution)
    if (!denoiser.initialize(1920, 1080, params)) {
        std::cerr << "Failed to initialize denoiser\n";
        return 1;
    }
    
    // 4. Prepare input/output buffers
    size_t frameSize = 1920 * 1080 * 4; // RGBA8
    std::vector<uint8_t> inputFrame(frameSize);
    std::vector<uint8_t> outputFrame(frameSize);
    
    // TODO: Load your frame data into inputFrame
    // For now, fill with test data
    for (size_t i = 0; i < frameSize; i++) {
        inputFrame[i] = rand() % 256;
    }
    
    // 5. Process frame
    if (denoiser.processFrame(inputFrame.data(), outputFrame.data())) {
        std::cout << "Frame denoised successfully!\n";
        // TODO: Save or display outputFrame
    }
    
    // 6. Cleanup (automatic in destructor)
    return 0;
}
```

Compile:
```bash
g++ my_denoiser.cpp -o my_denoiser \
    -I./include \
    -L./build \
    -lvulkan_video_denoiser \
    -lvulkan \
    -std=c++17
    
./my_denoiser
```

## Common Use Cases

### Video File Processing

```cpp
#include "vulkan_video_denoiser.hpp"
// + your video library (FFmpeg, OpenCV, etc.)

vk_denoise::VulkanVideoDenoiser denoiser;
vk_denoise::DenoiseParams params;
params.temporalWeight = 0.7f;
denoiser.initialize(width, height, params);

// Process each frame
while (video.hasMoreFrames()) {
    auto frame = video.readFrame(); // RGBA8 data
    denoiser.processFrame(frame.data(), outputBuffer);
    video.writeFrame(outputBuffer);
}
```

### Real-time Camera Stream

```cpp
Camera camera;
camera.start();

vk_denoise::VulkanVideoDenoiser denoiser;
denoiser.initialize(camera.width(), camera.height(), params);

while (running) {
    auto frame = camera.captureFrame();
    denoiser.processFrame(frame.data(), displayBuffer);
    display(displayBuffer);
}
```

### Batch Image Processing

```cpp
std::vector<std::string> images = {"img1.png", "img2.png", "img3.png"};

vk_denoise::VulkanVideoDenoiser denoiser;
denoiser.initialize(1920, 1080, params);

for (const auto& imgPath : images) {
    auto img = loadImage(imgPath); // Load as RGBA8
    denoiser.processFrame(img.data(), outputBuffer);
    saveImage(outputBuffer, "denoised_" + imgPath);
}
```

## Parameter Quick Reference

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| temporalWeight | 0.0-1.0 | 0.7 | Higher = more temporal smoothing |
| spatialSigma | 0.5-3.0 | 1.5 | Higher = more spatial smoothing |
| colorSigma | 0.05-0.3 | 0.15 | Lower = stronger edge preservation |
| frameCount | 1-4 | 4 | More frames = better quality, more memory |
| kernelSize | 3/5/7 | 5 | Larger = smoother, slower |

## Troubleshooting

**Problem: "Vulkan not found"**
```bash
# Set environment variable
export VULKAN_SDK=/path/to/vulkan/sdk
export PATH=$VULKAN_SDK/bin:$PATH
```

**Problem: "Failed to initialize denoiser"**
- Check GPU supports Vulkan 1.2: `vulkaninfo | grep apiVersion`
- Update graphics drivers
- Ensure sufficient VRAM for your resolution

**Problem: Image too blurry**
```cpp
// Reduce temporal weight
params.temporalWeight = 0.4f;
// Increase adaptive strength
params.adaptiveStrength = 0.7f;
```

**Problem: Still too noisy**
```cpp
// Increase temporal weight
params.temporalWeight = 0.9f;
// Use more frames
params.frameCount = 4;
// Larger kernel
params.kernelSize = 7;
```

## Performance Tips

1. **Reduce resolution** for real-time on slower GPUs
2. **Use fewer frames** (frameCount=2 or 3)
3. **Smaller kernel** (kernelSize=3)
4. **Disable motion compensation** if not needed
5. **Process multiple streams** with multiple denoiser instances

## Example Output

```
Vulkan video denoiser initialized successfully
Resolution: 1920x1080
Frame history: 4 frames
Selected GPU: NVIDIA GeForce RTX 3080

Processing...
Frame 1: 5.2ms
Frame 2: 5.1ms
Frame 3: 5.3ms
Frame 4: 5.0ms
Frame 5: 5.1ms

Average: 5.14ms per frame
Throughput: 194.6 FPS
```

## Next Steps

1. ✅ Built and ran example
2. 📖 Read [README.md](README.md) for full documentation
3. 🎯 Try different parameter presets
4. 🔧 Integrate with your video pipeline
5. 🚀 Optimize for your specific use case

## Sample Presets

**Documentary/Interview (Low Noise):**
```cpp
params.temporalWeight = 0.5f;
params.spatialSigma = 1.0f;
params.colorSigma = 0.08f;
params.frameCount = 2;
params.kernelSize = 3;
```

**Action/Sports (High Motion):**
```cpp
params.temporalWeight = 0.4f;
params.spatialSigma = 1.2f;
params.frameCount = 2;
params.kernelSize = 3;
params.useMotionCompensation = true;
```

**Night/Low Light (High Noise):**
```cpp
params.temporalWeight = 0.9f;
params.spatialSigma = 2.0f;
params.colorSigma = 0.2f;
params.frameCount = 4;
params.kernelSize = 7;
```

**Archive Restoration:**
```cpp
params.temporalWeight = 0.85f;
params.spatialSigma = 1.8f;
params.frameCount = 4;
params.adaptiveStrength = 0.3f;
```

## Getting Help

- 📚 Full docs: [README.md](README.md)
- 🏗️ Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- 🔨 Build issues: [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)
- 💡 Examples: [examples/main.cpp](examples/main.cpp)

Happy denoising! 🎬✨

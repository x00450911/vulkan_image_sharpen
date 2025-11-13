# Quick Start Guide

## Prerequisites Check

Before building, ensure you have:

1. **Vulkan SDK** installed
   ```bash
   # Check installation
   vulkaninfo
   ```

2. **glslangValidator** for shader compilation
   ```bash
   # Check installation
   glslangValidator --version
   ```

3. **CMake** and a C++17 compiler
   ```bash
   cmake --version
   g++ --version  # or clang++ --version
   ```

## Build Steps

### Step 1: Compile the Shader

```bash
./compile_shader.sh
```

This creates `denoise.comp.spv` - the compiled compute shader.

### Step 2: Build the Project

```bash
mkdir build
cd build
cmake ..
make
```

On Windows:
```cmd
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019"
cmake --build . --config Release
```

### Step 3: Run

```bash
cd build
./vulkan_denoiser
```

This will:
- Generate 30 test frames with noise
- Process them through the denoiser
- Save sample frames as PPM images

## Using in Your Code

### Minimal Example

```cpp
#include "vulkan_denoiser.h"

int main() {
    // Initialize
    VulkanDenoiser denoiser(1920, 1080, 5);
    denoiser.Initialize();
    
    // Set parameters
    denoiser.SetDenoisingStrength(0.7f);
    denoiser.SetTemporalWeight(0.8f);
    
    // Process frames
    uint8_t* inputData;   // RGBA8, 1920*1080*4 bytes
    uint8_t* outputData;  // RGBA8, 1920*1080*4 bytes
    
    for (uint32_t i = 0; i < numFrames; i++) {
        // Load your frame into inputData...
        denoiser.ProcessFrame(inputData, outputData, i);
        // Use outputData...
    }
    
    // Cleanup
    denoiser.Cleanup();
    return 0;
}
```

## Common Issues

### "Failed to find GPUs with Vulkan support"

**Solution:**
- Install Vulkan drivers for your GPU
- Check with `vulkaninfo`
- On Linux, may need: `sudo apt-get install mesa-vulkan-drivers`

### "Failed to load compute shader"

**Solution:**
- Ensure `denoise.comp.spv` exists in working directory
- Recompile: `./compile_shader.sh`
- Check shader compilation errors

### "Validation layer errors"

**Solution:**
- These are warnings, not fatal
- Build in Release mode to disable: `cmake -DCMAKE_BUILD_TYPE=Release ..`
- Or fix the reported issues

### Poor Performance

**Solution:**
- Reduce number of frames: `VulkanDenoiser(width, height, 3)` instead of 5
- Lower resolution for testing
- Check GPU is being used (not CPU fallback)
- Ensure using dedicated GPU (not integrated)

## Next Steps

1. **Read the README.md** for detailed documentation
2. **Check IMPLEMENTATION.md** for algorithm details
3. **See example_usage.cpp** for more examples
4. **Integrate with your video pipeline** (FFmpeg, OpenCV, etc.)

## Performance Tips

- Use 3-5 frames for temporal window (balance quality/speed)
- Denoising strength 0.5-0.8 works well for most content
- Temporal weight 0.7-0.9 for good temporal stability
- Process at native resolution for best quality
- Use async I/O to overlap file operations with GPU processing

## Getting Help

- Check Vulkan validation layer output
- Use `vulkaninfo` to verify GPU support
- Profile with RenderDoc or similar tools
- Review shader code in `denoise.comp` for algorithm details

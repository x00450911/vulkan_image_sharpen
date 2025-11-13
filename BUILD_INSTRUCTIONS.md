# Build Instructions

## Quick Start

### Prerequisites

1. **Vulkan SDK**
   - Download from: https://vulkan.lunarg.com/
   - Install and ensure `VULKAN_SDK` environment variable is set
   - Verify installation: `glslc --version`

2. **C++ Compiler**
   - GCC 7+ (Linux)
   - Clang 5+ (macOS/Linux)
   - MSVC 2019+ (Windows)

3. **CMake**
   - Version 3.15 or higher
   - Download from: https://cmake.org/download/

### Linux Build

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install cmake g++ vulkan-sdk

# Or install Vulkan SDK from LunarG
wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list \
    https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
sudo apt update
sudo apt install vulkan-sdk

# Build
mkdir build
cd build
cmake ..
make -j$(nproc)

# Run examples
./denoiser_example
```

### macOS Build

```bash
# Install dependencies
brew install cmake vulkan-sdk

# Build
mkdir build
cd build
cmake ..
make -j$(sysctl -n hw.ncpu)

# Run examples
./denoiser_example
```

### Windows Build

```powershell
# Install Vulkan SDK from https://vulkan.lunarg.com/

# Using Visual Studio 2019/2022
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release

# Run examples
.\Release\denoiser_example.exe
```

Or use CMake GUI:
1. Set source directory to project root
2. Set build directory to `project_root/build`
3. Click "Configure" and select your compiler
4. Click "Generate"
5. Open the generated solution file

### Android Build

```bash
# Set Android NDK path
export ANDROID_NDK=/path/to/android-ndk

# Build with gradle
./gradlew assembleRelease

# Or use CMake directly
mkdir build-android
cd build-android
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24
make -j$(nproc)
```

## Build Options

### CMake Options

```bash
# Build with examples (default: ON)
cmake .. -DBUILD_EXAMPLES=ON

# Build without examples
cmake .. -DBUILD_EXAMPLES=OFF

# Build tests (default: OFF)
cmake .. -DBUILD_TESTS=ON

# Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Release build with optimizations
cmake .. -DCMAKE_BUILD_TYPE=Release

# Custom Vulkan SDK path
cmake .. -DVULKAN_SDK=/path/to/vulkan/sdk
```

### Compiler Flags

```bash
# Enable all warnings
cmake .. -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic"

# Enable optimizations
cmake .. -DCMAKE_CXX_FLAGS="-O3 -march=native"

# Enable debugging symbols
cmake .. -DCMAKE_CXX_FLAGS="-g"
```

## Manual Shader Compilation

If automatic shader compilation fails:

```bash
# Navigate to shaders directory
cd shaders

# Compile compute shader
glslc temporal_denoise.comp -o temporal_denoise.spv

# Verify compiled shader
spirv-val temporal_denoise.spv

# Copy to build directory
mkdir -p ../build/shaders
cp temporal_denoise.spv ../build/shaders/
```

## Troubleshooting

### Issue: "Vulkan not found"

**Solution**:
```bash
# Set VULKAN_SDK environment variable
export VULKAN_SDK=/path/to/vulkan/sdk  # Linux/macOS
set VULKAN_SDK=C:\VulkanSDK\1.x.x.x    # Windows

# Add to PATH
export PATH=$VULKAN_SDK/bin:$PATH      # Linux/macOS
set PATH=%VULKAN_SDK%\Bin;%PATH%       # Windows
```

### Issue: "glslc not found"

**Solution**:
```bash
# Verify glslc is in PATH
which glslc       # Linux/macOS
where glslc       # Windows

# If not found, add Vulkan SDK bin directory to PATH
export PATH=$VULKAN_SDK/bin:$PATH
```

### Issue: Compilation errors with GCC

**Solution**: Ensure GCC 7+ is installed
```bash
g++ --version  # Should be 7.0 or higher

# Update GCC if needed
sudo apt-get install gcc-9 g++-9
export CC=gcc-9
export CXX=g++-9
```

### Issue: Linker errors

**Solution**:
```bash
# Clean and rebuild
rm -rf build
mkdir build
cd build
cmake ..
make clean
make -j$(nproc)
```

### Issue: Runtime validation errors

**Solution**: Ensure your GPU supports Vulkan 1.2
```bash
# Check Vulkan support
vulkaninfo | grep "apiVersion"

# Update graphics drivers
# NVIDIA: https://www.nvidia.com/drivers
# AMD: https://www.amd.com/support
# Intel: https://downloadcenter.intel.com/
```

## Performance Optimization

### Build for Maximum Performance

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -ffast-math"
make -j$(nproc)
```

### Build with Link-Time Optimization

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
make -j$(nproc)
```

## Installation

### System-wide Installation (Linux/macOS)

```bash
cd build
sudo make install

# Default locations:
# - Headers: /usr/local/include/
# - Library: /usr/local/lib/
# - Shaders: /usr/local/shaders/
```

### Local Installation

```bash
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local
make install

# Files installed to:
# - $HOME/.local/include/
# - $HOME/.local/lib/
# - $HOME/.local/shaders/
```

### Using Installed Library

```cmake
# In your CMakeLists.txt
find_package(VulkanVideoDenoiser REQUIRED)
target_link_libraries(your_target PRIVATE VulkanVideoDenoiser::vulkan_video_denoiser)
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Build

on: [push, pull_request]

jobs:
  build-linux:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake g++ vulkan-sdk
      - name: Build
        run: |
          mkdir build && cd build
          cmake ..
          make -j$(nproc)
      - name: Test
        run: cd build && ctest --output-on-failure
```

## Development Build

For active development with faster iteration:

```bash
# Use Ninja for faster builds
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug
ninja

# Or use ccache for faster recompilation
cmake .. -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
make -j$(nproc)
```

## Cross-Compilation

### For ARM Linux (Raspberry Pi, etc.)

```bash
# Install cross-compiler
sudo apt-get install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf

# Create toolchain file
cat > arm-toolchain.cmake << EOF
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)
EOF

# Build
mkdir build-arm
cd build-arm
cmake .. -DCMAKE_TOOLCHAIN_FILE=../arm-toolchain.cmake
make -j$(nproc)
```

## Docker Build

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    wget \
    vulkan-sdk

WORKDIR /app
COPY . .

RUN mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc)

CMD ["./build/denoiser_example"]
```

Build and run:
```bash
docker build -t vulkan-denoiser .
docker run --gpus all -it vulkan-denoiser
```

## Verification

After building, verify the installation:

```bash
# Check that example runs
cd build
./denoiser_example

# Verify shader compilation
ls -lh shaders/*.spv

# Run tests (if enabled)
ctest --output-on-failure
```

## Support

If you encounter build issues not covered here, please:
1. Check that all prerequisites are correctly installed
2. Verify Vulkan SDK installation with `vulkaninfo`
3. Update graphics drivers
4. Open an issue with build log output

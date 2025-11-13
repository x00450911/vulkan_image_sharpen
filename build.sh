#!/bin/bash

# Build script for Vulkan Video Denoiser
# Usage: ./build.sh [options]
# Options:
#   -c, --clean     Clean build directory before building
#   -r, --release   Build in Release mode (default)
#   -d, --debug     Build in Debug mode
#   -j, --jobs N    Number of parallel jobs (default: auto-detect)
#   -h, --help      Show this help message

set -e  # Exit on error

# Default values
BUILD_TYPE="Release"
CLEAN_BUILD=false
NUM_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
BUILD_DIR="build"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -r|--release)
            BUILD_TYPE="Release"
            shift
            ;;
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -j|--jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -c, --clean     Clean build directory before building"
            echo "  -r, --release   Build in Release mode (default)"
            echo "  -d, --debug     Build in Debug mode"
            echo "  -j, --jobs N    Number of parallel jobs (default: $NUM_JOBS)"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Clean if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
echo "Creating build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Check for Vulkan SDK
if [ -z "$VULKAN_SDK" ]; then
    echo "Warning: VULKAN_SDK environment variable is not set"
    echo "Attempting to build anyway, but this may fail..."
fi

# Check for glslc
if ! command -v glslc &> /dev/null; then
    echo "Warning: glslc shader compiler not found in PATH"
    echo "Shaders will not be compiled automatically"
    echo "Please install Vulkan SDK or compile shaders manually"
fi

# Run CMake
echo "Running CMake (Build Type: $BUILD_TYPE)..."
cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DBUILD_EXAMPLES=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
echo "Building with $NUM_JOBS parallel jobs..."
cmake --build . --config $BUILD_TYPE -j $NUM_JOBS

# Success message
echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo "Build type: $BUILD_TYPE"
echo "Build directory: $BUILD_DIR"
echo ""

# Check if example was built
if [ -f "denoiser_example" ] || [ -f "Release/denoiser_example.exe" ] || [ -f "Debug/denoiser_example.exe" ]; then
    echo "Example executable built successfully!"
    echo "Run it with:"
    if [ -f "denoiser_example" ]; then
        echo "  ./build/denoiser_example"
    else
        echo "  ./build/$BUILD_TYPE/denoiser_example.exe"
    fi
fi

# Check if shaders were compiled
if [ -f "shaders/temporal_denoise.spv" ]; then
    echo ""
    echo "Shaders compiled successfully!"
    SHADER_SIZE=$(du -h shaders/temporal_denoise.spv | cut -f1)
    echo "  Shader size: $SHADER_SIZE"
else
    echo ""
    echo "Warning: Shaders were not compiled!"
    echo "You may need to compile them manually:"
    echo "  cd shaders"
    echo "  glslc temporal_denoise.comp -o temporal_denoise.spv"
fi

echo ""

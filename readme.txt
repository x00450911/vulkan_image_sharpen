Vulkan Multi-Frame Video Denoiser
===================================

A high-performance GPU-accelerated video denoising implementation using Vulkan compute shaders.

Quick Start:
------------
1. Install Vulkan SDK: https://vulkan.lunarg.com/
2. Build: ./build.sh
3. Run example: ./build/denoiser_example

Documentation:
--------------
- README.md - Complete documentation and API reference
- BUILD_INSTRUCTIONS.md - Detailed build guide for all platforms
- ARCHITECTURE.md - System design and implementation details

Features:
---------
- Multi-frame temporal filtering (up to 4 frames)
- Adaptive denoising with detail preservation
- Real-time performance (60+ FPS at 1080p)
- Motion compensation support
- Bilateral spatial filtering
- Cross-platform (Windows, Linux, macOS, Android)

Directory Structure:
--------------------
include/          - Public API headers
src/              - Implementation files
shaders/          - Vulkan compute shaders (GLSL)
examples/         - Example applications
cmake/            - CMake configuration files
build/            - Build output (created during build)

For more information, see README.md

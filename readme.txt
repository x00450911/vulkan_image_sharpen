Vulkan Multi-Frame Video Denoiser
=================================

This sample demonstrates how to build a multi-frame temporal denoising pipeline for video content using Vulkan compute shaders.

Project structure
-----------------

- `CMakeLists.txt` – build configuration that compiles the host application and the compute shader (via `glslc`).
- `src/` – C++ host-side source code.
  - `VulkanContext.*` – bootstrap for the Vulkan instance, device, queue, and command pool.
  - `VideoDenoiser.*` – GPU resources, descriptor sets, compute pipeline, upload/readback buffers, and per-frame processing.
  - `FrameProvider.*` – synthetic frame generator used for demonstration.
  - `FrameWriter.*` – utility to write denoised frames to disk (PPM format).
  - `main.cpp` – command-line harness that ties everything together.
- `shaders/denoise.comp` – GLSL compute shader implementing temporal & spatial denoising with history feedback.

Prerequisites
-------------

- C++20-capable compiler
- CMake ≥ 3.16
- Vulkan SDK (for headers, loader, and `glslc`)

Build & run
-----------

```bash
cmake -S . -B build
cmake --build build
./build/vulkan_multi_frame_denoiser --width 1280 --height 720 --frames 180 --blend 0.8 --output ./output
```

The executable generates noisy synthetic frames, denoises them using the Vulkan compute pipeline, and writes the results as `output/denoised_<index>.ppm`.

To use real video data, replace the `SyntheticFrameProvider` with a provider that decodes frames (e.g. via FFmpeg) into RGBA8 pixel buffers and feeds them into `VideoDenoiser::processFrame`.
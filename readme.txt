Fast Image Edge Sharpening on Vulkan (Adreno 650)
=================================================

This sample demonstrates an adaptive, edge-aware sharpening filter implemented as a Vulkan compute pipeline. It is tuned for Qualcomm® Adreno™ 650 GPUs, but will fall back to any Vulkan-capable device for development and validation on desktop.

Key Capabilities
----------------
- Edge-aware sharpening that boosts contrast only where gradients exceed a configurable threshold.
- Pure compute pipeline with 16×16 workgroups for efficient occupancy on Adreno 650.
- Minimal dependencies: loads/saves uncompressed TGA images without third-party libraries.
- Push-constant controlled strength/threshold so you can tune quality/performance at dispatch time.

Project Layout
--------------
- `src/main.cpp` – C++ host application that prepares buffers, dispatches the compute workload, and writes the result.
- `shaders/edge_sharpen.comp` – GLSL compute shader implementing the adaptive sharpening kernel.
- `CMakeLists.txt` – Build script targeting Vulkan SDK 1.1+.

Prerequisites
-------------
1. Vulkan SDK 1.2 or newer installed (desktop) or Android NDK r23+ with Vulkan headers/toolchain (device build).
2. `glslc` from `shaderc` (ships with the Vulkan SDK) to compile the GLSL shader.
3. CMake 3.16+ and a C++17 compiler.
4. Input/output images in **uncompressed TGA (24/32-bit)** format. Convert other formats (PNG/JPEG/etc.) using tools like ImageMagick or `ffmpeg`.

Shader Compilation
------------------
Compile the compute shader into SPIR-V before running the host application:

```bash
glslc shaders/edge_sharpen.comp -O -o shaders/edge_sharpen.comp.spv
```

For Qualcomm Android builds, copy the resulting `.spv` into your asset bundle or alongside the executable on-device.

Building (Linux / Desktop Validation)
-------------------------------------
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

This produces `vulkan_edge_sharpen`. You can run it on any Vulkan-capable GPU to validate correctness before deploying to Adreno hardware.

Running the Sharpen Pass
------------------------
```bash
./vulkan_edge_sharpen input.tga output.tga \
    --strength 0.65 \
    --edge-threshold 0.045
```

CLI Flags
- `--strength` controls the aggressiveness of the sharpening (default `0.65`). Use `0.3–1.2` for typical photographic content.
- `--edge-threshold` suppresses sharpening in low-contrast regions (default `0.045`). Increase to reduce amplification of noise.
- `--shader` overrides the default SPIR-V path (`shaders/edge_sharpen.comp.spv`).
- `--quiet` silences runtime logging.

Output images are written as top-left-origin 32-bit TGA to simplify ingestion by other tooling.

Adreno 650 Deployment Notes
---------------------------
- Use the Android NDK toolchain to cross-compile `src/main.cpp` into a shared library or binary as appropriate for your app architecture.
- Ensure the device manifest enables Vulkan (`android.hardware.vulkan.level`).
- Upload the SPIR-V module with `vkCreateShaderModule` exactly as on desktop; no specialization is required.
- For low-latency pipelines, migrate storage buffers to device-local memory and use transfer queues to stage uploads.

Quality & Performance Tuning
----------------------------
- Workgroup size (`16×16`) is selected to align with common Adreno wavefront sizes; adjust if profiling suggests otherwise.
- The compute shader applies both cross and diagonal Laplacian components for balanced sharpening.
- Adaptive strength is derived from gradient magnitude to avoid ringing in flat regions; tweak `edgeThreshold` to match content.
- The host sample uses host-visible coherent buffers for simplicity. For production, prefer device-local memory and explicit barriers for better throughput.

Troubleshooting
---------------
- **Validation errors**: Enable Vulkan validation layers via `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` during development for richer diagnostics.
- **No Adreno device detected**: The sample picks the first Vulkan GPU as a fallback. Verify the device is exposing Vulkan (settings → developer options).
- **Banding or artifacts**: Ensure input images are converted to 32-bit TGA or adjust the `--strength` parameter downward.

License
-------
The code in this repository is provided under the MIT license. Replace or augment as needed for your project. Qualcomm and Adreno are trademarks of Qualcomm Incorporated.

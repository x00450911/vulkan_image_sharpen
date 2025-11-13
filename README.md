## Vulkan Multi-Frame Video Denoising

This sample demonstrates how to implement a temporal (multi-frame) denoiser for floating-point video data using Vulkan compute shaders. The application consumes a directory of `.pfm` (Portable Float Map) frames, combines multiple consecutive frames to suppress noise, and writes the denoised results back to disk.

### Features
- Vulkan 1.3 compute-only pipeline with optional validation layers.
- Ring-buffer history of up to four floating-point frames (`RGBA16F` on GPU).
- Temporal accumulation buffer to stabilize output over time.
- Configurable blend factor and luminance-based weighting for history contributions.

For architectural details see `docs/design.md`.

---

### Prerequisites
- A Vulkan 1.3-capable GPU and drivers.
- Vulkan SDK installed (required for `glslangValidator`).
- CMake ≥ 3.20 and a C++20 compiler (e.g., clang 15+, GCC 11+, MSVC 2022).

The build system expects `glslangValidator` to be available via the Vulkan SDK.

---

### Building
```bash
cmake -S . -B build
cmake --build build
```

The shader is compiled automatically to `build/shaders/denoise.comp.spv` during the build.

---

### Preparing Input Frames
The sample processes **PFM** images (`.pfm` extension) encoded as RGB floating-point data. Each image is converted to RGBA on load with an implicit alpha of 1.0.

You can generate PFM frames via tools such as Blender, OpenImageIO, or by exporting from scientific software. All frames must share the same resolution.

---

### Running
```bash
./build/denoiser \
  --input path/to/noisy_frames \
  --output path/to/denoised_frames \
  --history 4 \
  --blend 0.1 \
  --sigma 0.2 \
  [--shader /custom/path/to/denoise.comp.spv] \
  [--validation]
```

**Key options**
- `--input`: Directory containing `.pfm` frames to denoise (required).
- `--output`: Destination directory (created if missing). Defaults to `denoised_output`.
- `--history`: Number of history frames to blend (1–4). Defaults to 4.
- `--blend`: Factor (0–1) controlling reliance on the previous accumulated result.
- `--sigma`: Luminance sigma for history weighting (higher = stronger smoothing).
- `--shader`: Override for the compute shader path. Defaults to `./shaders/denoise.comp.spv` relative to the working directory.
- `--validation`: Enables Vulkan validation layers (requires SDK runtime).

Progress is printed per frame. The output frames retain the original filenames.

---

### Algorithm Overview
1. Upload the latest frame into a GPU history image.
2. Compute luminance-weighted averages across the available history.
3. Fuse the temporal average with the accumulated denoised result using the blend factor.
4. Store both the denoised frame (for readback) and the updated accumulation buffer.

The compute shader is located at `shaders/denoise.comp`. Host-side orchestration lives in `src/multi_frame_denoiser.cpp` and `src/main.cpp`.

---

### Extending the Sample
- Integrate optical flow or motion vectors to perform motion-compensated sampling.
- Replace the PFM loader with an FFmpeg-based decoder for direct video pipelines.
- Use timeline semaphores and persistent command buffers to overlap transfers and compute.

---

### Troubleshooting
- **glslangValidator not found**: Ensure the Vulkan SDK is installed and `glslangValidator` is in your `PATH`.
- **Validation errors**: Run with `--validation` to obtain detailed diagnostics while developing.
- **Mismatched frame sizes**: All input frames must share the same width and height.

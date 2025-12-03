# Qualcomm Gen2 Hybrid Image Enhancement

This repository contains a C++17 reference pipeline that combines multi-frame
MCTF denoising and super-resolution, mirroring how a production-quality camera
stack can utilize the heterogeneous resources (CPU + DSP + GPU + NPU) available
on Qualcomm Snapdragon Gen2 platforms.

## Features
- Lightweight floating-point image container + utilities (no external deps).
- Motion-compensated temporal filtering (MCTF) accelerated on DSP + CPU.
- Two-stage super-resolution: bilinear upsample on CPU, Laplacian detail lift on
  GPU, and neural refinement placeholder on NPU.
- Qualcomm Gen2 runtime shim that models power/topology management and exposes a
  single dispatch interface for workloads destined to CPU/DSP/GPU/NPU.
- Demo driver that synthesizes low-res noisy frames, denoises them, runs super
  resolution, and emits a `sr_output.pgm` artifact.

## Building (host simulation)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/gen2_demo
```
The executable prints per-accelerator latency/utilization estimates and writes a
PGM image you can inspect with tools such as `display` or `ffplay`.

## Deploying on Snapdragon Gen2
1. **Toolchain setup**: Install the Qualcomm AI Stack (QNN/SNPE), Hexagon SDK,
   and the Android NDK r26+. Export `QNN_SDK_ROOT`, `HEXAGON_SDK_ROOT`, and
   update your PATH to include `hexagon-clang`.
2. **Cross build**:
   ```bash
   cmake -S . -B build-android \
         -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 \
         -DGEN2_ENABLE_OPENMP=ON
   cmake --build build-android -j
   ```
3. **DSP/NPU delegation**: Replace the placeholder lambdas inside
   `super_resolution.cpp` and `mctf_denoiser.cpp` with invocations to your QNN
   or SNPE graph APIs. The `QualcommGen2Runtime` class is the choke point for
   loading contexts, submitting inferences, and synchronizing with the host.
4. **Deployment**: Push the binary plus any QNN/SNPE artifacts (`.bin`, `.dlc`)
   to `/data/local/tmp`, start `adbd` in root mode, and run the executable. Hexagon
   fastRPC services must be enabled for DSP offload.

## Extending the pipeline
- Swap the synthetic gradient generator for actual RAW frames captured from the
  ISP or a file reader (e.g., DNG via `libraw`).
- Train an SR model (EDSR, RCAN, etc.) and export via ONNX → QNN format. Bind
  it inside the NPU stage (`NeuralRefine` workload).
- Replace the simple Laplacian edge boost with FFT-based frequency separation on
  the GPU to better preserve textures.
- Add a quality assurance harness that compares PSNR/SSIM before/after MCTF + SR
  and records power/perf deltas across the four accelerators.

## Repository layout
```
include/
  image.h                # Minimal tensor-like container
  accelerator/           # Qualcomm runtime shim
  super_resolution.h     # SR orchestrator
  mctf_denoiser.h        # Motion-compensated temporal filter
src/
  *.cpp                  # Implementations + demo entry point
CMakeLists.txt
```

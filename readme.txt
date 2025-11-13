Vulkan Android Video Pipeline
=============================

This repository contains a reference implementation of a zero-copy Android video
enhancement pipeline optimised for Adreno 650 GPUs. It ingests RGB
`AHardwareBuffer` frames, runs a sequence of Vulkan compute shaders (rotation +
sharpen, exposure analysis, multi-frame denoise, and super-resolution), and
emits another `AHardwareBuffer` for downstream consumers.

Key Components
--------------
- `src/VulkanAndroidFrameProcessor.{h,cpp}` — high-level orchestration layer that
  imports hardware buffers, manages descriptor sets, and submits the compute
  workloads.
- `src/VulkanContext.{h,cpp}` — Vulkan instance/device bootstrapper with Android
  external memory support enabled.
- `shaders/*.comp` — GLSL compute shaders for each processing stage.
- `shaders/compile_shaders.sh` — helper script that compiles GLSL into SPIR-V
  using `glslangValidator`.

Build Instructions
------------------
1. Ensure the Android NDK provides `glslangValidator` in your `PATH`.
2. Compile the shaders once (this will emit the `.spv` binaries that the loader
   expects):

       cd shaders
       ./compile_shaders.sh

3. Build the static library with CMake (from an Android-aware toolchain):

       cmake -S . -B build-android -DANDROID_ABI=arm64-v8a \
             -DANDROID_PLATFORM=android-29 \
             -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake
       cmake --build build-android --target vulkan_android_video_pipeline

Usage Outline
-------------
```cpp
using namespace video::processing;

auto context = VulkanContext::create({});
FrameDimensions dims{width, height};
PipelineConfig config{};

VulkanAndroidFrameProcessor processor(context, dims, config);
PipelineOutputs outputs = processor.processFrame(inputBuffer, historyBuffers, outputBuffer);
```

`historyBuffers` is a span of previous frames (newest first) that are reused for
temporal denoising. The processor imports and releases all buffers internally.

Runtime Notes
-------------
- The loader will automatically invoke `glslangValidator` if it cannot find the
  compiled `.spv` file. Ship precompiled shaders in production to avoid spawn
  overhead.
- Exposure statistics are returned as per-tile ratios (0–1). The average of these
  tiles serves as a quick luminance proxy.
- Super-resolution currently keeps the output resolution identical to the input
  while enhancing high-frequency detail. Adjust the final pass to write into a
  larger target image if true upscaling is required.

License
-------
Provided as-is for evaluation purposes. Incorporate into your project under
your preferred licensing policy.

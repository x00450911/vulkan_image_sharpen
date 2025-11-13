Motion Vector Estimation for Denoising
======================================

This repository provides a self-contained C++17 implementation of block-based
motion estimation suitable for video denoising workflows. The core utility
computes motion vectors between a reference frame and a neighbouring frame using
sum-of-absolute-differences (SAD) or sum-of-squared-differences (SSD) matching.
The resulting motion field can be used to align adjacent frames prior to temporal
filtering.

Project Layout
--------------

- `include/motion_vectors.hpp` — public API for frame views, parameter handling,
  and motion vector output.
- `src/motion_vectors.cpp` — block-matching implementation with optional motion
  regularisation.
- `src/main.cpp` — demonstration program that synthetically shifts a noisy frame
  and recovers its motion vectors.
- `CMakeLists.txt` — simple CMake build configuration that produces both a static
  library and the demo executable.

Building
--------

```
cmake -S . -B build
cmake --build build
```

> **Note:** The sample environment used for development was missing the C++
> runtime (`libstdc++`). If your toolchain does not provide it by default,
> install the standard library package for your compiler (e.g. `libstdc++-dev`
> on Debian/Ubuntu) before configuring CMake.

Running the Demo
----------------

```
./build/motion_vectors_demo
```

The program reports a sample of motion vectors plus the mean displacement,
allowing you to verify that the recovered motion aligns with the known synthetic
shift.

Integrating with Denoising
--------------------------

The primary function, `denoise::compute_motion_vectors`, returns a vector of
`BlockMotionVector` entries. Each entry contains the block origin `(block_x,
block_y)`, the optimal displacement `(dx, dy)`, and a normalised matching cost.
Use these vectors to warp neighbouring frames back onto the reference grid
before blending or applying temporal filters. For example:

```cpp
const auto motion_field = denoise::compute_motion_vectors(ref_view, neighbour_view, params);
// Warp neighbour frame using motion_field, then blend across frames to reduce noise.
```

Adjust the `MotionEstimationParams` fields (`block_size`, `search_range`,
`step`, `cost_type`, and `penalty_lambda`) to balance accuracy, robustness to
noise, and runtime.
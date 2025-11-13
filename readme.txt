Motion Vector Utilities
=======================

This repository contains a lightweight Python implementation of block-based
motion vector estimation suitable for motion-compensated denoising or other
temporal processing steps.

Key features:
- Full-search block matching with configurable block size and search radius.
- Optional pyramid refinement for large displacements.
- Half-pixel refinement using bilinear interpolation.
- Frame compensation helper to warp the reference frame using the estimated
  motion field.

Quickstart
----------
1. Install the only dependency:
   ```
   pip install numpy
   ```
2. Estimate motion vectors between two grayscale frames and perform simple
   motion-compensated averaging:
   ```python
   import numpy as np
   from src.motion_vectors import (
       BlockMatcherConfig,
       estimate_motion_field,
       compensate_frame,
   )

   prev = np.load("frame_000.npy")  # shape (H, W)
   curr = np.load("frame_001.npy")

   config = BlockMatcherConfig(block_size=16, search_radius=8, pyramid_levels=1)
   motion = estimate_motion_field(prev, curr, config)
   denoised = 0.5 * (curr + compensate_frame(prev, motion, config.block_size))
   np.save("frame_001_denoised.npy", denoised)
   ```

All core functionality lives in `src/motion_vectors.py`. See the module
docstring for additional guidance and notes.

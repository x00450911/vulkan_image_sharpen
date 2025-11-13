"""Motion vector estimation utilities for motion-compensated denoising.

This module provides a flexible block-matching implementation that produces
motion vectors between two successive video frames. The resulting motion field
can be used to drive motion compensated denoising pipelines (for instance,
temporal averaging aligned by the estimated motion).

The implementation intentionally avoids non-standard dependencies so it can run
in constrained environments. NumPy is the only external requirement.

Example
-------
```python
import numpy as np
from motion_vectors import BlockMatcherConfig, estimate_motion_field, compensate_frame

prev = np.load("frame_000.npy")  # shape (H, W), dtype float32
curr = np.load("frame_001.npy")

config = BlockMatcherConfig(block_size=16, search_radius=8, pyramid_levels=1)
motion = estimate_motion_field(prev, curr, config)
denoised = 0.5 * (curr + compensate_frame(prev, motion))
```

Notes
-----
* Frames are expected in grayscale (single channel). Use luminance or convert
  RGB frames to Y using your preferred method before estimating motion.
* Block matching is most effective when frames are pre-filtered to suppress
  noise that might confuse the matching metric. A small bilateral or Gaussian
  filter often helps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class BlockMatcherConfig:
    """Configuration parameters for block matching.

    Attributes
    ----------
    block_size:
        Size (in pixels) of the square blocks. Larger blocks provide smoother
        motion but may miss fine detail. Must evenly divide the frame height and
        width; padding can be applied beforehand if necessary.
    search_radius:
        Maximum displacement (in pixels) searched in both horizontal and
        vertical directions at each pyramid level.
    penalty_lambda:
        Optional L2 penalty applied to vector magnitude. This biases the search
        towards smaller motion vectors when multiple candidates yield identical
        matching costs.
    pyramid_levels:
        Number of additional downsampled levels (besides the original) to use.
        Values greater than zero enable a coarse-to-fine refinement strategy.
    use_half_pixel:
        When True, a half-pixel refinement step is executed after integer-pixel
        block matching by evaluating the cost in a 2×2 neighborhood using
        bilinear interpolation.
    """

    block_size: int = 16
    search_radius: int = 7
    penalty_lambda: float = 0.0
    pyramid_levels: int = 0
    use_half_pixel: bool = False

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.search_radius <= 0:
            raise ValueError("search_radius must be positive")
        if self.pyramid_levels < 0:
            raise ValueError("pyramid_levels cannot be negative")
        if self.penalty_lambda < 0.0:
            raise ValueError("penalty_lambda cannot be negative")


def estimate_motion_field(
    ref_frame: np.ndarray,
    target_frame: np.ndarray,
    config: BlockMatcherConfig | None = None,
) -> np.ndarray:
    """Estimate motion vectors between two frames using block matching.

    Parameters
    ----------
    ref_frame:
        The reference (previous) frame, grayscale float or uint array of shape
        (H, W). Any dtype convertible to float32 is accepted; values are used as-is.
    target_frame:
        The target (current) frame to which motion is measured, same shape as
        `ref_frame`.
    config:
        Matching configuration. If omitted, defaults to :class:`BlockMatcherConfig`.

    Returns
    -------
    np.ndarray
        Motion vectors with shape (num_blocks_y, num_blocks_x, 2). The last axis
        stores (dy, dx) displacements in pixels relative to the reference frame.

    Raises
    ------
    ValueError
        If frame shapes differ or are not divisible by `block_size`.
    """

    if config is None:
        config = BlockMatcherConfig()

    ref = np.asarray(ref_frame, dtype=np.float32)
    tgt = np.asarray(target_frame, dtype=np.float32)

    if ref.shape != tgt.shape:
        raise ValueError("ref_frame and target_frame must share the same shape")
    if ref.ndim != 2:
        raise ValueError("frames must be 2D grayscale arrays")

    block = config.block_size
    h, w = ref.shape
    if h % block != 0 or w % block != 0:
        raise ValueError(
            f"frame dimensions ({h}, {w}) must be divisible by block_size={block}"
        )

    pyr_ref = list(_build_pyramid(ref, config.pyramid_levels))
    pyr_tgt = list(_build_pyramid(tgt, config.pyramid_levels))

    motion = np.zeros((h // block, w // block, 2), dtype=np.float32)
    if config.pyramid_levels == 0:
        motion = _block_match_level(pyr_ref[0], pyr_tgt[0], config, motion)
    else:
        scale = 1 << config.pyramid_levels
        # Start at coarsest level
        motion = _block_match_level(
            pyr_ref[-1], pyr_tgt[-1], config, motion / scale, level=config.pyramid_levels
        )
        # Refine towards full resolution
        for level in reversed(range(config.pyramid_levels)):
            motion = _upsample_motion(motion, pyr_ref[level].shape, config.block_size)
            motion = _block_match_level(
                pyr_ref[level], pyr_tgt[level], config, motion, level=level
            )

    return motion


def compensate_frame(
    ref_frame: np.ndarray,
    motion: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """Warp a reference frame using block motion vectors.

    Parameters
    ----------
    ref_frame:
        Base frame to be warped (2D array of shape (H, W)).
    motion:
        Motion vectors as returned by :func:`estimate_motion_field`.
    block_size:
        Block size (must match configuration used for motion estimation).

    Returns
    -------
    np.ndarray
        Motion compensated frame with the same shape as `ref_frame`.
    """

    ref = np.asarray(ref_frame, dtype=np.float32)
    h, w = ref.shape
    blocks_y, blocks_x, _ = motion.shape
    if blocks_y * block_size != h or blocks_x * block_size != w:
        raise ValueError("motion field shape does not match frame and block size")

    compensated = np.zeros_like(ref)
    for by in range(blocks_y):
        for bx in range(blocks_x):
            dy, dx = motion[by, bx]
            y0 = by * block_size
            x0 = bx * block_size
            block = _bilinear_sample(
                ref,
                y0 + dy + np.arange(block_size)[:, None],
                x0 + dx + np.arange(block_size)[None, :],
            )
            compensated[y0 : y0 + block_size, x0 : x0 + block_size] = block
    return compensated


# ---------------------------------------------------------------------------
# Internal helpers


def _block_match_level(
    ref: np.ndarray,
    tgt: np.ndarray,
    config: BlockMatcherConfig,
    init_motion: np.ndarray,
    level: int = 0,
) -> np.ndarray:
    block = config.block_size
    search = config.search_radius / (1 << level)
    search = int(max(1, round(search)))

    h, w = ref.shape
    blocks_y, blocks_x, _ = init_motion.shape
    motion = np.empty_like(init_motion, dtype=np.float32)

    for by in range(blocks_y):
        for bx in range(blocks_x):
            y0 = by * block
            x0 = bx * block

            candidate = init_motion[by, bx]
            best_cost = np.inf
            best_vec = candidate

            for dy in range(-search, search + 1):
                for dx in range(-search, search + 1):
                    vec = candidate + np.array([dy, dx], dtype=np.float32)
                    cost = _matching_cost(
                        ref, tgt, y0, x0, vec, block, config.penalty_lambda
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_vec = vec

            if config.use_half_pixel:
                best_vec = _refine_half_pixel(
                    ref, tgt, y0, x0, best_vec, block, config.penalty_lambda
                )

            motion[by, bx] = best_vec

    return motion


def _matching_cost(
    ref: np.ndarray,
    tgt: np.ndarray,
    y0: int,
    x0: int,
    vec: np.ndarray,
    block_size: int,
    penalty_lambda: float,
) -> float:
    y = y0 + vec[0]
    x = x0 + vec[1]
    block_ref = ref[y0 : y0 + block_size, x0 : x0 + block_size]
    block_tgt = _bilinear_sample(
        tgt,
        y + np.arange(block_size)[:, None],
        x + np.arange(block_size)[None, :],
    )
    sad = np.abs(block_ref - block_tgt).mean()
    penalty = penalty_lambda * float(np.dot(vec, vec))
    return sad + penalty


def _refine_half_pixel(
    ref: np.ndarray,
    tgt: np.ndarray,
    y0: int,
    x0: int,
    vec: np.ndarray,
    block_size: int,
    penalty_lambda: float,
) -> np.ndarray:
    offsets = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, -0.5],
            [0.5, 0.0],
            [-0.5, 0.0],
            [0.5, 0.5],
            [0.5, -0.5],
            [-0.5, 0.5],
            [-0.5, -0.5],
        ],
        dtype=np.float32,
    )
    best_vec = vec
    best_cost = _matching_cost(
        ref, tgt, y0, x0, vec, block_size, penalty_lambda
    )
    for off in offsets[1:]:
        candidate = vec + off
        cost = _matching_cost(
            ref, tgt, y0, x0, candidate, block_size, penalty_lambda
        )
        if cost < best_cost:
            best_cost = cost
            best_vec = candidate
    return best_vec


def _bilinear_sample(img: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    h, w = img.shape
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)

    wy = ys - y0
    wx = xs - x0

    top = (1 - wx) * img[y0, x0] + wx * img[y0, x1]
    bottom = (1 - wx) * img[y1, x0] + wx * img[y1, x1]
    return (1 - wy) * top + wy * bottom


def _build_pyramid(frame: np.ndarray, levels: int) -> Iterable[np.ndarray]:
    pyramid = [frame]
    current = frame
    for _ in range(levels):
        current = _downsample(current)
        pyramid.append(current)
    return pyramid


def _downsample(frame: np.ndarray) -> np.ndarray:
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    temp = _convolve1d(frame, kernel, axis=0)
    blurred = _convolve1d(temp, kernel, axis=1)
    return blurred[::2, ::2]


def _convolve1d(img: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    padded = np.pad(img, [(radius, radius) if i == axis else (0, 0) for i in range(2)], mode="edge")
    out = np.empty_like(img)
    if axis == 0:
        for y in range(img.shape[0]):
            out[y] = np.sum(
                kernel[:, None] * padded[y : y + len(kernel)], axis=0
            )
    else:
        for x in range(img.shape[1]):
            out[:, x] = np.sum(
                kernel[None, :] * padded[:, x : x + len(kernel)],
                axis=1,
            )
    return out


def _upsample_motion(
    motion: np.ndarray, target_shape: Tuple[int, int], block_size: int
) -> np.ndarray:
    blocks_y = target_shape[0] // block_size
    blocks_x = target_shape[1] // block_size
    upsampled = np.zeros((blocks_y, blocks_x, 2), dtype=np.float32)
    scale_y = upsampled.shape[0] / motion.shape[0]
    scale_x = upsampled.shape[1] / motion.shape[1]

    for by in range(blocks_y):
        for bx in range(blocks_x):
            src_y = min(int(by / scale_y), motion.shape[0] - 1)
            src_x = min(int(bx / scale_x), motion.shape[1] - 1)
            upsampled[by, bx] = motion[src_y, src_x] * np.array(
                [scale_y, scale_x], dtype=np.float32
            )

    return upsampled


__all__ = [
    "BlockMatcherConfig",
    "estimate_motion_field",
    "compensate_frame",
]

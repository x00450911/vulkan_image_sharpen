"""High-level orchestration for video super-resolution inference."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from .config import ConfigBundle
from .io import VideoReader, VideoWriter
from .model_loader import build_model
from .utils import (
    LOGGER,
    chunk_iterable,
    get_autocast,
    maybe_enable_benchmark,
    setup_logging,
    torch_no_grad,
)


class SuperResolutionPipeline:
    """End-to-end inference helper."""

    def __init__(self, config: ConfigBundle) -> None:
        self.cfg = config
        setup_logging()
        maybe_enable_benchmark(config.pipeline.benchmark_cuda)
        self.model, self.device = build_model(
            config.model, device_override=config.pipeline.device
        )
        self.model_dtype = next(self.model.parameters()).dtype
        self._effective_precision = (
            self.cfg.model.precision if self.device.type == "cuda" else "fp32"
        )

    def _window_to_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Stack a temporal window into CHW tensor."""
        window = np.stack(frames, axis=0)  # (T, H, W, C)
        window = window[..., ::-1]  # BGR -> RGB
        window = window.astype(np.float32) / 255.0
        window = np.transpose(window, (0, 3, 1, 2))  # (T, C, H, W)
        t, c, h, w = window.shape
        window = window.reshape(t * c, h, w)
        return torch.from_numpy(window)

    def _infer_batch(self, batch_tensors: List[torch.Tensor]) -> List[np.ndarray]:
        """Run the neural network on a batch of windows."""
        batch = torch.stack(batch_tensors, dim=0)
        batch = batch.to(self.device, dtype=self.model_dtype)
        with get_autocast(self.device, self._effective_precision), torch_no_grad():
            predictions = self.model(batch)
        predictions = predictions.clamp(0.0, 1.0).to(torch.float32)
        frames = predictions.cpu().numpy()
        frames = np.transpose(frames, (0, 2, 3, 1))  # (B, H, W, C)
        frames = (frames[..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)  # RGB -> BGR
        return [frame for frame in frames]

    def _build_windows(self, frames: List[np.ndarray]) -> List[List[np.ndarray]]:
        """Pre-compute temporal windows for each frame."""
        radius = self.cfg.model.temporal_radius
        if radius == 0:
            return [[frame] for frame in frames]
        windows: List[List[np.ndarray]] = []
        total = len(frames)
        for idx in range(total):
            window = []
            for offset in range(-radius, radius + 1):
                neighbor = min(max(idx + offset, 0), total - 1)
                window.append(frames[neighbor])
            windows.append(window)
        return windows

    def _warmup(self, windows: List[List[np.ndarray]]) -> None:
        """Optional warmup passes to stabilize CUDA clocks."""
        warmup_steps = self.cfg.pipeline.warmup_steps
        if warmup_steps <= 0 or self.device.type != "cuda":
            return
        LOGGER.info("Running %d warmup steps on %s", warmup_steps, self.device)
        batch = [self._window_to_tensor(windows[0])]
        for _ in range(warmup_steps):
            _ = self._infer_batch(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def run(self) -> Path:
        """Execute the full super-resolution pipeline."""
        video_cfg = self.cfg.video
        LOGGER.info("Reading video %s", video_cfg.input_path)
        reader = VideoReader(video_cfg.input_path)
        frames = reader.read_all(max_frames=video_cfg.max_frames)
        if not frames:
            raise RuntimeError("No frames loaded from the input video.")

        upscale = self.cfg.model.upscale_factor
        output_size = (reader.width * upscale, reader.height * upscale)
        fps = video_cfg.fps or reader.fps
        writer = VideoWriter(
            video_cfg.output_path,
            fps=fps,
            frame_size=output_size,
            codec=video_cfg.codec,
        )

        windows = self._build_windows(frames)
        self._warmup(windows)
        chunk_size = max(1, video_cfg.chunk_size)
        progress = tqdm(total=len(frames), desc="Super-resolving", unit="frame")

        try:
            for chunk_indices in chunk_iterable(list(range(len(frames))), chunk_size):
                batch = [self._window_to_tensor(windows[idx]) for idx in chunk_indices]
                outputs = self._infer_batch(batch)
                for frame in outputs:
                    writer.write(frame)
                    progress.update(1)
        finally:
            progress.close()
            reader.release()
            writer.release()

        LOGGER.info("Saved super-resolved video to %s", video_cfg.output_path)
        return Path(video_cfg.output_path)

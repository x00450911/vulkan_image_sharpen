"""Video IO utilities built on top of OpenCV."""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from .utils import LOGGER


class VideoReader:
    """Thin wrapper over cv2.VideoCapture."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Unable to open video {self.path}")
        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    def read_all(self, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """Read the entire video into memory."""
        frames: List[np.ndarray] = []
        limit = max_frames or self.frame_count or float("inf")
        success = True
        while success and len(frames) < limit:
            success, frame = self.cap.read()
            if success:
                frames.append(frame)
        LOGGER.info("Loaded %d frames from %s", len(frames), self.path)
        return frames

    def release(self) -> None:
        if self.cap:
            self.cap.release()


class VideoWriter:
    """Write numpy frames into a video file."""

    def __init__(
        self,
        path: str | Path,
        fps: float,
        frame_size,
        codec: str = "mp4v",
    ) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(self.path, fourcc, fps, frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {self.path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def write_frames(self, frames: List[np.ndarray]) -> None:
        for frame in frames:
            self.write(frame)

    def release(self) -> None:
        if self.writer:
            self.writer.release()

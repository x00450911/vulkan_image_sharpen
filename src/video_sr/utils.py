"""General utility helpers."""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator

import torch


LOGGER = logging.getLogger("video_sr")


def setup_logging(verbose: bool = True) -> None:
    """Configure root logging only once."""
    if LOGGER.handlers:
        return
    level = logging.INFO if verbose else logging.WARNING
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    LOGGER.setLevel(level)
    LOGGER.addHandler(handler)


def resolve_device(device_str: str) -> torch.device:
    """Resolve a device string into a torch.device with fallbacks."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def should_use_half(precision: str, device: torch.device) -> bool:
    """Return True when the selected precision benefits from half tensors."""
    return device.type == "cuda" and precision.lower() in {"fp16", "bf16"}


def get_autocast(device: torch.device, precision: str):
    """Return a context manager for autocast based on the device/precision."""
    if device.type != "cuda":
        return nullcontext()
    precision = precision.lower()
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(precision)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def ensure_checkpoint(path: str | Path) -> Path:
    """Validate that a checkpoint exists."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_path}' is missing. Please provide a .pth file."
        )
    return checkpoint_path


@contextmanager
def torch_no_grad():
    """Shortcut for torch.inference_mode() with a descriptive name."""
    with torch.inference_mode():
        yield


def maybe_enable_benchmark(flag: bool) -> None:
    """Enable cudnn benchmark for better throughput on RTX 40 series GPUs."""
    if flag and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


def chunk_iterable(items, chunk_size: int) -> Iterator[list]:
    """Yield successive fixed-size chunks from an iterable list."""
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]

"""Factory utilities for constructing models and loading checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import torch

from .config import ModelConfig
from .model_zoo import VideoSRNet
from .utils import ensure_checkpoint, resolve_device, should_use_half

LOGGER = logging.getLogger("video_sr")


def _instantiate_model(cfg: ModelConfig) -> torch.nn.Module:
    architecture = cfg.architecture.lower()
    if architecture == "videosrnet":
        return VideoSRNet(
            in_channels=cfg.in_channels,
            hidden_channels=cfg.hidden_channels,
            num_residual_blocks=cfg.num_residual_blocks,
            upscale_factor=cfg.upscale_factor,
            temporal_radius=cfg.temporal_radius,
        )
    raise ValueError(f"Unsupported architecture '{cfg.architecture}'")


def _load_checkpoint(
    model: torch.nn.Module, weights_path: Path, device: torch.device, strict: bool
) -> None:
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        LOGGER.warning("Missing keys when loading checkpoint: %s", missing)
    if unexpected:
        LOGGER.warning("Unexpected keys when loading checkpoint: %s", unexpected)


def build_model(
    cfg: ModelConfig,
    device_override: str | None = None,
    strict_weights: bool = True,
) -> Tuple[torch.nn.Module, torch.device]:
    """Instantiate a model, move it onto the target device, and load weights."""
    device = resolve_device(device_override or "auto")
    model = _instantiate_model(cfg).to(device)
    if should_use_half(cfg.precision, device):
        dtype = torch.float16 if cfg.precision.lower() == "fp16" else torch.bfloat16
        model = model.to(dtype=dtype)

    weights_path = ensure_checkpoint(cfg.weights_path)
    _load_checkpoint(model, weights_path, device, strict=strict_weights)
    model.eval()
    if strict_weights:
        LOGGER.info("Loaded checkpoint %s onto %s", weights_path, device)
    return model, device

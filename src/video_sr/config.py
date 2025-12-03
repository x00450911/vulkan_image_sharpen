"""Configuration helpers for the video super-resolution pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ModelConfig:
    """High-level knobs for the neural network."""

    architecture: str = "videosrnet"
    upscale_factor: int = 4
    in_channels: int = 3
    hidden_channels: int = 64
    num_residual_blocks: int = 12
    temporal_radius: int = 1
    weights_path: str = "checkpoints/videosrnet_x4.pth"
    precision: str = "fp16"  # one of {"fp32", "fp16", "bf16"}


@dataclass
class VideoIOConfig:
    """Input/output settings for decoding and encoding video streams."""

    input_path: str = "input.mp4"
    output_path: str = "output_sr.mp4"
    fps: Optional[float] = None
    codec: str = "mp4v"
    chunk_size: int = 24
    max_frames: Optional[int] = None


@dataclass
class PipelineConfig:
    """Runtime configuration for GPU execution."""

    device: str = "auto"  # "auto" | "cuda" | "cpu"
    warmup_steps: int = 2
    benchmark_cuda: bool = True


@dataclass
class ConfigBundle:
    """Container for the three config sections."""

    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoIOConfig = field(default_factory=VideoIOConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ConfigBundle":
        """Create a bundle from plain dictionaries, applying defaults."""

        def build(section_cls, section_data):
            if section_data is None:
                return section_cls()
            return section_cls(**section_data)

        return cls(
            model=build(ModelConfig, config_dict.get("model")),
            video=build(VideoIOConfig, config_dict.get("video")),
            pipeline=build(PipelineConfig, config_dict.get("pipeline")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ConfigBundle":
        """Load bundle from a YAML file."""
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize bundle to a nested dictionary."""
        return {
            "model": asdict(self.model),
            "video": asdict(self.video),
            "pipeline": asdict(self.pipeline),
        }

    def dump_yaml(self, path: str | Path) -> None:
        """Persist current configuration into a YAML file."""
        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

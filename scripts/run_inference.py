#!/usr/bin/env python3
"""Command-line interface for the video super-resolution pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from video_sr.config import ConfigBundle
from video_sr.pipeline import SuperResolutionPipeline
from video_sr.utils import LOGGER, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video super-resolution runner")
    parser.add_argument("--config", type=Path, help="Optional YAML config file")
    parser.add_argument("--input", required=True, help="Path to the input video")
    parser.add_argument("--output", required=True, help="Path to save the SR video")
    parser.add_argument("--weights", required=True, help="Path to the .pth checkpoint")
    parser.add_argument("--device", default="auto", help="Device string, e.g. cuda:0")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--chunk-size", type=int, default=24, help="Frames per batch")
    parser.add_argument("--max-frames", type=int, help="Limit number of frames")
    parser.add_argument("--upscale", type=int, default=4, help="Upscale factor")
    parser.add_argument("--temporal-radius", type=int, default=1, help="Temporal window radius")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    if args.config:
        cfg = ConfigBundle.from_yaml(args.config)
    else:
        cfg = ConfigBundle()

    cfg.video.input_path = args.input
    cfg.video.output_path = args.output
    cfg.video.chunk_size = args.chunk_size
    cfg.video.max_frames = args.max_frames

    cfg.model.weights_path = args.weights
    cfg.model.precision = args.precision
    cfg.model.upscale_factor = args.upscale
    cfg.model.temporal_radius = args.temporal_radius

    cfg.pipeline.device = args.device

    LOGGER.info("Starting inference with config: %s", cfg)
    pipeline = SuperResolutionPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()

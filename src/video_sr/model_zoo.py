"""Lightweight reference models for video super-resolution."""

from __future__ import annotations

from typing import List

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Standard two-layer residual block with channel attention."""

    def __init__(self, channels: int, expansion: int = 4, use_attention: bool = True):
        super().__init__()
        hidden = channels * expansion // 2
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 8, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 8, channels, kernel_size=1),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        if self.use_attention:
            x = x * self.attention(x)
        return x + residual


class UpsampleBlock(nn.Module):
    """PixelShuffle-based upsampling."""

    def __init__(self, channels: int, scale: int):
        super().__init__()
        blocks: List[nn.Module] = []
        factor = scale
        while factor > 1:
            blocks += [
                nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ]
            factor //= 2
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VideoSRNet(nn.Module):
    """A simple yet effective spatio-temporal SR network."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        num_residual_blocks: int = 12,
        upscale_factor: int = 4,
        temporal_radius: int = 1,
    ) -> None:
        super().__init__()
        self.temporal_radius = temporal_radius
        stacked_channels = in_channels * (2 * temporal_radius + 1)
        self.head = nn.Conv2d(stacked_channels, hidden_channels, kernel_size=3, padding=1)

        body = [ResidualBlock(hidden_channels) for _ in range(num_residual_blocks)]
        self.body = nn.Sequential(*body)

        self.tail = nn.Sequential(
            UpsampleBlock(hidden_channels, upscale_factor),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shaped as (B, stacked_channels, H, W).
        """
        base = self.head(x)
        features = self.body(base)
        output = self.tail(features + base)
        return output

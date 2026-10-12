"""Shared test doubles.

The real networks are large and slow, so tests use a nearest-neighbor fake that
satisfies the same SuperResolutionNetwork contract and gives exact, predictable
outputs.
"""

from __future__ import annotations

import pytest
import torch

from upgrade_resolution.cli import UpscalerFactory
from upgrade_resolution.models import ModelSpec
from upgrade_resolution.upscaler import Upscaler


class NearestNetwork:
    """Upscales by repeating every pixel ``scale`` times in each direction."""

    def __init__(self, scale: int) -> None:
        self.scale = scale

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image.repeat_interleave(self.scale, dim=2).repeat_interleave(self.scale, dim=3)


@pytest.fixture
def fake_factory() -> UpscalerFactory:
    def factory(spec: ModelSpec, device: str, tile_size: int) -> Upscaler:
        return Upscaler(NearestNetwork(2), device=torch.device("cpu"), tile_size=tile_size)

    return factory

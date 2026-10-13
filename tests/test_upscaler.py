"""Tests for upgrade_resolution.upscaler.

Testing strategy for Upscaler.upscale:
    partition on scale: None (native), integer == native, 1, non-integer, <= 0
    partition on alpha: absent, present
    partition on image size vs tile size: smaller, larger
    partition on network output: plausible; black or NaN without full-precision
        fallback; black with a fallback that is plausible, or also black
Testing strategy for Upscaler construction: valid, tile_size < 1, overlap < 0.
Testing strategy for resolve_device: "auto", explicit name.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import BlackNetwork, NanNetwork, NearestNetwork
from upgrade_resolution.image_io import RasterImage
from upgrade_resolution.upscaler import CorruptOutputError, Upscaler, resolve_device

CPU = torch.device("cpu")


def random_image(height: int, width: int, *, with_alpha: bool = False) -> RasterImage:
    rng = np.random.default_rng(1)
    alpha = rng.random((height, width, 1), dtype=np.float32) if with_alpha else None
    return RasterImage(rgb=rng.random((height, width, 3), dtype=np.float32), alpha=alpha)


def test_native_scale_matches_network() -> None:
    image = random_image(6, 10)
    upscaler = Upscaler(NearestNetwork(2), device=CPU, tile_size=4, overlap=1)

    result = upscaler.upscale(image)

    assert upscaler.native_scale == 2
    assert (result.width, result.height) == (20, 12)
    expected = np.repeat(np.repeat(image.rgb, 2, axis=0), 2, axis=1)
    np.testing.assert_allclose(result.rgb, expected, atol=1e-6)


@pytest.mark.parametrize(("scale", "size"), [(1, (10, 6)), (2, (20, 12)), (1.5, (15, 9))])
def test_requested_scale_sets_output_size(scale: float, size: tuple[int, int]) -> None:
    upscaler = Upscaler(NearestNetwork(2), device=CPU, tile_size=64)

    result = upscaler.upscale(random_image(6, 10), scale=scale)

    assert (result.width, result.height) == size
    assert float(result.rgb.min()) >= 0.0 and float(result.rgb.max()) <= 1.0


def test_alpha_is_resized_with_the_image() -> None:
    upscaler = Upscaler(NearestNetwork(2), device=CPU)

    result = upscaler.upscale(random_image(5, 7, with_alpha=True), scale=3)

    assert result.alpha is not None
    assert result.alpha.shape == (15, 21, 1)


def test_input_image_is_unchanged() -> None:
    image = random_image(4, 4)
    original = image.rgb.copy()

    Upscaler(NearestNetwork(2), device=CPU).upscale(image)

    np.testing.assert_array_equal(image.rgb, original)


@pytest.mark.parametrize("scale", [0, -1.0])
def test_non_positive_scale_is_rejected(scale: float) -> None:
    with pytest.raises(ValueError):
        Upscaler(NearestNetwork(2), device=CPU).upscale(random_image(2, 2), scale=scale)


@pytest.mark.parametrize(("tile_size", "overlap"), [(0, 0), (8, -1)])
def test_invalid_configuration_is_rejected(tile_size: int, overlap: int) -> None:
    with pytest.raises(AssertionError):
        Upscaler(NearestNetwork(2), device=CPU, tile_size=tile_size, overlap=overlap)


@pytest.mark.parametrize("network", [BlackNetwork(2), NanNetwork(2)])
def test_corrupt_output_is_rejected(network: NearestNetwork) -> None:
    upscaler = Upscaler(network, device=CPU, tile_size=4)

    with pytest.raises(CorruptOutputError):
        upscaler.upscale(random_image(6, 10))


def test_corrupt_half_precision_falls_back_to_full_precision() -> None:
    calls = 0

    def full_precision() -> NearestNetwork:
        nonlocal calls
        calls += 1
        return NearestNetwork(2)

    image = random_image(6, 10)
    upscaler = Upscaler(
        BlackNetwork(2),
        device=CPU,
        dtype=torch.float16,
        tile_size=4,
        full_precision=full_precision,
    )

    result = upscaler.upscale(image)

    assert calls == 1
    expected = np.repeat(np.repeat(image.rgb, 2, axis=0), 2, axis=1)
    np.testing.assert_allclose(result.rgb, expected, atol=1e-6)


def test_corrupt_full_precision_fallback_is_rejected() -> None:
    upscaler = Upscaler(
        BlackNetwork(2),
        device=CPU,
        dtype=torch.float16,
        full_precision=lambda: BlackNetwork(2),
    )

    with pytest.raises(CorruptOutputError):
        upscaler.upscale(random_image(6, 10))


def test_resolve_device() -> None:
    assert resolve_device("cpu") == CPU
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}

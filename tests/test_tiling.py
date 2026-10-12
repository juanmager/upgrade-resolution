"""Tests for upgrade_resolution.tiling.

Testing strategy for upscale_tiled:
    partition on image size vs tile_size: smaller than one tile, exact multiple,
        not a multiple
    partition on overlap: 0, 0 < overlap < tile_size, overlap >= tile_size
    partition on scale: 1, > 1
    partition on channels: 1, 3
    invalid arguments: empty image, scale < 1, tile_size < 1, overlap < 0,
        upscale_tile returning the wrong shape
    progress callback: called once per tile, ending at (total, total)

Because nearest-neighbor upscaling is purely local, the stitched result must equal
upscaling the whole image at once; any off-by-one in cropping breaks equality.
"""

from __future__ import annotations

import numpy as np
import pytest

from upgrade_resolution.tiling import FloatImage, TileUpscaler, count_tiles, upscale_tiled


def nearest(scale: int) -> TileUpscaler:
    def upscale(tile: FloatImage) -> FloatImage:
        return np.repeat(np.repeat(tile, scale, axis=0), scale, axis=1)

    return upscale


def random_image(height: int, width: int, channels: int) -> FloatImage:
    return np.random.default_rng(0).random((height, width, channels), dtype=np.float32)


@pytest.mark.parametrize(
    ("height", "width", "channels", "scale", "tile_size", "overlap"),
    [
        (5, 7, 3, 2, 16, 4),  # smaller than one tile
        (32, 16, 3, 4, 8, 0),  # exact multiple, no overlap
        (33, 17, 1, 3, 8, 2),  # not a multiple, 1 channel
        (20, 20, 3, 1, 6, 10),  # overlap >= tile_size, scale 1
    ],
)
def test_tiled_result_equals_whole_image_result(
    height: int, width: int, channels: int, scale: int, tile_size: int, overlap: int
) -> None:
    image = random_image(height, width, channels)

    result = upscale_tiled(
        image, nearest(scale), scale=scale, tile_size=tile_size, overlap=overlap
    )

    np.testing.assert_array_equal(result, nearest(scale)(image))
    assert result.dtype == np.float32


def test_input_is_not_modified() -> None:
    image = random_image(10, 10, 3)
    original = image.copy()

    upscale_tiled(image, nearest(2), scale=2, tile_size=4, overlap=2)

    np.testing.assert_array_equal(image, original)


def test_progress_reports_every_tile() -> None:
    calls: list[tuple[int, int]] = []

    upscale_tiled(
        random_image(10, 25, 3),
        nearest(2),
        scale=2,
        tile_size=8,
        overlap=1,
        on_progress=lambda done, total: calls.append((done, total)),
    )

    total = count_tiles(10, 25, 8)
    assert total == 8
    assert calls == [(done, total) for done in range(1, total + 1)]


@pytest.mark.parametrize(
    ("image", "scale", "tile_size", "overlap"),
    [
        (np.zeros((0, 4, 3), dtype=np.float32), 2, 4, 0),
        (np.zeros((4, 4), dtype=np.float32), 2, 4, 0),
        (random_image(4, 4, 3), 0, 4, 0),
        (random_image(4, 4, 3), 2, 0, 0),
        (random_image(4, 4, 3), 2, 4, -1),
    ],
)
def test_invalid_arguments_are_rejected(
    image: FloatImage, scale: int, tile_size: int, overlap: int
) -> None:
    with pytest.raises(ValueError):
        upscale_tiled(
            image, nearest(max(scale, 1)), scale=scale, tile_size=tile_size, overlap=overlap
        )


def test_wrong_tile_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="upscale_tile"):
        upscale_tiled(random_image(4, 4, 3), nearest(3), scale=2, tile_size=4, overlap=0)

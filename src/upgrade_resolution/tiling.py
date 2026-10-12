"""Tile-based processing, so that images of any size fit in memory."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatImage = NDArray[np.float32]
TileUpscaler = Callable[[FloatImage], FloatImage]
ProgressCallback = Callable[[int, int], None]


def count_tiles(height: int, width: int, tile_size: int) -> int:
    """Return how many tiles of side ``tile_size`` cover a ``height`` x ``width`` image."""
    return math.ceil(height / tile_size) * math.ceil(width / tile_size)


def upscale_tiled(
    image: FloatImage,
    upscale_tile: TileUpscaler,
    *,
    scale: int,
    tile_size: int,
    overlap: int,
    on_progress: ProgressCallback | None = None,
) -> FloatImage:
    """Upscale ``image`` piece by piece and stitch the pieces together seamlessly.

    Each tile is processed together with up to ``overlap`` pixels of surrounding
    context on every side, and only the tile's own region of the result is kept.
    The network therefore never sees an artificial border inside the image, which
    is what would otherwise produce visible seams.

    Args:
        image: Array of shape (H, W, C) with H, W, C >= 1. It is not modified.
        upscale_tile: Function mapping an (h, w, C) array to an
            (h * scale, w * scale, C) array.
        scale: Integer upscaling factor of ``upscale_tile``; must be >= 1.
        tile_size: Side of each tile in input pixels; must be >= 1.
        overlap: Context pixels added around each tile; must be >= 0.
        on_progress: Called as ``on_progress(done, total)`` after every tile.

    Returns:
        A new float32 array of shape (H * scale, W * scale, C).

    Raises:
        ValueError: If an argument is out of range or ``upscale_tile`` returns an
            array of the wrong shape.
    """
    if image.ndim != 3 or 0 in image.shape:
        raise ValueError(f"se esperaba una imagen (H, W, C) no vacía, se recibió {image.shape}")
    if scale < 1:
        raise ValueError(f"scale debe ser >= 1, se recibió {scale}")
    if tile_size < 1:
        raise ValueError(f"tile_size debe ser >= 1, se recibió {tile_size}")
    if overlap < 0:
        raise ValueError(f"overlap debe ser >= 0, se recibió {overlap}")

    height, width, channels = image.shape
    output = np.empty((height * scale, width * scale, channels), dtype=np.float32)
    total = count_tiles(height, width, tile_size)
    done = 0

    for top in range(0, height, tile_size):
        bottom = min(top + tile_size, height)
        context_top = max(top - overlap, 0)
        context_bottom = min(bottom + overlap, height)
        for left in range(0, width, tile_size):
            right = min(left + tile_size, width)
            context_left = max(left - overlap, 0)
            context_right = min(right + overlap, width)

            context = image[context_top:context_bottom, context_left:context_right]
            upscaled = upscale_tile(context)
            expected = (context.shape[0] * scale, context.shape[1] * scale, channels)
            if upscaled.shape != expected:
                raise ValueError(f"upscale_tile devolvió {upscaled.shape}, se esperaba {expected}")

            offset_y = (top - context_top) * scale
            offset_x = (left - context_left) * scale
            tile_height = (bottom - top) * scale
            tile_width = (right - left) * scale
            output[top * scale : bottom * scale, left * scale : right * scale] = upscaled[
                offset_y : offset_y + tile_height, offset_x : offset_x + tile_width
            ]

            done += 1
            if on_progress is not None:
                on_progress(done, total)

    return output

"""Tests for upgrade_resolution.image_io.

Testing strategy:
    RasterImage:
        partition on alpha: None, valid, wrong shape
        partition on rgb shape: valid, wrong channel count, empty
        immutability: arrays are read-only after construction
    load_image:
        partition on source mode: RGB, grayscale, RGBA translucent, RGBA opaque
        partition on EXIF orientation: absent, rotated
        invalid file content
    save_image:
        partition on alpha x format: (no alpha, any), (alpha, PNG), (alpha, JPEG)
        unsupported extension
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from upgrade_resolution.image_io import RasterImage, load_image, save_image

_ORIENTATION_TAG = 0x0112
_ROTATE_90_CW = 6


def pixel_grid(height: int, width: int, channels: int) -> np.ndarray:
    values = np.arange(height * width * channels, dtype=np.uint8) * 7
    return values.reshape(height, width, channels)


def test_raster_image_arrays_are_read_only() -> None:
    image = RasterImage(
        rgb=np.zeros((2, 3, 3), dtype=np.float32), alpha=np.ones((2, 3, 1), dtype=np.float32)
    )

    assert (image.height, image.width) == (2, 3)
    assert image.rgb.dtype == np.float32
    assert image.alpha is not None
    with pytest.raises(ValueError):
        image.rgb[0, 0, 0] = 1.0
    with pytest.raises(ValueError):
        image.alpha[0, 0, 0] = 0.0


@pytest.mark.parametrize(
    ("rgb_shape", "alpha_shape"),
    [((2, 3, 4), None), ((0, 3, 3), None), ((2, 3, 3), (2, 3)), ((2, 3, 3), (3, 2, 1))],
)
def test_raster_image_rejects_bad_shapes(
    rgb_shape: tuple[int, ...], alpha_shape: tuple[int, ...] | None
) -> None:
    alpha = None if alpha_shape is None else np.ones(alpha_shape, dtype=np.float32)
    with pytest.raises(ValueError):
        RasterImage(rgb=np.zeros(rgb_shape, dtype=np.float32), alpha=alpha)


def test_rgb_png_round_trip_is_lossless(tmp_path: Path) -> None:
    pixels = pixel_grid(4, 5, 3)
    Image.fromarray(pixels).save(tmp_path / "in.png")

    image = load_image(tmp_path / "in.png")
    save_image(image, tmp_path / "out.png")

    assert image.alpha is None
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / "out.png")), pixels)


def test_grayscale_is_loaded_as_rgb(tmp_path: Path) -> None:
    Image.fromarray(pixel_grid(3, 3, 1)[..., 0]).save(tmp_path / "gray.png")

    image = load_image(tmp_path / "gray.png")

    assert image.rgb.shape == (3, 3, 3)
    np.testing.assert_array_equal(image.rgb[..., 0], image.rgb[..., 2])


def test_translucent_rgba_keeps_alpha(tmp_path: Path) -> None:
    pixels = pixel_grid(2, 2, 4)
    pixels[..., 3] = [[0, 128], [255, 255]]
    Image.fromarray(pixels).save(tmp_path / "rgba.png")

    image = load_image(tmp_path / "rgba.png")
    save_image(image, tmp_path / "out.png")

    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / "out.png")), pixels)


def test_opaque_rgba_drops_alpha(tmp_path: Path) -> None:
    pixels = pixel_grid(2, 2, 4)
    pixels[..., 3] = 255
    Image.fromarray(pixels).save(tmp_path / "opaque.png")

    assert load_image(tmp_path / "opaque.png").alpha is None


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    exif = Image.Exif()
    exif[_ORIENTATION_TAG] = _ROTATE_90_CW
    Image.new("RGB", (4, 2)).save(tmp_path / "rotated.jpg", exif=exif)

    image = load_image(tmp_path / "rotated.jpg")

    assert (image.width, image.height) == (2, 4)


def test_alpha_is_flattened_on_white_for_jpeg(tmp_path: Path) -> None:
    transparent_black = RasterImage(
        rgb=np.zeros((8, 8, 3), dtype=np.float32), alpha=np.zeros((8, 8, 1), dtype=np.float32)
    )

    save_image(transparent_black, tmp_path / "out.jpg")

    saved = np.asarray(Image.open(tmp_path / "out.jpg"))
    assert saved.shape == (8, 8, 3)
    assert saved.min() >= 250


def test_save_creates_missing_directories(tmp_path: Path) -> None:
    destination = tmp_path / "a" / "b" / "out.webp"

    save_image(RasterImage(rgb=np.zeros((2, 2, 3), dtype=np.float32)), destination)

    assert destination.is_file()


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no soportado"):
        save_image(RasterImage(rgb=np.zeros((2, 2, 3), dtype=np.float32)), tmp_path / "out.gif")


def test_invalid_file_raises_os_error(tmp_path: Path) -> None:
    (tmp_path / "broken.png").write_bytes(b"not an image")

    with pytest.raises(OSError):
        load_image(tmp_path / "broken.png")

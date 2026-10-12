"""Reading and writing images as immutable float arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from PIL import Image, ImageOps

from upgrade_resolution.tiling import FloatImage

SUPPORTED_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})
_ALPHA_SUFFIXES = frozenset({".png", ".webp", ".tif", ".tiff"})
_LOSSY_QUALITY = 95


@dataclass(frozen=True, eq=False)
class RasterImage:
    """An immutable raster image whose pixel values are floats in [0, 1].

    Attributes:
        rgb: Color channels, shape (H, W, 3).
        alpha: Opacity channel, shape (H, W, 1), or None for a fully opaque image.

    Rep invariant:
        rgb is a float32 array of shape (H, W, 3) with H, W >= 1;
        alpha is None or a float32 array of shape (H, W, 1);
        both arrays are read-only.

    Safety from rep exposure:
        The constructor takes ownership of the arrays it receives and marks them
        read-only, so no client can mutate an image after it has been created.
    """

    rgb: FloatImage
    alpha: FloatImage | None = None

    def __post_init__(self) -> None:
        """Freeze the arrays and check the rep invariant."""
        rgb = _read_only(self.rgb)
        alpha = None if self.alpha is None else _read_only(self.alpha)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or 0 in rgb.shape:
            raise ValueError(f"rgb debe tener forma (H, W, 3) no vacía, tiene {rgb.shape}")
        if alpha is not None and alpha.shape != (*rgb.shape[:2], 1):
            raise ValueError(f"alpha debe tener forma {(*rgb.shape[:2], 1)}, tiene {alpha.shape}")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "alpha", alpha)

    @property
    def height(self) -> int:
        """Number of pixel rows."""
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        """Number of pixel columns."""
        return int(self.rgb.shape[1])


def load_image(path: Path) -> RasterImage:
    """Read an image file, applying its EXIF orientation.

    Grayscale and palette images are converted to RGB. The alpha channel is kept
    only if some pixel is actually translucent, so that opaque screenshots saved
    as RGBA are not processed twice.

    Args:
        path: Image file in any format Pillow can read.

    Returns:
        The decoded image.

    Raises:
        OSError: If the file cannot be read or is not a valid image.
    """
    with Image.open(path) as opened:
        oriented = ImageOps.exif_transpose(opened)
        if _has_alpha(oriented):
            rgba = _to_float(oriented.convert("RGBA"))
            alpha = rgba[..., 3:]
            if float(alpha.min()) < 1.0:
                return RasterImage(rgb=rgba[..., :3], alpha=alpha)
            return RasterImage(rgb=rgba[..., :3])
        return RasterImage(rgb=_to_float(oriented.convert("RGB")))


def save_image(image: RasterImage, path: Path) -> None:
    """Write ``image`` to ``path``, choosing the format from its extension.

    Formats without transparency (JPEG, BMP) receive the image flattened onto a
    white background. Lossy formats are written at high quality. Missing parent
    directories are created.

    Args:
        image: Image to write.
        path: Destination; its extension must be in ``SUPPORTED_SUFFIXES``.

    Raises:
        ValueError: If the extension is not supported.
        OSError: If the file cannot be written.
    """
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"formato de salida no soportado: {suffix or '(sin extensión)'}")

    if image.alpha is None:
        pixels = image.rgb
    elif suffix in _ALPHA_SUFFIXES:
        pixels = np.concatenate([image.rgb, image.alpha], axis=2)
    else:
        pixels = image.rgb * image.alpha + (1.0 - image.alpha)

    options: dict[str, Any] = {}
    if suffix in {".jpg", ".jpeg"}:
        options = {"quality": _LOSSY_QUALITY, "subsampling": 0}
    elif suffix == ".webp":
        options = {"quality": _LOSSY_QUALITY}

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(pixels)).save(path, **options)


def _has_alpha(image: Image.Image) -> bool:
    return image.mode in {"RGBA", "LA", "PA"} or (
        image.mode == "P" and "transparency" in image.info
    )


def _to_float(image: Image.Image) -> FloatImage:
    return np.asarray(image, dtype=np.float32) / np.float32(255.0)


def _to_uint8(pixels: FloatImage) -> np.ndarray[Any, np.dtype[np.uint8]]:
    return np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)


def _read_only(array: ArrayLike) -> FloatImage:
    result = np.asarray(array, dtype=np.float32)
    result.flags.writeable = False
    return result

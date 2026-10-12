"""Registry of super-resolution models and management of their weight files."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

_CACHE_ENV_VAR = "UPGRADE_RESOLUTION_CACHE"
_CHUNK_SIZE = 1 << 20  # 1 MiB
_DOWNLOAD_TIMEOUT_SECONDS = 60
_RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"


class WeightsIntegrityError(RuntimeError):
    """Raised when a downloaded weights file does not match its expected digest."""


@dataclass(frozen=True)
class ModelSpec:
    """Immutable description of a downloadable super-resolution model.

    Attributes:
        name: Short identifier used on the command line.
        url: HTTPS location of the weights file.
        sha256: Expected hexadecimal SHA-256 digest of the weights file.
        scale: Native upscaling factor of the network.
        description: One-line, human-readable summary.
    """

    name: str
    url: str
    sha256: str
    scale: int
    description: str

    @property
    def filename(self) -> str:
        """Name of the weights file, taken from the last segment of the URL."""
        return self.url.rsplit("/", 1)[-1]


MODELS: Mapping[str, ModelSpec] = MappingProxyType(
    {
        spec.name: spec
        for spec in (
            ModelSpec(
                name="photo",
                url=f"{_RELEASES}/v0.1.0/RealESRGAN_x4plus.pth",
                sha256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
                scale=4,
                description="máxima calidad para fotos reales",
            ),
            ModelSpec(
                name="general",
                url=f"{_RELEASES}/v0.2.5.0/realesr-general-x4v3.pth",
                sha256="8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
                scale=4,
                description="rápido y equilibrado para cualquier imagen",
            ),
            ModelSpec(
                name="anime",
                url=f"{_RELEASES}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                sha256="f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
                scale=4,
                description="ilustraciones, dibujos y capturas con colores planos o texto",
            ),
        )
    }
)

DEFAULT_MODEL = "photo"


def default_cache_dir() -> Path:
    """Return the directory where weights are stored.

    Returns:
        The path in the ``UPGRADE_RESOLUTION_CACHE`` environment variable if it is
        set and non-empty, otherwise ``~/.cache/upgrade_resolution``.
    """
    override = os.environ.get(_CACHE_ENV_VAR)
    if override:
        return Path(override)
    return Path.home() / ".cache" / "upgrade_resolution"


def is_cached(spec: ModelSpec, cache_dir: Path | None = None) -> bool:
    """Tell whether the weights of ``spec`` are already present in the cache."""
    return (_resolve(cache_dir) / spec.filename).is_file()


def ensure_weights(spec: ModelSpec, cache_dir: Path | None = None) -> Path:
    """Return the local path of the weights of ``spec``, downloading them if needed.

    The download is written to a temporary ``.part`` file and only moved into place
    after its digest has been verified, so the cache never holds a partial or
    tampered file.

    Args:
        spec: Model whose weights are required.
        cache_dir: Directory used as cache; defaults to :func:`default_cache_dir`.

    Returns:
        Path to a weights file whose name is ``spec.filename``.

    Raises:
        WeightsIntegrityError: If the downloaded file's digest differs from ``spec.sha256``.
        OSError: If the file cannot be downloaded or written.
    """
    directory = _resolve(cache_dir)
    target = directory / spec.filename
    if target.is_file():
        return target

    directory.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    try:
        digest = _download(spec.url, partial)
        if digest != spec.sha256:
            raise WeightsIntegrityError(
                f"{spec.filename}: SHA-256 esperado {spec.sha256}, obtenido {digest}"
            )
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def _resolve(cache_dir: Path | None) -> Path:
    return default_cache_dir() if cache_dir is None else cache_dir


def _download(url: str, destination: Path) -> str:
    """Stream ``url`` into ``destination`` and return the SHA-256 of the content."""
    hasher = hashlib.sha256()
    with (
        urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response,
        destination.open("wb") as output,
    ):
        while chunk := response.read(_CHUNK_SIZE):
            hasher.update(chunk)
            output.write(chunk)
    return hasher.hexdigest()

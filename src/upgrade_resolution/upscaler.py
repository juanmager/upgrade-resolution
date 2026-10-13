"""Super-resolution engine: runs a neural network over an image, tile by tile."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import spandrel
import torch
import torch.nn.functional as F  # noqa: N812 - conventional alias

from upgrade_resolution.image_io import RasterImage
from upgrade_resolution.tiling import FloatImage, ProgressCallback, upscale_tiled

DEFAULT_TILE_SIZE = 512
DEFAULT_TILE_OVERLAP = 16

# A healthy network's output, averaged back down to the input size, differs from the
# input by about 0.02 per channel; a broken GPU kernel's black or garbled output
# differs by 0.3 or more. The threshold sits between them, with room for noisy
# photos, whose noise the network removes.
MAX_TILE_DEVIATION = 0.15


class CorruptOutputError(RuntimeError):
    """The network produced an output that cannot be a version of its input."""


class SuperResolutionNetwork(Protocol):
    """Anything that upscales an image tensor by a fixed integer factor."""

    @property
    def scale(self) -> int:
        """Integer upscaling factor."""
        ...

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Map a (1, 3, H, W) tensor in [0, 1] to a (1, 3, H*scale, W*scale) tensor."""
        ...


def resolve_device(name: str) -> torch.device:
    """Translate a device name into a torch device.

    Args:
        name: ``"auto"`` or any name torch accepts (``"cpu"``, ``"cuda"``, ``"cuda:1"``...).
            ``"auto"`` picks the first available of GPU (CUDA, or ROCm on AMD cards,
            which PyTorch also exposes as ``cuda``), Apple MPS, and CPU.

    Returns:
        The selected device.

    Raises:
        RuntimeError: If ``name`` is not a valid torch device name.
    """
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Return a human-readable name for ``device``, e.g. ``"GPU AMD Radeon RX 7700 XT"``."""
    if device.type == "cuda":
        return f"GPU {torch.cuda.get_device_name(device)}"
    if device.type == "mps":
        return "GPU Apple (MPS)"
    return "CPU"


class Upscaler:
    """Enhances images and raises their resolution with a neural network.

    Abstraction function:
        AF(network, device, dtype, tile_size, overlap) = the image transformation that
        runs ``network`` on ``device`` in ``dtype`` precision over tiles of
        ``tile_size`` pixels with ``overlap`` pixels of context, then resamples the
        result to the requested scale.

    Rep invariant:
        network.scale >= 1, tile_size >= 1, overlap >= 0;
        full_precision is None if dtype is float32.

    Safety from rep exposure:
        All fields are private and never returned; ``device`` returns an immutable
        torch.device. Inputs and outputs are RasterImage values, which are immutable.
    """

    def __init__(
        self,
        network: SuperResolutionNetwork,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        tile_size: int = DEFAULT_TILE_SIZE,
        overlap: int = DEFAULT_TILE_OVERLAP,
        full_precision: Callable[[], SuperResolutionNetwork] | None = None,
    ) -> None:
        """Wrap ``network``, which must already live on ``device`` in ``dtype``.

        Args:
            network: Network that enlarges each tile.
            device: Device on which ``network`` lives.
            dtype: Precision of ``network``'s parameters.
            tile_size: Side of each tile in input pixels; must be >= 1.
            overlap: Context pixels added around each tile; must be >= 0.
            full_precision: For a reduced-precision ``network``, returns the same
                network in float32. It is called at most once, if a tile comes out
                corrupt, and every later tile then runs in float32.
        """
        self._network = network
        self._device = device
        self._dtype = dtype
        self._tile_size = tile_size
        self._overlap = overlap
        self._full_precision = full_precision
        self._check_rep()

    @classmethod
    def from_weights(
        cls,
        weights: Path,
        *,
        device: str = "auto",
        tile_size: int = DEFAULT_TILE_SIZE,
        overlap: int = DEFAULT_TILE_OVERLAP,
    ) -> Upscaler:
        """Load a network from a weights file.

        Half precision is used on GPUs when the architecture supports it: it halves
        memory use and roughly doubles speed with no visible loss of quality.

        Raises:
            ValueError: If the file does not contain a single-image model.
            OSError: If the file cannot be read.
        """
        torch_device = resolve_device(device)
        descriptor = spandrel.ModelLoader(device=torch_device).load_from_file(weights)
        if not isinstance(descriptor, spandrel.ImageModelDescriptor):
            raise ValueError(f"{weights.name} no contiene un modelo de imagen compatible")
        descriptor.eval()
        dtype = torch.float32
        full_precision = None
        if torch_device.type == "cuda" and descriptor.supports_half:
            descriptor.half()
            dtype = torch.float16
            full_precision = descriptor.float
        return cls(
            descriptor,
            device=torch_device,
            dtype=dtype,
            tile_size=tile_size,
            overlap=overlap,
            full_precision=full_precision,
        )

    @property
    def native_scale(self) -> int:
        """Upscaling factor the network produces by itself."""
        return self._network.scale

    @property
    def device(self) -> torch.device:
        """Device on which the network runs."""
        return self._device

    def upscale(
        self,
        image: RasterImage,
        *,
        scale: float | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> RasterImage:
        """Return an enhanced copy of ``image`` enlarged by ``scale``.

        Colors are reconstructed by the network at its native scale and then
        resampled to the requested one, so any scale benefits from the network:
        ``scale=1`` keeps the size and only removes noise, blur and compression
        artifacts. Transparency is resampled with bicubic interpolation.

        Args:
            image: Image to enhance; it is not modified.
            scale: Output size factor, > 0. Defaults to ``native_scale``.
            on_progress: Called as ``on_progress(done, total)`` after every tile.

        Returns:
            An image of round(width * scale) x round(height * scale) pixels (at least 1x1).

        Raises:
            ValueError: If ``scale`` is not positive.
            CorruptOutputError: If the network returns a tile that is not a plausible
                version of its input (e.g. black), even in full precision.
        """
        target_scale = self.native_scale if scale is None else scale
        if target_scale <= 0:
            raise ValueError(f"scale debe ser > 0, se recibió {target_scale}")

        enhanced = upscale_tiled(
            image.rgb,
            self._run_network,
            scale=self.native_scale,
            tile_size=self._tile_size,
            overlap=self._overlap,
            on_progress=on_progress,
        )
        height = max(1, round(image.height * target_scale))
        width = max(1, round(image.width * target_scale))
        alpha = None if image.alpha is None else _resize(image.alpha, height, width)
        return RasterImage(rgb=_resize(enhanced, height, width), alpha=alpha)

    def _run_network(self, tile: FloatImage) -> FloatImage:
        output = self._infer(tile)
        if _is_plausible(tile, output, self.native_scale):
            return output
        if self._full_precision is not None:
            self._network = self._full_precision()
            self._dtype = torch.float32
            self._full_precision = None
            self._check_rep()
            output = self._infer(tile)
            if _is_plausible(tile, output, self.native_scale):
                return output
        raise CorruptOutputError(
            f"la red devolvió un resultado corrupto en {describe_device(self._device)}; "
            "prueba otra vez o usa --device cpu"
        )

    def _infer(self, tile: FloatImage) -> FloatImage:
        batch = _to_tensor(tile).to(device=self._device, dtype=self._dtype)
        with torch.inference_mode():
            output = self._network(batch)
        return _to_array(output.float().cpu())

    def _check_rep(self) -> None:
        assert self._network.scale >= 1, "network.scale debe ser >= 1"
        assert self._tile_size >= 1, "tile_size debe ser >= 1"
        assert self._overlap >= 0, "overlap debe ser >= 0"
        assert self._full_precision is None or self._dtype != torch.float32, (
            "full_precision solo tiene sentido con dtype reducido"
        )


def _is_plausible(tile: FloatImage, output: FloatImage, scale: int) -> bool:
    """Return whether ``output`` could be ``tile`` enlarged ``scale`` times.

    The output is averaged over each ``scale`` x ``scale`` block, which undoes the
    enlargement, and compared with the tile: a genuine enhancement only changes
    fine detail, so the mean absolute difference stays small.
    """
    if not np.isfinite(output).all():
        return False
    height, width, channels = tile.shape
    reduced = output.reshape(height, scale, width, scale, channels).mean(axis=(1, 3))
    return float(np.abs(reduced - tile).mean()) <= MAX_TILE_DEVIATION


def _resize(image: FloatImage, height: int, width: int) -> FloatImage:
    """Resample an (H, W, C) image to (height, width, C) with antialiased bicubic."""
    if image.shape[:2] == (height, width):
        return image
    resized = F.interpolate(
        _to_tensor(image),
        size=(height, width),
        mode="bicubic",
        antialias=True,
        align_corners=False,
    )
    return _to_array(resized)


def _to_tensor(image: FloatImage) -> torch.Tensor:
    """Convert an (H, W, C) array into a fresh (1, C, H, W) float32 tensor."""
    return torch.from_numpy(np.array(image.transpose(2, 0, 1), dtype=np.float32)).unsqueeze(0)


def _to_array(tensor: torch.Tensor) -> FloatImage:
    """Convert a (1, C, H, W) CPU tensor into an (H, W, C) float32 array clipped to [0, 1]."""
    return np.ascontiguousarray(tensor.squeeze(0).clamp(0.0, 1.0).permute(1, 2, 0).numpy())

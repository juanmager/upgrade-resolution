"""Command-line interface: ``upgrade-resolution IMAGEN... [opciones]``."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from upgrade_resolution.image_io import SUPPORTED_SUFFIXES, load_image, save_image
from upgrade_resolution.models import DEFAULT_MODEL, MODELS, ModelSpec, ensure_weights, is_cached
from upgrade_resolution.upscaler import DEFAULT_TILE_SIZE, Upscaler, describe_device

UpscalerFactory = Callable[[ModelSpec, str, int], Upscaler]

EXIT_OK = 0
EXIT_SOME_FAILED = 1
EXIT_USAGE = 2

OUTPUT_FORMATS = ("png", "jpg", "webp")
DEFAULT_OUTPUT_SUBDIR = "upscaled"
_MAX_SCALE = 16.0


class InputError(ValueError):
    """Raised when a command-line input path cannot be used."""


def collect_inputs(paths: Sequence[Path]) -> list[Path]:
    """Expand files and directories into the list of images to process.

    Directories contribute the supported images directly inside them (not in
    subdirectories), sorted by name. Repeated paths appear only once.

    Raises:
        InputError: If a path does not exist, or is a file with an unsupported extension.
    """
    found: dict[Path, None] = {}
    for path in paths:
        if path.is_dir():
            images = (p for p in sorted(path.iterdir()) if _is_supported_file(p))
            found.update(dict.fromkeys(images))
        elif path.is_file():
            if not _is_supported_file(path):
                raise InputError(f"formato no soportado: {path}")
            found[path] = None
        else:
            raise InputError(f"no existe: {path}")
    return list(found)


def output_path(source: Path, *, output_dir: Path | None, scale: float, extension: str) -> Path:
    """Return where the result for ``source`` is written.

    The name is ``<stem>_x<scale>.<extension>``, placed in ``output_dir`` or, if it
    is None, in an ``upscaled`` folder next to ``source`` so results are never
    mistaken for inputs when a folder is processed again.
    """
    directory = source.parent / DEFAULT_OUTPUT_SUBDIR if output_dir is None else output_dir
    return directory / f"{source.stem}_x{scale:g}.{extension}"


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser of the command."""
    models_help = "; ".join(f"{spec.name}: {spec.description}" for spec in MODELS.values())
    parser = argparse.ArgumentParser(
        prog="upgrade-resolution",
        description="Mejora la calidad y la resolución de imágenes con IA (Real-ESRGAN).",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        metavar="IMAGEN",
        help="archivos o carpetas con imágenes (png, jpg, webp, bmp, tif)",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=sorted(MODELS),
        default=DEFAULT_MODEL,
        help=f"modelo de IA (por defecto: {DEFAULT_MODEL}). {models_help}",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=_scale_argument,
        default=None,
        help="factor de aumento, p. ej. 2 o 1.5; 1 solo mejora la calidad "
        "(por defecto: el nativo del modelo, 4)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=f"carpeta de salida (por defecto: '{DEFAULT_OUTPUT_SUBDIR}' junto a cada imagen)",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=OUTPUT_FORMATS,
        default="png",
        help="formato de salida (por defecto: png, sin pérdida)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda... (por defecto: auto, usa la GPU si existe)",
    )
    parser.add_argument(
        "--tile",
        type=_positive_int,
        default=DEFAULT_TILE_SIZE,
        help=f"tamaño de bloque en píxeles (por defecto: {DEFAULT_TILE_SIZE}); "
        "redúcelo si falta memoria de GPU",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="reemplazar resultados existentes en lugar de saltarlos",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    upscaler_factory: UpscalerFactory | None = None,
) -> int:
    """Run the command and return its exit status.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.
        upscaler_factory: Builds the upscaler from (model, device, tile size).
            Defaults to downloading the weights and loading the real network.

    Returns:
        0 if every image was processed or skipped, 1 if some failed,
        2 on invalid inputs.
    """
    args = build_parser().parse_args(argv)
    factory = upscaler_factory or load_upscaler

    try:
        inputs = collect_inputs(args.inputs)
    except InputError as error:
        _report(f"Error: {error}")
        return EXIT_USAGE
    if not inputs:
        _report("No se encontraron imágenes para procesar.")
        return EXIT_USAGE

    spec = MODELS[args.model]
    try:
        upscaler = factory(spec, args.device, args.tile)
    except (OSError, RuntimeError, ValueError) as error:
        _report(f"Error al cargar el modelo '{spec.name}': {error}")
        return EXIT_SOME_FAILED

    scale = upscaler.native_scale if args.scale is None else args.scale
    _report(
        f"Modelo: {spec.name} | Dispositivo: {describe_device(upscaler.device)} "
        f"| Escala: x{scale:g}"
    )

    failures = 0
    for index, source in enumerate(inputs, start=1):
        destination = output_path(
            source, output_dir=args.output_dir, scale=scale, extension=args.format
        )
        prefix = f"[{index}/{len(inputs)}] {source.name}"
        if destination.exists() and not args.overwrite:
            _report(f"{prefix}: ya existe {destination}, se omite (usa --overwrite)")
            continue
        if not _process(upscaler, source, destination, scale, prefix):
            failures += 1

    return EXIT_SOME_FAILED if failures else EXIT_OK


def load_upscaler(spec: ModelSpec, device: str, tile_size: int) -> Upscaler:
    """Download the weights of ``spec`` if needed and load the network."""
    if not is_cached(spec):
        _report(f"Descargando el modelo '{spec.name}' (solo la primera vez)...")
    return Upscaler.from_weights(ensure_weights(spec), device=device, tile_size=tile_size)


def _process(
    upscaler: Upscaler, source: Path, destination: Path, scale: float, prefix: str
) -> bool:
    """Enhance one image; report the outcome and return whether it succeeded."""
    started = time.perf_counter()
    try:
        image = load_image(source)
        result = upscaler.upscale(
            image,
            scale=scale,
            on_progress=lambda done, total: _report_progress(prefix, done, total),
        )
        save_image(result, destination)
    except torch.cuda.OutOfMemoryError:
        _report(f"\n{prefix}: memoria de GPU insuficiente; prueba con un --tile más pequeño")
        return False
    except (OSError, ValueError) as error:
        _report(f"\n{prefix}: error: {error}")
        return False
    elapsed = time.perf_counter() - started
    _report(
        f"\r{prefix}: {image.width}x{image.height} -> {result.width}x{result.height} "
        f"en {elapsed:.1f} s -> {destination}"
    )
    return True


def _report_progress(prefix: str, done: int, total: int) -> None:
    print(f"\r{prefix}: bloque {done}/{total}", end="", file=sys.stderr, flush=True)


def _report(message: str) -> None:
    print(message, file=sys.stderr)


def _is_supported_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES


def _scale_argument(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"no es un número: {text}") from None
    if not 0 < value <= _MAX_SCALE:
        raise argparse.ArgumentTypeError(f"debe estar entre 0 (excluido) y {_MAX_SCALE:g}")
    return value


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"no es un entero: {text}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("debe ser >= 1")
    return value

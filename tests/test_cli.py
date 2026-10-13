"""Tests for upgrade_resolution.cli.

Testing strategy for main:
    partition on inputs: single file, directory (mixed content), missing path,
        unsupported file, empty directory
    partition on destination: absent, present without --overwrite, present with it
    partition on image validity: all valid, some corrupt
    partition on network output: plausible, corrupt
    partition on options: defaults, --scale, --output-dir, --format, invalid --scale,
        --size (image wider / taller than the box), invalid --size, --size with --scale
Testing strategy for output_path: with and without output_dir.
Testing strategy for fit_scale: limited by width, by height; enlarging, reducing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import BlackNetwork
from upgrade_resolution.cli import (
    EXIT_OK,
    EXIT_SOME_FAILED,
    EXIT_USAGE,
    UpscalerFactory,
    fit_scale,
    main,
    output_path,
)
from upgrade_resolution.models import ModelSpec
from upgrade_resolution.upscaler import Upscaler


def write_image(path: Path, width: int = 6, height: int = 4) -> Path:
    Image.fromarray(np.full((height, width, 3), 120, dtype=np.uint8)).save(path)
    return path


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def test_output_path() -> None:
    source = Path("fotos") / "playa.jpg"

    assert output_path(source, output_dir=None, tag="x4", extension="png") == (
        Path("fotos") / "upscaled" / "playa_x4.png"
    )
    assert output_path(source, output_dir=Path("out"), tag="3840x2160", extension="webp") == (
        Path("out") / "playa_3840x2160.webp"
    )


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((1920, 1080), 2.0),
        ((1917, 815), 3840 / 1917),
        ((1000, 2000), 1.08),
        ((8000, 4500), 0.48),
    ],
)
def test_fit_scale(size: tuple[int, int], expected: float) -> None:
    assert fit_scale(*size, (3840, 2160)) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [(6, 4, (30, 20)), (4, 6, (13, 20))],
)
def test_size_fits_image_inside_box(
    tmp_path: Path,
    fake_factory: UpscalerFactory,
    width: int,
    height: int,
    expected: tuple[int, int],
) -> None:
    source = write_image(tmp_path / "fondo.png", width=width, height=height)

    status = main([str(source), "--size", "30x20"], upscaler_factory=fake_factory)

    assert status == EXIT_OK
    assert image_size(tmp_path / "upscaled" / "fondo_30x20.png") == expected


@pytest.mark.parametrize(
    "arguments",
    [["--size", "30"], ["--size", "0x20"], ["--size", "axb"], ["--size", "30x20", "-s", "2"]],
)
def test_invalid_size_is_rejected(arguments: list[str], fake_factory: UpscalerFactory) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["x.png", *arguments], upscaler_factory=fake_factory)
    assert exit_info.value.code == EXIT_USAGE


def test_single_file_with_defaults(tmp_path: Path, fake_factory: UpscalerFactory) -> None:
    source = write_image(tmp_path / "captura.png")

    status = main([str(source)], upscaler_factory=fake_factory)

    assert status == EXIT_OK
    assert image_size(tmp_path / "upscaled" / "captura_x2.png") == (12, 8)


def test_directory_with_options(tmp_path: Path, fake_factory: UpscalerFactory) -> None:
    write_image(tmp_path / "a.jpg")
    write_image(tmp_path / "b.PNG")
    (tmp_path / "notas.txt").write_text("no es imagen")
    out = tmp_path / "salida"

    status = main(
        [str(tmp_path), "-s", "3", "-o", str(out), "-f", "webp"], upscaler_factory=fake_factory
    )

    assert status == EXIT_OK
    assert sorted(p.name for p in out.iterdir()) == ["a_x3.webp", "b_x3.webp"]
    assert image_size(out / "a_x3.webp") == (18, 12)


def test_existing_result_is_skipped_unless_overwrite(
    tmp_path: Path, fake_factory: UpscalerFactory
) -> None:
    source = write_image(tmp_path / "foto.png")
    destination = tmp_path / "upscaled" / "foto_x2.png"
    destination.parent.mkdir()
    destination.write_bytes(b"previous")

    assert main([str(source)], upscaler_factory=fake_factory) == EXIT_OK
    assert destination.read_bytes() == b"previous"

    assert main([str(source), "--overwrite"], upscaler_factory=fake_factory) == EXIT_OK
    assert image_size(destination) == (12, 8)


def test_corrupt_image_fails_but_others_continue(
    tmp_path: Path, fake_factory: UpscalerFactory
) -> None:
    (tmp_path / "rota.png").write_bytes(b"not an image")
    write_image(tmp_path / "sana.png")

    status = main([str(tmp_path)], upscaler_factory=fake_factory)

    assert status == EXIT_SOME_FAILED
    assert (tmp_path / "upscaled" / "sana_x2.png").is_file()


def test_corrupt_network_output_is_not_saved(tmp_path: Path) -> None:
    def black_factory(spec: ModelSpec, device: str, tile_size: int) -> Upscaler:
        return Upscaler(BlackNetwork(2), device=torch.device("cpu"), tile_size=tile_size)

    source = write_image(tmp_path / "foto.png")

    status = main([str(source)], upscaler_factory=black_factory)

    assert status == EXIT_SOME_FAILED
    assert not (tmp_path / "upscaled" / "foto_x2.png").exists()


@pytest.mark.parametrize("name", ["no_existe.png", "documento.txt", "vacia"])
def test_unusable_inputs_are_usage_errors(
    tmp_path: Path, fake_factory: UpscalerFactory, name: str
) -> None:
    if name == "documento.txt":
        (tmp_path / name).write_text("x")
    elif name == "vacia":
        (tmp_path / name).mkdir()

    assert main([str(tmp_path / name)], upscaler_factory=fake_factory) == EXIT_USAGE


@pytest.mark.parametrize("scale", ["0", "-2", "abc", "100"])
def test_invalid_scale_is_rejected(scale: str, fake_factory: UpscalerFactory) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["x.png", "--scale", scale], upscaler_factory=fake_factory)
    assert exit_info.value.code == EXIT_USAGE

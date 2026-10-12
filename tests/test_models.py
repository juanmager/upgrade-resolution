"""Tests for upgrade_resolution.models.

Testing strategy:
    MODELS registry: keys match names, fields are well formed
    ensure_weights:
        partition on cache state: hit (no network), miss
        partition on downloaded digest: matches, differs
    default_cache_dir: environment variable set, unset
"""

from __future__ import annotations

import hashlib
import io
import re
import urllib.request
from pathlib import Path

import pytest

from upgrade_resolution.models import (
    DEFAULT_MODEL,
    MODELS,
    ModelSpec,
    WeightsIntegrityError,
    default_cache_dir,
    ensure_weights,
    is_cached,
)

CONTENT = b"fake weights"


def spec_with_digest(digest: str) -> ModelSpec:
    return ModelSpec(
        name="fake", url="https://example.com/fake.pth", sha256=digest, scale=2, description="fake"
    )


def serve(monkeypatch: pytest.MonkeyPatch, content: bytes) -> list[str]:
    requested: list[str] = []

    def fake_urlopen(url: str, timeout: float) -> io.BytesIO:
        requested.append(url)
        return io.BytesIO(content)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return requested


def test_registry_is_well_formed() -> None:
    assert DEFAULT_MODEL in MODELS
    for name, spec in MODELS.items():
        assert spec.name == name
        assert spec.url.startswith("https://")
        assert spec.filename.endswith(".pth")
        assert re.fullmatch(r"[0-9a-f]{64}", spec.sha256)
        assert spec.scale >= 1


def test_cache_hit_does_not_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = spec_with_digest("unused")
    (tmp_path / spec.filename).write_bytes(CONTENT)
    requested = serve(monkeypatch, CONTENT)

    assert ensure_weights(spec, tmp_path) == tmp_path / spec.filename
    assert requested == []


def test_cache_miss_downloads_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = spec_with_digest(hashlib.sha256(CONTENT).hexdigest())
    requested = serve(monkeypatch, CONTENT)

    path = ensure_weights(spec, tmp_path / "cache")

    assert requested == [spec.url]
    assert path.read_bytes() == CONTENT
    assert is_cached(spec, tmp_path / "cache")


def test_digest_mismatch_leaves_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = spec_with_digest("0" * 64)
    serve(monkeypatch, CONTENT)

    with pytest.raises(WeightsIntegrityError):
        ensure_weights(spec, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_default_cache_dir_honors_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UPGRADE_RESOLUTION_CACHE", str(tmp_path))
    assert default_cache_dir() == tmp_path

    monkeypatch.delenv("UPGRADE_RESOLUTION_CACHE")
    assert default_cache_dir() == Path.home() / ".cache" / "upgrade_resolution"

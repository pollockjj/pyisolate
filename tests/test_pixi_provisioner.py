"""Tests for pixi binary auto-provisioner."""

from __future__ import annotations

import hashlib
import io
import os
import platform
import tarfile
import tempfile
import zipfile
from contextlib import closing
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from pyisolate._internal.pixi_provisioner import (
    PIXI_VERSION,
    _archive_extension,
    _binary_name,
    _get_target,
    ensure_pixi,
)


class TestGetTarget:
    def test_linux_x86_64(self) -> None:
        with patch("platform.system", return_value="Linux"), patch("platform.machine", return_value="x86_64"):
            assert _get_target() == "x86_64-unknown-linux-musl"

    def test_unsupported_raises(self) -> None:
        with (
            patch("platform.system", return_value="FreeBSD"),
            patch("platform.machine", return_value="sparc"),
            pytest.raises(RuntimeError, match="Unsupported platform"),
        ):
            _get_target()


class TestEnsurePixi:
    def test_downloads_and_caches(self, tmp_path: Any) -> None:
        """Full download path: fetch, verify, extract, cache."""
        version = PIXI_VERSION
        cache = tmp_path / "pyisolate" / "pixi" / version

        # Create a real archive with a fake pixi binary.
        fake_binary = b"#!/bin/sh\necho pixi"
        binary_name = _binary_name()
        archive_extension = _archive_extension()
        with closing(tempfile.NamedTemporaryFile(suffix=archive_extension, delete=False)) as archive_buf:
            archive_path = Path(archive_buf.name)
        if platform.system() == "Windows":
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr(binary_name, fake_binary)
        else:
            with tarfile.open(archive_path, "w:gz") as tf:
                info = tarfile.TarInfo(name=binary_name)
                info.size = len(fake_binary)
                info.mode = 0o755
                tf.addfile(info, io.BytesIO(fake_binary))

        archive_data = archive_path.read_bytes()
        os.unlink(archive_path)

        archive_hash = hashlib.sha256(archive_data).hexdigest()
        archive_name = f"pixi{archive_extension}"

        with (
            patch("shutil.which", return_value=None),
            patch("pyisolate._internal.pixi_provisioner._cache_dir", return_value=cache),
            patch("pyisolate._internal.pixi_provisioner._fetch_url") as fetch_mock,
        ):
            fetch_mock.side_effect = [
                f"{archive_hash}  {archive_name}".encode(),
                archive_data,
            ]
            result = ensure_pixi(version)
            assert Path(result).exists()
            assert Path(result).read_bytes() == fake_binary
            print(f"RESOLVED_PATH={result}")

    def test_path_traversal_member_is_safely_flattened(self, tmp_path: Any) -> None:
        """Path traversal entries must still extract only to the cache binary path."""
        version = PIXI_VERSION
        cache = tmp_path / "pyisolate" / "pixi" / version

        fake_binary = b"#!/bin/sh\necho pixi"
        archive_name = f"pixi{_archive_extension()}"
        with closing(tempfile.NamedTemporaryFile(suffix=_archive_extension(), delete=False)) as archive_buf:
            archive_path = Path(archive_buf.name)
        traversal_member = f"../{_binary_name()}"
        if platform.system() == "Windows":
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr(traversal_member, fake_binary)
        else:
            with tarfile.open(archive_path, "w:gz") as tf:
                info = tarfile.TarInfo(name=traversal_member)
                info.size = len(fake_binary)
                info.mode = 0o755
                tf.addfile(info, io.BytesIO(fake_binary))

        archive_data = archive_path.read_bytes()
        os.unlink(archive_path)
        archive_hash = hashlib.sha256(archive_data).hexdigest()

        with (
            patch("shutil.which", return_value=None),
            patch("pyisolate._internal.pixi_provisioner._cache_dir", return_value=cache),
            patch("pyisolate._internal.pixi_provisioner._fetch_url") as fetch_mock,
        ):
            fetch_mock.side_effect = [
                f"{archive_hash}  {archive_name}".encode(),
                archive_data,
            ]
            result = ensure_pixi(version)
            assert Path(result) == cache / _binary_name()
            assert Path(result).exists()
            assert not (cache.parent / "pixi").exists()

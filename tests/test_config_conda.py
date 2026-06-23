"""Tests for conda backend configuration and validation."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from pyisolate._internal.environment import validate_backend_config


def _make_config(**overrides: Any) -> Any:
    """Build a minimal ExtensionConfig dict with conda defaults."""
    base = {
        "name": "test_ext",
        "module_path": "/fake/path",
        "isolated": True,
        "dependencies": [],
        "apis": [],
        "share_torch": False,
        "share_cuda_ipc": False,
        "sandbox": {},
        "sandbox_mode": "disabled",
        "env": {},
        "package_manager": "uv",
    }
    base.update(overrides)
    return base


class TestCondaShareTorchRaises:
    def test_conda_share_torch_raises(self) -> None:
        """conda + share_torch=True must raise ValueError."""
        config = _make_config(
            package_manager="conda",
            share_torch=True,
            conda_channels=["conda-forge"],
        )
        with pytest.raises(ValueError, match="share_torch=False"):
            validate_backend_config(config)


class TestCondaMissingChannelsRaises:
    def test_conda_missing_channels_raises(self) -> None:
        """conda + empty/missing conda_channels must raise ValueError."""
        config = _make_config(
            package_manager="conda",
        )
        with pytest.raises(ValueError, match="conda_channels"):
            validate_backend_config(config)


class TestCondaMissingPixiRaises:
    @patch(
        "pyisolate._internal.pixi_provisioner.ensure_pixi",
        side_effect=RuntimeError("pixi bootstrap failed"),
    )
    def test_conda_missing_pixi_raises(self, mock_ensure_pixi: Any) -> None:
        """conda + failed pixi bootstrap must raise ValueError."""
        config = _make_config(
            package_manager="conda",
            conda_channels=["conda-forge"],
        )
        with pytest.raises(
            ValueError, match="pixi is required for conda backend but could not be provisioned"
        ):
            validate_backend_config(config)

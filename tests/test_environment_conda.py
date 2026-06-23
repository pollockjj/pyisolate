"""Tests for conda/pixi environment creation (environment_conda.py)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from pyisolate._internal.environment_conda import (
    _generate_pixi_toml,
    _install_cuda_wheels_into_pixi,
    _parse_dep,
    _resolve_pixi_python,
    _resolve_uv_exe,
    create_conda_env,
)
from pyisolate.config import ExtensionConfig


def _make_conda_config(**overrides: object) -> ExtensionConfig:
    """Minimal valid conda config for tests."""
    base: ExtensionConfig = {
        "package_manager": "conda",
        "conda_channels": ["conda-forge"],
        "conda_dependencies": ["numpy"],
        "dependencies": ["requests"],
        "share_torch": False,
        "module": "test_ext",
        "name": "test_ext",
        "isolated": True,
        "apis": [],
        "share_cuda_ipc": False,
    }
    return cast(ExtensionConfig, {**base, **overrides})


def _pixi_python_path(env_path: Path) -> Path:
    if os.name == "nt":
        return env_path / ".pixi" / "envs" / "default" / "python.exe"
    return env_path / ".pixi" / "envs" / "default" / "bin" / "python"


# ── _generate_pixi_toml ──────────────────────────────────────────────


class TestGeneratePixiToml:
    def test_generate_pixi_toml_excludes_cuda_wheel_packages_from_pypi_dependencies(
        self,
    ) -> None:
        config = _make_conda_config(
            dependencies=["requests>=2.0", "spconv", "cumm", "flash-attn"],
            cuda_wheels={
                "index_url": "https://example.invalid/cuda-wheels/",
                "packages": ["spconv", "cumm", "flash-attn"],
            },
        )
        toml_str = _generate_pixi_toml(config)
        assert "[pypi-dependencies]" in toml_str
        assert 'requests = ">=2.0"' in toml_str
        assert "spconv =" not in toml_str
        assert "cumm =" not in toml_str
        assert "flash-attn =" not in toml_str

    def test_generate_pixi_toml_marker_not_in_version(self) -> None:
        config = _make_conda_config(dependencies=["jax[cuda12]>=0.4.30; sys_platform == 'linux'"])
        toml_str = _generate_pixi_toml(config)
        # The marker must NOT appear inside the version field
        assert 'version = ">=0.4.30; sys_platform' not in toml_str
        # It must appear in a separate markers field
        assert 'markers = "sys_platform ==' in toml_str

    def test_generate_pixi_toml_pypi_fallback_produces_parseable_toml(self, tmp_path: Path) -> None:
        try:
            import tomllib  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[import-not-found]

        config = _make_conda_config(dependencies=["jax[cuda12]>=0.4.30", "numpy>=2.2"])
        with patch(
            "pyisolate._internal.environment_conda._pyisolate_source_path",
            return_value=tmp_path,
        ):
            toml_str = _generate_pixi_toml(config)
        parsed = tomllib.loads(toml_str)
        pyisolate_dep = parsed["pypi-dependencies"]["pyisolate"]
        assert isinstance(pyisolate_dep, str), f"Expected string, got {type(pyisolate_dep)}: {pyisolate_dep}"
        assert pyisolate_dep.startswith("=="), (
            f"Expected version pin starting with '==', got: {pyisolate_dep}"
        )


# ── _parse_dep ──────────────────────────────────────────────────────


class TestParseDep:
    def test_parse_dep_marker_extras(self) -> None:
        name, sep, ver, extras, marker = _parse_dep("jax[cuda12]>=0.4.30; sys_platform == 'linux'")
        assert name == "jax"
        assert sep == ">="
        assert ver == ">=0.4.30"
        assert extras == ["cuda12"]
        assert marker == "sys_platform == 'linux'"


# ── create_conda_env ─────────────────────────────────────────────────


class TestCreateCondaEnv:
    def test_sanitizes_invalid_tmpdir_for_pixi_install(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_path = tmp_path / "env"
        config = _make_conda_config()
        stale_tmpdir = tmp_path / "deleted" / "ipc_shared"
        monkeypatch.setenv("TMPDIR", str(stale_tmpdir))

        with (
            patch("pyisolate._internal.pixi_provisioner.ensure_pixi", return_value="/usr/bin/pixi"),
            patch("subprocess.check_call") as mock_call,
            patch(
                "pyisolate._internal.environment_conda._resolve_pixi_python",
                return_value=_pixi_python_path(env_path),
            ),
        ):
            create_conda_env(env_path, config, "test_ext")

        call_kwargs = mock_call.call_args.kwargs
        passed_env = call_kwargs["env"]
        assert passed_env["TMPDIR"] != str(stale_tmpdir)
        assert Path(passed_env["TMPDIR"]).exists()

    def test_fingerprint_skip(self, tmp_path: Path) -> None:
        """If fingerprint matches, pixi install should be skipped."""
        env_path = tmp_path / "env"
        env_path.mkdir(parents=True)
        config = _make_conda_config()

        # Pre-create a matching fingerprint
        import hashlib

        toml_content = _generate_pixi_toml(config)
        descriptor = {
            "conda_dependencies": config.get("conda_dependencies", []),
            "pip_dependencies": config.get("dependencies", []),
            "channels": config.get("conda_channels", []),
            "platforms": config.get("conda_platforms", []),
            "cuda_wheels": config.get("cuda_wheels"),
            "find_links": config.get("find_links", []),
            "pixi_toml": toml_content,
        }
        fingerprint = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
        lock_path = env_path / ".pyisolate_deps.json"
        lock_path.write_text(json.dumps({"fingerprint": fingerprint, "descriptor": descriptor}))

        pixi_python = _pixi_python_path(env_path)
        pixi_python.parent.mkdir(parents=True, exist_ok=True)
        pixi_python.touch()

        with (
            patch("pyisolate._internal.pixi_provisioner.ensure_pixi", return_value="/usr/bin/pixi"),
            patch("subprocess.check_call") as mock_call,
        ):
            create_conda_env(env_path, config, "test_ext")

        # pixi install should NOT have been called
        assert not mock_call.called


# ── _resolve_pixi_python ─────────────────────────────────────────────


class TestResolvePixiPython:
    def test_missing_python_raises(self, tmp_path: Path) -> None:
        env_path = tmp_path / "env"
        env_path.mkdir(parents=True)
        with pytest.raises(RuntimeError, match="Python.*not found"):
            _resolve_pixi_python(env_path)

    def test_never_returns_host_python(self, tmp_path: Path) -> None:
        env_path = tmp_path / "env"
        expected = _pixi_python_path(env_path)
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.touch()
        result = _resolve_pixi_python(env_path)
        assert str(result) != sys.executable
        assert ".pixi" in str(result)


# ── _install_cuda_wheels_into_pixi target_python threading ─────────────


def test_install_cuda_wheels_passes_target_python(monkeypatch: Any, tmp_path: Any) -> Any:
    """AC-1: conda_python='3.12.*' is parsed and passed as target_python=(3, 12)."""
    captured_kwargs: list[dict] = []

    def mock_resolve(deps: Any, config: Any, **kwargs: Any) -> Any:
        captured_kwargs.append(kwargs)
        return deps  # return unchanged (no wheel resolution)

    monkeypatch.setattr(
        "pyisolate._internal.cuda_wheels.resolve_cuda_wheel_requirements",
        mock_resolve,
    )

    python_exe = tmp_path / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.touch()

    config = _make_conda_config(
        conda_python="3.12.*",
        dependencies=["flash-attn"],
        cuda_wheels={
            "index_url": "https://example.invalid/",
            "packages": ["flash-attn"],
        },
    )

    _install_cuda_wheels_into_pixi(python_exe, config, config["cuda_wheels"], "test")

    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["target_python"] == (3, 12)


def test_install_cuda_wheels_wildcard_python_uses_host_tags(monkeypatch: Any, tmp_path: Any) -> Any:
    """AC-2: conda_python='*' passes target_python=None (host tags fallback)."""
    captured_kwargs: list[dict] = []

    def mock_resolve(deps: Any, config: Any, **kwargs: Any) -> Any:
        captured_kwargs.append(kwargs)
        return deps

    monkeypatch.setattr(
        "pyisolate._internal.cuda_wheels.resolve_cuda_wheel_requirements",
        mock_resolve,
    )

    python_exe = tmp_path / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.touch()

    config = _make_conda_config(
        conda_python="*",
        dependencies=["flash-attn"],
        cuda_wheels={
            "index_url": "https://example.invalid/",
            "packages": ["flash-attn"],
        },
    )

    _install_cuda_wheels_into_pixi(python_exe, config, config["cuda_wheels"], "test")

    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["target_python"] is None


# ── _resolve_uv_exe / uv path fallback ─────────────────────────────


class TestResolveUvExe:
    def test_install_cuda_wheels_uv_exe_prefers_local(self, tmp_path: Any) -> None:
        """When python_exe.parent/uv exists, it is preferred over shutil.which."""
        python_exe = tmp_path / "bin" / "python"
        python_exe.parent.mkdir(parents=True)
        python_exe.touch()
        local_uv = tmp_path / "bin" / "uv"
        local_uv.touch()

        resolved = _resolve_uv_exe(python_exe)
        assert resolved == str(local_uv)

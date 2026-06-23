"""Synthetic/unit coverage for CUDA wheel resolution.

These tests intentionally use monkeypatches and fake indexes. They do not
perform a real wheel download or a real install.
"""

import builtins
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from packaging.tags import sys_tags

from pyisolate._internal import environment
from pyisolate._internal.cuda_wheels import (
    CUDAWheelResolutionError,
    CUDAWheelRuntime,
    get_cuda_wheel_runtime,
    resolve_cuda_wheel_requirements,
)
from pyisolate.config import ExtensionConfig, SandboxMode


def _runtime() -> CUDAWheelRuntime:
    return {
        "torch": "2.8",
        "torch_nodot": "28",
        "cuda": "12.8",
        "cuda_nodot": "128",
        "python_tags": [str(tag) for tag in sys_tags()],
    }


def _wheel_filename(distribution: str, version: str) -> str:
    tag = next(iter(sys_tags()))
    return f"{distribution}-{version}-{tag.interpreter}-{tag.abi}-{tag.platform}.whl"


def _simple_index_html(*filenames: str) -> str:
    links = [f'<a href="{filename}">{filename}</a>' for filename in filenames]
    return "<html><body>" + "".join(links) + "</body></html>"


def _fake_venv_python(venv_path: Path) -> Path:
    if sys.platform == "win32":
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        python_path = venv_path / "bin" / "python"
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("#!/bin/sh\n")
    python_path.chmod(0o755)
    return python_path


def test_resolve_cuda_wheel_requirement_picks_highest_matching_version(monkeypatch: Any) -> None:
    runtime = _runtime()
    compatible_old = _wheel_filename("flash_attn", "1.1.0+cu128torch28")
    compatible_new = _wheel_filename("flash_attn", "1.3.0+pt28cu128")
    incompatible_cuda = _wheel_filename("flash_attn", "1.4.0+cu127torch28")
    out_of_range = _wheel_filename("flash_attn", "2.0.0+cu128torch28")
    page_url = "https://example.invalid/cuda-wheels/flash-attn/"

    monkeypatch.setattr("pyisolate._internal.cuda_wheels.get_cuda_wheel_runtime", lambda **kw: runtime)
    monkeypatch.setattr(
        "pyisolate._internal.cuda_wheels._fetch_index_html",
        lambda url: (
            _simple_index_html(
                compatible_old,
                compatible_new,
                incompatible_cuda,
                out_of_range,
            )
            if url == page_url
            else None
        ),
    )

    resolved = resolve_cuda_wheel_requirements(
        ["flash-attn>=1.0,<2.0"],
        {
            "index_url": "https://example.invalid/cuda-wheels/",
            "packages": ["flash-attn"],
            "package_map": {},
        },
    )

    assert resolved == [page_url + compatible_new]


def test_resolve_cuda_wheel_requirement_raises_when_no_match(monkeypatch: Any) -> None:
    runtime = _runtime()
    wheel = _wheel_filename("flash_attn", "1.1.0+cu127torch28")
    page_url = "https://example.invalid/cuda-wheels/flash-attn/"

    monkeypatch.setattr("pyisolate._internal.cuda_wheels.get_cuda_wheel_runtime", lambda **kw: runtime)
    monkeypatch.setattr(
        "pyisolate._internal.cuda_wheels._fetch_index_html",
        lambda url: _simple_index_html(wheel) if url == page_url else None,
    )

    with pytest.raises(CUDAWheelResolutionError, match="No compatible CUDA wheel found"):
        resolve_cuda_wheel_requirements(
            ["flash-attn>=1.0"],
            {
                "index_url": "https://example.invalid/cuda-wheels/",
                "packages": ["flash-attn"],
                "package_map": {},
            },
        )


def test_get_cuda_wheel_runtime_raises_without_torch(monkeypatch: Any) -> Any:
    real_import = builtins.__import__

    def missing_torch(name: Any, *args: Any, **kwargs: Any) -> Any:
        if name == "torch":
            raise ImportError("missing torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_torch)

    with pytest.raises(CUDAWheelResolutionError, match="host torch"):
        get_cuda_wheel_runtime()


def test_get_cuda_wheel_runtime_raises_without_cuda(monkeypatch: Any) -> None:
    fake_torch = SimpleNamespace(
        __version__="2.8.1",
        version=SimpleNamespace(cuda=None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(CUDAWheelResolutionError, match="CUDA-enabled host torch"):
        get_cuda_wheel_runtime()


def test_install_dependencies_cache_invalidation_tracks_cuda_runtime(monkeypatch: Any, tmp_path: Any) -> Any:
    import os

    venv_path = tmp_path / "venv"
    python_exe = venv_path / "Scripts" / "python.exe" if os.name == "nt" else venv_path / "bin" / "python"
    python_exe.parent.mkdir(parents=True, exist_ok=True)
    python_exe.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    monkeypatch.setattr(environment.shutil, "which", lambda binary: "/usr/bin/uv")
    monkeypatch.setattr(
        environment,
        "exclude_satisfied_requirements",
        lambda config, requirements, python_exe: requirements,
    )
    monkeypatch.setattr(
        environment,
        "resolve_cuda_wheel_requirements",
        lambda requirements, config: ["https://example.invalid/flash_attn.whl"],
    )

    current_runtime = {"value": {"torch": "2.8", "cuda": "12.8", "python_tags": ["cp312"]}}
    monkeypatch.setattr(
        environment,
        "get_cuda_wheel_runtime_descriptor",
        lambda: current_runtime["value"],
    )

    popen_calls: list[list[str]] = []

    class MockPopen:
        def __init__(self, cmd: Any, **kwargs: Any) -> None:
            popen_calls.append(cmd)
            self.stdout = io.StringIO("installed\n")

        def wait(self) -> Any:
            return 0

        def __enter__(self) -> Any:
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
            return False

    monkeypatch.setattr(environment.subprocess, "Popen", MockPopen)

    config: ExtensionConfig = {
        "name": "demo",
        "isolated": True,
        "dependencies": ["flash-attn>=1.0"],
        "apis": [],
        "share_torch": True,
        "share_cuda_ipc": False,
        "cuda_wheels": {
            "index_url": "https://example.invalid/cuda-wheels/",
            "packages": ["flash-attn"],
            "package_map": {},
        },
    }

    environment.install_dependencies(venv_path, config, "demo")
    environment.install_dependencies(venv_path, config, "demo")

    current_runtime["value"] = {"torch": "2.8", "cuda": "12.9", "python_tags": ["cp312"]}
    environment.install_dependencies(venv_path, config, "demo")

    assert len(popen_calls) == 2


def test_share_torch_no_deps_rejects_invalid_type(tmp_path: Any, monkeypatch: Any) -> None:
    """Invalid share_torch_no_deps config should fail fast."""
    from pyisolate._internal.environment import install_dependencies

    venv_path = tmp_path / "venvs" / "test-ext"
    _fake_venv_python(venv_path)

    config = {
        "name": "test-ext",
        "module_path": str(tmp_path),
        "isolated": True,
        "dependencies": ["timm"],
        "apis": [],
        "share_torch": True,
        "share_torch_no_deps": "timm",
        "share_cuda_ipc": False,
        "sandbox_mode": SandboxMode.DISABLED,
        "sandbox": {},
    }

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(
        "pyisolate._internal.environment.exclude_satisfied_requirements",
        lambda config, reqs, python_exe: reqs,
    )

    with pytest.raises(TypeError, match="share_torch_no_deps"):
        install_dependencies(venv_path, cast(ExtensionConfig, config), "test-ext")

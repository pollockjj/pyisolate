"""Smoke tests that execute the built-in examples end-to-end."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_example(script: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("UV_PIP_DISABLE_EXTERNALLY_MANAGED", "1")
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


@pytest.mark.smoke
def test_multi_extension_example_runs() -> None:
    result = _run_example(REPO_ROOT / "example" / "main.py")
    assert "Extension1" in result.stdout
    assert "Extension2" in result.stdout
    assert "Extension3" in result.stdout


@pytest.mark.smoke
def test_comfy_hello_world_runs() -> None:
    result = _run_example(REPO_ROOT / "comfy_hello_world" / "main.py")
    assert "SimpleTextNode" in result.stdout
    assert "ComfyUI Hello World Complete" in result.stdout

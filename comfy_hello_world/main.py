"""Standalone demo that mirrors the ComfyUI → PyIsolate flow."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Type

import yaml

from pyisolate import ExtensionManager, ExtensionManagerConfig
from pyisolate.shared import ProxiedSingleton

try:  # Support running as a script: ``python comfy_hello_world/main.py``
    from .extension import ComfyNodeExtension
except ImportError:  # pragma: no cover - exercised manually
    sys.path.insert(0, str(Path(__file__).parent))
    from extension import ComfyNodeExtension  # type: ignore

BASE_DIR = Path(__file__).parent.resolve()
NODE_NAME = "simple_text_node"
NODE_DIR = BASE_DIR / "custom_nodes" / NODE_NAME
MANIFEST_PATH = NODE_DIR / "pyisolate.yaml"

os.environ.setdefault("UV_PIP_DISABLE_EXTERNALLY_MANAGED", "1")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _load_shared_model_manager() -> Type[ProxiedSingleton]:
    """Import the shared singleton class that both host and node use."""

    if str(NODE_DIR) not in sys.path:
        sys.path.insert(0, str(NODE_DIR))
        added = True
    else:
        added = False

    from shared_services import HelloWorldModelManager  # type: ignore

    if added:
        sys.path.pop(0)

    return HelloWorldModelManager

def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if "dependencies" not in data or not data["dependencies"]:
        raise RuntimeError(f"Manifest must declare dependencies: {MANIFEST_PATH}")
    return data


async def main() -> None:
    print("📦 Loading isolated custom node...")
    venv_root = BASE_DIR / "node-venvs"
    venv_root.mkdir(exist_ok=True)

    manager = ExtensionManager(
        ComfyNodeExtension,
        ExtensionManagerConfig(venv_root_path=str(venv_root)),
    )

    manifest = _load_manifest()
    HelloWorldModelManager = _load_shared_model_manager()
    extension = manager.load_extension(
        {
            "name": NODE_NAME,
            "module_path": str(NODE_DIR),
            "isolated": bool(manifest.get("isolated", True)),
            "dependencies": manifest["dependencies"],
            "apis": [HelloWorldModelManager],
            "share_torch": bool(manifest.get("share_torch", False)),
        }
    )

    nodes = await extension.list_nodes()
    print(f"   Available nodes: {nodes}")

    print("🚀 Executing node...")
    result = await extension.execute_node(
        "SimpleTextNode",
        text="Hello from ComfyUI with PyIsolate!",
    )
    payload = result[0] if isinstance(result, (list, tuple)) else result
    print(f"✅ Result: {payload}")

    print("🔗 Testing shared model manager...")
    default_model = await extension.fetch_default_model_name()
    print(f"   Remote model manager returned: {default_model}")

    manager.stop_extension(NODE_NAME)
    print("✅ ComfyUI Hello World Complete!")


if __name__ == "__main__":
    asyncio.run(main())

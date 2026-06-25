"""Child-process bootstrap for PyIsolate.

This module resolves the "config before path" paradox by applying the host's
snapshot (sys.path + adapter metadata) before any heavy imports occur in the
child process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any, cast

from ..interfaces import IsolationAdapter
from ..path_helpers import build_child_sys_path
from .serialization_registry import SerializerRegistry

logger = logging.getLogger(__name__)

# Parsed host snapshot from the most recent bootstrap_child(); consulted later by
# install_service_module_shims() once the RPC channel exists in the child.
_LAST_SNAPSHOT: dict[str, Any] | None = None


def _should_apply_host_sys_path(snapshot: dict[str, Any]) -> bool:
    return bool(snapshot.get("apply_host_sys_path", True))


def _merge_sys_path_front(paths: list[str]) -> None:
    """Prepend paths to sys.path while preserving order and removing duplicates."""
    seen = set()
    merged: list[str] = []

    def add_path(p: str) -> None:
        norm = os.path.normcase(os.path.abspath(p))
        if norm in seen:
            return
        seen.add(norm)
        merged.append(p)

    for p in paths:
        add_path(p)

    for p in sys.path:
        add_path(p)

    sys.path[:] = merged


def _apply_sealed_opt_in_paths(snapshot: dict[str, Any]) -> None:
    raw_paths = snapshot.get("sealed_host_ro_paths", [])
    if not isinstance(raw_paths, list):
        return

    opt_in_paths: list[str] = []
    for path in raw_paths:
        if not isinstance(path, str) or not path.strip():
            continue
        if not os.path.isabs(path):
            continue
        if not os.path.exists(path):
            continue
        opt_in_paths.append(path)

    if not opt_in_paths:
        return

    _merge_sys_path_front(opt_in_paths)
    logger.debug("Applied %d sealed opt-in import paths", len(opt_in_paths))


def _apply_sys_path(snapshot: dict[str, Any]) -> None:
    if not _should_apply_host_sys_path(snapshot):
        _apply_sealed_opt_in_paths(snapshot)
        logger.debug("Skipping host sys.path reconstruction for sealed child")
        return

    host_paths = snapshot.get("sys_path", [])
    extra_paths = snapshot.get("additional_paths", [])

    preferred_root: str | None = snapshot.get("preferred_root")
    if not preferred_root:
        context_data = snapshot.get("context_data", {})
        module_path = context_data.get("module_path") or os.environ.get("PYISOLATE_MODULE_PATH")
        if module_path:
            preferred_root = str(Path(module_path).parent.parent)

    filtered_subdirs = snapshot.get("filtered_subdirs")
    child_paths = build_child_sys_path(host_paths, extra_paths, preferred_root, filtered_subdirs)

    if not child_paths:
        return

    # Rebuild sys.path with child paths first while preserving any existing entries
    # that are not already in the computed set.
    _merge_sys_path_front(child_paths)
    logger.debug("Applied %d paths from snapshot (preferred_root=%s)", len(child_paths), preferred_root)


def _rehydrate_adapter(start_ref: str) -> IsolationAdapter:
    """Import and instantiate adapter from string reference."""
    import importlib

    from .adapter_registry import AdapterRegistry

    try:
        module_path, class_name = start_ref.split(":", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        # Instantiate and register immediately
        adapter = cls()

        # KEY STEP: Register in child's memory space so subsequent calls work
        AdapterRegistry.register(adapter)

        return cast(IsolationAdapter, adapter)
    except Exception as exc:
        raise ValueError(f"Failed to rehydrate adapter '{start_ref}': {exc}") from exc


def bootstrap_child() -> IsolationAdapter | None:
    """Initialize child environment using host snapshot.

    Returns:
        The loaded adapter instance, or None if no snapshot/adapter present.

    Raises:
        ValueError: If snapshot is malformed or adapter cannot be loaded.
    """
    snapshot_env = os.environ.get("PYISOLATE_HOST_SNAPSHOT")
    if not snapshot_env:
        logger.debug("No PYISOLATE_HOST_SNAPSHOT set; skipping bootstrap")
        return None

    snapshot: dict[str, Any]

    # PYISOLATE_HOST_SNAPSHOT may be either a JSON string or a file path.
    # If it starts with '{', assume it's a JSON payload.
    if snapshot_env.strip().startswith("{"):
        looks_like_path = False
    else:
        looks_like_path = os.path.sep in snapshot_env or snapshot_env.endswith(".json")

    if looks_like_path:
        try:
            with open(snapshot_env, encoding="utf-8") as fh:
                snapshot_text = fh.read()
        except FileNotFoundError:
            logger.debug("Snapshot path missing (%s); skipping bootstrap", snapshot_env)
            return None

        try:
            snapshot = json.loads(snapshot_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to decode snapshot file {snapshot_env}: {exc}") from exc
    else:
        try:
            snapshot = json.loads(snapshot_env)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to decode PYISOLATE_HOST_SNAPSHOT: {exc}") from exc

    global _LAST_SNAPSHOT
    _LAST_SNAPSHOT = snapshot

    _apply_sys_path(snapshot)

    adapter: IsolationAdapter | None = None
    is_sealed = not _should_apply_host_sys_path(snapshot)

    adapter_ref = snapshot.get("adapter_ref")
    if adapter_ref:
        try:
            adapter = _rehydrate_adapter(adapter_ref)
        except Exception as exc:
            logger.warning(
                "Failed to rehydrate adapter from ref %s: %s",
                adapter_ref,
                exc,
            )

    if not adapter and adapter_ref and not is_sealed:
        raise ValueError("Snapshot contained adapter info but adapter could not be loaded")

    if adapter:
        if not is_sealed:
            adapter.setup_child_environment(snapshot)
        registry = SerializerRegistry.get_instance()
        adapter.register_serializers(registry)

    return adapter


def _make_service_shim(
    module_name: str, service_id: str, methods: dict[str, Any], rpc: Any
) -> types.ModuleType:
    """Build a synthetic module that forwards function calls to a host service.

    Each declared function dispatches synchronously over RPC by name; the host
    resolves ``service_id`` + the rpc method and runs the real implementation.
    Calls only work at execution time (RPC running), not at module-import time.
    """
    mod = types.ModuleType(module_name)
    mod.__pyisolate_service_shim__ = True  # type: ignore[attr-defined]
    for func_name, rpc_method in methods.items():
        target = rpc_method or func_name

        def _forward(
            *args: Any,
            _service: str = service_id,
            _method: str = target,
            _fn: str = func_name,
            **kwargs: Any,
        ) -> Any:
            # Only dispatchable at execution time: the RPC loop must be running
            # and we must be off that loop's thread (a worker thread) to block on
            # the result. At module-import time the extension runs synchronously
            # on the RPC loop thread; present as a missing attribute so callers
            # that probe host modules at import scope degrade (their typical
            # `except (ImportError, AttributeError)`) instead of crashing.
            loop = getattr(rpc, "default_loop", None)
            try:
                on_loop = asyncio.get_running_loop() is loop
            except RuntimeError:
                on_loop = False
            if loop is None or not loop.is_running() or on_loop:
                raise AttributeError(
                    f"{module_name}.{_fn} is only callable at execution time "
                    "(RPC running, off the event-loop thread)"
                )
            return rpc.call_service_sync(_service, _method, *args, **kwargs)

        mod.__dict__[func_name] = _forward
    return mod


def install_service_module_shims(rpc: Any) -> int:
    """Install host-declared service module shims into ``sys.modules`` (sealed only).

    Reads the ``service_module_map`` carried in the host snapshot and, for a sealed
    child (which lacks the host framework's real modules), installs forwarding
    shims so e.g. ``import folder_paths`` resolves to RPC calls. Host-coupled
    children keep their real modules and are skipped. Returns the count installed.
    """
    snapshot = _LAST_SNAPSHOT
    if snapshot is None:
        return 0
    # Host-coupled children import the real modules; only sealed workers need shims.
    if _should_apply_host_sys_path(snapshot):
        return 0
    service_map = snapshot.get("service_module_map") or {}
    if not isinstance(service_map, dict):
        return 0

    installed = 0
    for module_name, spec in service_map.items():
        if not isinstance(spec, dict):
            continue
        service_id = spec.get("service")
        methods = spec.get("methods") or {}
        if not service_id or not isinstance(methods, dict):
            continue
        sys.modules[str(module_name)] = _make_service_shim(str(module_name), str(service_id), methods, rpc)
        installed += 1

    if installed:
        logger.debug("Installed %d host-service module shim(s) for sealed worker", installed)
    return installed

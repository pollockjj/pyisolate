import asyncio
import sys
import threading

import pytest

from pyisolate._internal import bootstrap


class _FakeRPC:
    def __init__(self, loop=None) -> None:
        self.default_loop = loop
        self.calls: list[tuple] = []

    def call_service_sync(self, service, method, *args, **kwargs):
        self.calls.append((service, method, args, kwargs))
        return f"{service}.{method}"


_MAP = {
    "folder_paths_testshim": {
        "service": "FolderPathsProxy",
        "methods": {"get_input_directory": "rpc_get_input_directory", "get_full_path": "rpc_get_full_path"},
    }
}


def _restore(prev_snapshot):
    bootstrap._LAST_SNAPSHOT = prev_snapshot
    sys.modules.pop("folder_paths_testshim", None)


def _running_loop_in_thread():
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    ready.wait(timeout=5)
    return loop, t


def test_shims_install_for_sealed_and_forward_at_execution() -> None:
    prev = bootstrap._LAST_SNAPSHOT
    loop, t = _running_loop_in_thread()
    rpc = _FakeRPC(loop=loop)
    try:
        bootstrap._LAST_SNAPSHOT = {"apply_host_sys_path": False, "service_module_map": _MAP}
        assert bootstrap.install_service_module_shims(rpc) == 1

        shim = sys.modules["folder_paths_testshim"]
        assert getattr(shim, "__pyisolate_service_shim__", False) is True

        # Caller is the main thread (off the running loop) -> dispatches.
        shim.get_input_directory()
        shim.get_full_path("checkpoints", "x.safetensors")
        assert rpc.calls[0][:2] == ("FolderPathsProxy", "rpc_get_input_directory")
        assert rpc.calls[1] == ("FolderPathsProxy", "rpc_get_full_path", ("checkpoints", "x.safetensors"), {})
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5)
        _restore(prev)


def test_shim_degrades_to_attributeerror_when_not_dispatchable() -> None:
    """Import-time call (no running loop) presents as missing, not a crash."""
    prev = bootstrap._LAST_SNAPSHOT
    try:
        bootstrap._LAST_SNAPSHOT = {"apply_host_sys_path": False, "service_module_map": _MAP}
        assert bootstrap.install_service_module_shims(_FakeRPC(loop=None)) == 1
        shim = sys.modules["folder_paths_testshim"]
        with pytest.raises(AttributeError):
            shim.get_input_directory()
    finally:
        _restore(prev)


def test_shims_not_installed_for_host_coupled() -> None:
    prev = bootstrap._LAST_SNAPSHOT
    try:
        bootstrap._LAST_SNAPSHOT = {"apply_host_sys_path": True, "service_module_map": _MAP}
        assert bootstrap.install_service_module_shims(_FakeRPC()) == 0
        assert "folder_paths_testshim" not in sys.modules
    finally:
        _restore(prev)

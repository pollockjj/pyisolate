import sys

from pyisolate._internal import bootstrap


class _FakeRPC:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple, dict]] = []

    def call_service_sync(self, service, method, *args, **kwargs):
        self.calls.append((service, method, args, kwargs))
        return f"{service}.{method}->{args}"


_MAP = {
    "folder_paths_testshim": {
        "service": "FolderPathsProxy",
        "methods": {"get_input_directory": "rpc_get_input_directory", "get_full_path": "rpc_get_full_path"},
    }
}


def _restore(prev_snapshot):
    bootstrap._LAST_SNAPSHOT = prev_snapshot
    sys.modules.pop("folder_paths_testshim", None)


def test_shims_install_for_sealed_and_forward_by_name() -> None:
    prev = bootstrap._LAST_SNAPSHOT
    rpc = _FakeRPC()
    try:
        bootstrap._LAST_SNAPSHOT = {"apply_host_sys_path": False, "service_module_map": _MAP}
        count = bootstrap.install_service_module_shims(rpc)
        assert count == 1

        shim = sys.modules["folder_paths_testshim"]
        assert getattr(shim, "__pyisolate_service_shim__", False) is True

        shim.get_input_directory()
        shim.get_full_path("checkpoints", "x.safetensors")

        assert rpc.calls[0][:2] == ("FolderPathsProxy", "rpc_get_input_directory")
        assert rpc.calls[1] == ("FolderPathsProxy", "rpc_get_full_path", ("checkpoints", "x.safetensors"), {})
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

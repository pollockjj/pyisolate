import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pyisolate._internal.sandbox_detect import RestrictionModel

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="bubblewrap command composition is Linux-specific",
)

_DEFAULTS: dict[str, Any] = {
    "python_exe": "/venv/bin/python",
    "module_path": "/path/to/module",
    "venv_path": "/venv",
    "uds_address": "/run/user/1000/pyisolate/test.sock",
    "allow_gpu": False,
    "restriction_model": RestrictionModel.NONE,
}


def _mockbuild_bwrap_command(**kwargs: Any) -> list[str]:
    mock_pyisolate = MagicMock()
    mock_pyisolate.__file__ = "/fake/pyisolate/__init__.py"
    import builtins

    original_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kw: Any) -> Any:
        if name == "comfy":
            raise ImportError("No module named 'comfy'")
        return original_import(name, *args, **kw)

    with (
        patch.dict("sys.modules", {"pyisolate": mock_pyisolate}),
        patch.object(sys, "executable", "/fake/python"),
        patch.object(Path, "glob", return_value=[]),
        patch("os.path.exists", return_value=True),
        patch("os.getuid", return_value=kwargs.pop("uid", 1000)),
        patch.object(builtins, "__import__", mock_import),
    ):
        from pyisolate._internal.host import build_bwrap_command

        return build_bwrap_command(**kwargs)


def _bwrap(**overrides: Any) -> list[str]:
    return _mockbuild_bwrap_command(**{**_DEFAULTS, **overrides})


def test_die_with_parent_always_present() -> None:
    assert "--die-with-parent" in _bwrap()


def test_namespace_isolation_when_available() -> None:
    cmd = _bwrap()
    assert "--unshare-user" in cmd
    assert "--unshare-pid" in cmd
    assert "--unshare-ipc" not in cmd


def test_namespace_isolation_degraded_ubuntu() -> None:
    cmd = _bwrap(restriction_model=RestrictionModel.UBUNTU_APPARMOR)
    assert "--unshare-user" not in cmd
    assert "--unshare-pid" not in cmd
    assert "--unshare-ipc" not in cmd


def test_network_always_isolated() -> None:
    cmd = _bwrap()
    assert "--unshare-net" in cmd
    assert "--share-net" not in cmd


def test_uds_parent_directories_created() -> None:
    cmd_str = " ".join(_bwrap())
    assert "--dir /run" in cmd_str
    assert "--dir /run/user/1000" in cmd_str
    assert "--dir /run/user/1000/pyisolate" in cmd_str


def test_dev_shm_always_bound_for_tensor_sharing() -> None:
    cmd = _bwrap()
    shm_bound = any(
        arg in ("--bind", "--dev-bind") and i + 1 < len(cmd) and "/dev/shm" in cmd[i + 1]  # noqa: S108
        for i, arg in enumerate(cmd)
    )
    assert shm_bound, "/dev/shm must be bound for SharedMemory Lease tensor transfer"  # noqa: S108


def test_base_prefix_ro_bound() -> None:
    with patch.object(sys, "base_prefix", "/opt/custom_python"):
        cmd_str = " ".join(_bwrap())
    assert "--ro-bind /opt/custom_python /opt/custom_python" in cmd_str


def test_adapter_system_paths_ro_bound() -> None:
    mock_adapter = MagicMock()
    mock_adapter.get_sandbox_system_paths.return_value = ["/app/framework"]
    cmd_str = " ".join(_bwrap(adapter=mock_adapter))
    assert "--ro-bind /app/framework /app/framework" in cmd_str


def test_resolved_python_prefix_ro_bound() -> None:
    resolved = "/home/linuxbrew/.linuxbrew/Cellar/python@3.13/3.13.12_1"
    with patch("pathlib.Path.resolve", return_value=Path(resolved + "/bin/python3.13")):
        cmd_str = " ".join(_bwrap())
    assert f"--ro-bind {resolved} {resolved}" in cmd_str


def test_venv_readonly() -> None:
    cmd = _bwrap()
    venv_readonly = any(
        arg == "--ro-bind" and i + 1 < len(cmd) and "/venv" in cmd[i + 1] for i, arg in enumerate(cmd)
    )
    assert venv_readonly, "Venv should be read-only to prevent infection"


def test_tmpfs_tmp_and_no_host_tmp_bind() -> None:
    cmd_str = " ".join(_bwrap(sandbox_config={"writable_paths": ["/dev/shm", "/tmp", "/tmp/"]}))
    assert "--tmpfs /tmp" in cmd_str
    assert "--bind /tmp /tmp" not in cmd_str


def test_pyisolate_child_set() -> None:
    cmd = _bwrap()
    found = any(
        arg == "--setenv" and i + 2 < len(cmd) and cmd[i + 1] == "PYISOLATE_CHILD" and cmd[i + 2] == "1"
        for i, arg in enumerate(cmd)
    )
    assert found, "PYISOLATE_CHILD=1 should be set"


def test_uds_address_set() -> None:
    uds_path = "/run/user/1000/pyisolate/test.sock"
    cmd = _bwrap(uds_address=uds_path)
    found = any(
        arg == "--setenv"
        and i + 2 < len(cmd)
        and cmd[i + 1] == "PYISOLATE_UDS_ADDRESS"
        and cmd[i + 2] == uds_path
        for i, arg in enumerate(cmd)
    )
    assert found, "PYISOLATE_UDS_ADDRESS should be set to socket path"


def test_ends_with_python_uds_client() -> None:
    cmd = _bwrap()
    assert cmd[-3] == "/venv/bin/python"
    assert cmd[-2] == "-m"
    assert cmd[-1] == "pyisolate._internal.uds_client"


def test_sealed_worker_does_not_bind_host_site_packages() -> None:
    cmd_str = " ".join(_bwrap(execution_model="sealed_worker"))
    assert "site-packages" not in cmd_str


def test_sealed_worker_does_not_bind_dev_shm() -> None:
    cmd = _bwrap(execution_model="sealed_worker")
    for i, arg in enumerate(cmd):
        if arg in ("--bind", "--dev-bind") and i + 1 < len(cmd):
            assert "/dev/shm" not in cmd[i + 1]


def test_sealed_worker_host_policy_ro_paths_add_ro_bind_and_keep_clearenv() -> None:
    ro_path = "/opt/comfyui"
    cmd = _bwrap(execution_model="sealed_worker", sealed_host_ro_paths=[ro_path])
    cmd_str = " ".join(cmd)
    assert f"--ro-bind {ro_path} {ro_path}" in cmd_str
    assert "--clearenv" in cmd
    assert "PYTHONPATH" not in cmd
    assert "--setenv PYTHONPATH " not in cmd_str

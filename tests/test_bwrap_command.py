"""Unit tests for bwrap command building and sandbox integration.

Tests cover:
- _build_bwrap_command() flag composition
- Lifecycle coupling (--die-with-parent)
- Namespace isolation (conditional based on RestrictionModel)
- Network configuration
- UDS mount topology
- GPU passthrough
- Read-only filesystem bindings
"""

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


def _mockbuild_bwrap_command(**kwargs: Any) -> list[str]:
    """Call build_bwrap_command with proper mocking."""
    # Mock pyisolate package import
    mock_pyisolate = MagicMock()
    mock_pyisolate.__file__ = "/fake/pyisolate/__init__.py"

    # Mock comfy package to raise ImportError (not in ComfyUI context)
    # This simulates running outside ComfyUI
    import builtins

    original_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kw: Any) -> Any:
        if name == "comfy":
            raise ImportError("No module named 'comfy'")
        return original_import(name, *args, **kw)

    # Mock sys.executable for host site-packages lookup
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


class TestLifecycleCoupling:
    """Test --die-with-parent lifecycle coupling."""

    def test_die_with_parent_always_present(self) -> None:
        """Verify --die-with-parent flag is always present."""
        cmd = _mockbuild_bwrap_command(
            python_exe="/venv/bin/python",
            module_path="/path/to/module",
            venv_path="/venv",
            uds_address="/run/user/1000/pyisolate/test.sock",
            allow_gpu=False,
            restriction_model=RestrictionModel.NONE,
        )
        assert "--die-with-parent" in cmd


class TestNamespaceIsolation:
    """Test conditional namespace isolation based on RestrictionModel."""

    def test_namespace_isolation_when_available(self) -> None:
        """Verify namespace flags when no restrictions.

        Note: IPC namespace (--unshare-ipc) is NOT isolated because SharedMemory Lease
        requires shared IPC namespace for zero-copy tensor transfer via /dev/shm.
        """
        cmd = _mockbuild_bwrap_command(
            python_exe="/venv/bin/python",
            module_path="/path/to/module",
            venv_path="/venv",
            uds_address="/run/user/1000/pyisolate/test.sock",
            allow_gpu=False,
            restriction_model=RestrictionModel.NONE,
        )
        assert "--unshare-user" in cmd
        assert "--unshare-pid" in cmd
        # IPC namespace is NOT unshared - required for SharedMemory Lease
        assert "--unshare-ipc" not in cmd

    def test_namespace_isolation_degraded_ubuntu(self) -> None:
        """Verify namespace flags absent when Ubuntu AppArmor restricted."""
        cmd = _mockbuild_bwrap_command(
            python_exe="/venv/bin/python",
            module_path="/path/to/module",
            venv_path="/venv",
            uds_address="/run/user/1000/pyisolate/test.sock",
            allow_gpu=False,
            restriction_model=RestrictionModel.UBUNTU_APPARMOR,
        )
        assert "--unshare-user" not in cmd
        assert "--unshare-pid" not in cmd
        assert "--unshare-ipc" not in cmd


class TestNetworkConfiguration:
    """Test network isolation (host-controlled, always isolated)."""

    def test_network_always_isolated(self) -> None:
        """Verify --unshare-net is always present (host policy, not user config)."""
        cmd = _mockbuild_bwrap_command(
            python_exe="/venv/bin/python",
            module_path="/path/to/module",
            venv_path="/venv",
            uds_address="/run/user/1000/pyisolate/test.sock",
            allow_gpu=False,
            restriction_model=RestrictionModel.NONE,
        )
        assert "--unshare-net" in cmd
        assert "--share-net" not in cmd


class TestUDSMountTopology:
    """Test UDS mount directory creation."""

    def test_uds_parent_directories_created(self) -> None:
        """Verify UDS parent directories are created before bind."""
        cmd = _mockbuild_bwrap_command(
            python_exe="/venv/bin/python",
            module_path="/path/to/module",
            venv_path="/venv",
            uds_address="/run/user/1000/pyisolate/test.sock",
            allow_gpu=False,
            restriction_model=RestrictionModel.NONE,
        )
        # Build command string pairs for easier inspection
        cmd_str = " ".join(cmd)

        # Verify parent dir creation
        assert "--dir /run" in cmd_str
        assert "--dir /run/user/1000" in cmd_str
        assert "--dir /run/user/1000/pyisolate" in cmd_str


class TestGPUPassthrough:
    """Test GPU device passthrough."""

    def test_dev_shm_always_bound_for_tensor_sharing(self) -> None:
        """/dev/shm is ALWAYS bound - required for SharedMemory Lease tensor transfer.

        The SharedMemory Lease pattern requires /dev/shm for zero-copy CPU tensor
        transfer between host and sandboxed child, regardless of GPU setting.
        """
        cmd = _mockbuild_bwrap_command(
            python_exe="/venv/bin/python",
            module_path="/path/to/module",
            venv_path="/venv",
            uds_address="/run/user/1000/pyisolate/test.sock",
            allow_gpu=False,
            restriction_model=RestrictionModel.NONE,
        )
        # /dev/shm MUST appear in bind commands for tensor sharing
        shm_bound = False
        for i, arg in enumerate(cmd):
            if arg in ("--bind", "--dev-bind") and i + 1 < len(cmd) and "/dev/shm" in cmd[i + 1]:  # noqa: S108
                shm_bound = True
                break
        assert shm_bound, "/dev/shm must be bound for SharedMemory Lease tensor transfer"  # noqa: S108


class TestFilesystemIsolation:
    """Test filesystem isolation properties."""

    def test_base_prefix_ro_bound(self) -> None:
        """Verify sys.base_prefix is added to --ro-bind to support non-standard host Pythons."""
        with patch.object(sys, "base_prefix", "/opt/custom_python"):
            cmd = _mockbuild_bwrap_command(
                python_exe="/venv/bin/python",
                module_path="/path/to/module",
                venv_path="/venv",
                uds_address="/run/user/1000/pyisolate/test.sock",
                allow_gpu=False,
                restriction_model=RestrictionModel.NONE,
            )

            cmd_str = " ".join(cmd)
            assert "--ro-bind /opt/custom_python /opt/custom_python" in cmd_str

    def test_adapter_system_paths_ro_bound(self) -> None:
        """Verify adapter-provided system paths are added to --ro-bind."""
        mock_adapter = MagicMock()
        mock_adapter.get_sandbox_system_paths.return_value = ["/app/framework"]

        cmd = _mockbuild_bwrap_command(
            python_exe="/venv/bin/python",
            module_path="/path/to/module",
            venv_path="/venv",
            uds_address="/run/user/1000/pyisolate/test.sock",
            allow_gpu=False,
            restriction_model=RestrictionModel.NONE,
            adapter=mock_adapter,
        )

        cmd_str = " ".join(cmd)
        assert "--ro-bind /app/framework /app/framework" in cmd_str

    def test_resolved_python_prefix_ro_bound(self) -> None:
        """Verify the resolved venv interpreter prefix is also bound read-only."""
        with patch(
            "pathlib.Path.resolve",
            return_value=Path("/home/linuxbrew/.linuxbrew/Cellar/python@3.13/3.13.12_1/bin/python3.13"),
        ):
            cmd = _mockbuild_bwrap_command(
                python_exe="/venv/bin/python",
                module_path="/path/to/module",
                venv_path="/venv",
                uds_address="/run/user/1000/pyisolate/test.sock",
                allow_gpu=False,
                restriction_model=RestrictionModel.NONE,
            )

            cmd_str = " ".join(cmd)
            assert (
                "--ro-bind /home/linuxbrew/.linuxbrew/Cellar/python@3.13/3.13.12_1 "
                "/home/linuxbrew/.linuxbrew/Cellar/python@3.13/3.13.12_1"
            ) in cmd_str

    def test_venv_readonly(self) -> None:
        """Verify venv is bound read-only."""
        cmd = _mockbuild_bwrap_command(
            python_exe="/venv/bin/python",
            module_path="/path/to/module",
            venv_path="/venv",
            uds_address="/run/user/1000/pyisolate/test.sock",
            allow_gpu=False,
            restriction_model=RestrictionModel.NONE,
        )
        # Find venv in ro-bind
        venv_readonly = False
        for i, arg in enumerate(cmd):
            if arg == "--ro-bind" and i + 1 < len(cmd) and "/venv" in cmd[i + 1]:
                venv_readonly = True
                break
        assert venv_readonly, "Venv should be read-only to prevent infection"

    def test_tmpfs_tmp_and_no_host_tmp_bind(self) -> None:
        """Verify host /tmp cannot override the private tmpfs."""
        cmd = _mockbuild_bwrap_command(
            python_exe="/venv/bin/python",
            module_path="/path/to/module",
            venv_path="/venv",
            uds_address="/run/user/1000/pyisolate/test.sock",
            allow_gpu=False,
            restriction_model=RestrictionModel.NONE,
            sandbox_config={"writable_paths": ["/dev/shm", "/tmp", "/tmp/"]},
        )

        cmd_str = " ".join(cmd)
        assert "--tmpfs /tmp" in cmd_str
        assert "--bind /tmp /tmp" not in cmd_str


class TestEnvironmentVariables:
    """Test environment variable passthrough."""

    def test_pyisolate_child_set(self) -> None:
        """Verify PYISOLATE_CHILD=1 is set."""
        cmd = _mockbuild_bwrap_command(
            python_exe="/venv/bin/python",
            module_path="/path/to/module",
            venv_path="/venv",
            uds_address="/run/user/1000/pyisolate/test.sock",
            allow_gpu=False,
            restriction_model=RestrictionModel.NONE,
        )
        # Find PYISOLATE_CHILD in setenv
        found = False
        for i, arg in enumerate(cmd):
            if (
                arg == "--setenv"
                and i + 2 < len(cmd)
                and cmd[i + 1] == "PYISOLATE_CHILD"
                and cmd[i + 2] == "1"
            ):
                found = True
                break
        assert found, "PYISOLATE_CHILD=1 should be set"

    def test_uds_address_set(self) -> None:
        """Verify PYISOLATE_UDS_ADDRESS is set."""
        uds_path = "/run/user/1000/pyisolate/test.sock"
        cmd = _mockbuild_bwrap_command(
            python_exe="/venv/bin/python",
            module_path="/path/to/module",
            venv_path="/venv",
            uds_address=uds_path,
            allow_gpu=False,
            restriction_model=RestrictionModel.NONE,
        )
        # Find PYISOLATE_UDS_ADDRESS in setenv
        found = False
        for i, arg in enumerate(cmd):
            if (
                arg == "--setenv"
                and i + 2 < len(cmd)
                and cmd[i + 1] == "PYISOLATE_UDS_ADDRESS"
                and cmd[i + 2] == uds_path
            ):
                found = True
                break
        assert found, "PYISOLATE_UDS_ADDRESS should be set to socket path"


class TestCommandStructure:
    """Test overall command structure."""

    def test_ends_with_python_uds_client(self) -> None:
        """Verify command ends with python -m pyisolate._internal.uds_client."""
        cmd = _mockbuild_bwrap_command(
            python_exe="/venv/bin/python",
            module_path="/path/to/module",
            venv_path="/venv",
            uds_address="/run/user/1000/pyisolate/test.sock",
            allow_gpu=False,
            restriction_model=RestrictionModel.NONE,
        )
        assert cmd[-3] == "/venv/bin/python"
        assert cmd[-2] == "-m"
        assert cmd[-1] == "pyisolate._internal.uds_client"


class TestSealedWorkerCommand:
    """Test strict sealed-worker sandbox policy."""

    def test_sealed_worker_does_not_bind_host_site_packages(self) -> None:
        cmd = _mockbuild_bwrap_command(
            python_exe="/venv/bin/python",
            module_path="/path/to/module",
            venv_path="/venv",
            uds_address="/run/user/1000/pyisolate/test.sock",
            allow_gpu=False,
            restriction_model=RestrictionModel.NONE,
            execution_model="sealed_worker",
        )
        cmd_str = " ".join(cmd)
        assert "site-packages" not in cmd_str

    def test_sealed_worker_does_not_bind_dev_shm(self) -> None:
        cmd = _mockbuild_bwrap_command(
            python_exe="/venv/bin/python",
            module_path="/path/to/module",
            venv_path="/venv",
            uds_address="/run/user/1000/pyisolate/test.sock",
            allow_gpu=False,
            restriction_model=RestrictionModel.NONE,
            execution_model="sealed_worker",
        )
        for i, arg in enumerate(cmd):
            if arg in ("--bind", "--dev-bind") and i + 1 < len(cmd):
                assert "/dev/shm" not in cmd[i + 1]

    def test_sealed_worker_host_policy_ro_paths_add_ro_bind_and_keep_clearenv(self) -> None:
        cmd = _mockbuild_bwrap_command(
            python_exe="/venv/bin/python",
            module_path="/path/to/module",
            venv_path="/venv",
            uds_address="/run/user/1000/pyisolate/test.sock",
            allow_gpu=False,
            restriction_model=RestrictionModel.NONE,
            execution_model="sealed_worker",
            sealed_host_ro_paths=["/home/johnj/ComfyUI"],
        )
        cmd_str = " ".join(cmd)
        assert "--ro-bind /home/johnj/ComfyUI /home/johnj/ComfyUI" in cmd_str
        assert "--clearenv" in cmd
        assert "PYTHONPATH" not in cmd
        assert "--setenv PYTHONPATH " not in cmd_str

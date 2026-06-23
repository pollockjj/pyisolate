
from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

from pyisolate.config import ExtensionConfig


def _make_config(**overrides: object) -> ExtensionConfig:
    config: ExtensionConfig = {
        "name": "test_ext",
        "module": "test_module",
        "dependencies": [],
        "share_torch": False,
        "share_cuda_ipc": False,
        "isolated": True,
        "apis": [],
    }
    return cast(ExtensionConfig, {**config, **overrides})


def _call_private_launch(ext: Any) -> Any:
    return ext._Extension__launch()




class TestLaunchDispatchConda:

    @patch("pyisolate._internal.host.validate_backend_config")
    @patch("pyisolate._internal.host.create_conda_env")
    @patch("pyisolate._internal.host.create_venv")
    @patch("pyisolate._internal.host.install_dependencies")
    def test_conda_calls_create_conda_env(
        self,
        mock_install_deps: MagicMock,
        mock_create_venv: MagicMock,
        mock_create_conda: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        from pyisolate._internal.host import Extension
        from pyisolate.shared import ExtensionBase

        config = _make_config(
            package_manager="conda",
            conda_channels=["conda-forge"],
            conda_dependencies=["numpy"],
        )

        ext = Extension.__new__(Extension)
        ext.name = "test_ext"
        ext.config = config
        ext.venv_path = Path("/fake/venv")
        ext.module_path = "/fake/module"
        ext.extension_type = ExtensionBase

        with patch.object(ext, "_launch_with_uds", return_value=MagicMock()):
            _call_private_launch(ext)

        mock_create_conda.assert_called_once()
        mock_create_venv.assert_not_called()
        mock_install_deps.assert_not_called()


class TestEnvPropagation:

    @patch("pyisolate._internal.host.subprocess.Popen")
    def test_windows_launch_propagates_config_env(
        self,
        mock_popen: MagicMock,
    ) -> None:
        from pyisolate._internal.host import Extension
        from pyisolate.shared import ExtensionBase

        config = _make_config(
            package_manager="conda",
            execution_model="sealed_worker",
            conda_channels=["conda-forge"],
            conda_dependencies=["boltons"],
            env={"PYISOLATE_ARTIFACT_DIR": r"C:\artifacts"},
        )

        ext = Extension.__new__(Extension)
        ext.name = "test_ext"
        ext.config = config
        ext.venv_path = Path(r"C:\fake\venv")
        ext.module_path = r"C:\fake\module"
        ext.extension_type = ExtensionBase
        ext._cuda_ipc_enabled = False
        ext._uds_path = None
        ext._uds_listener = None
        ext._client_sock = None

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc
        transport = MagicMock()
        transport.send = MagicMock()

        with (
            patch(
                "pyisolate._internal.host._resolve_pixi_python",
                return_value=Path(r"C:\fake\venv\.pixi\envs\default\python.exe"),
            ),
            patch("pyisolate._internal.host.socket") as mock_socket,
            patch("pyisolate._internal.host.JSONSocketTransport", return_value=transport),
            patch("pyisolate._internal.host.AsyncRPC"),
            patch("pyisolate._internal.host.build_extension_snapshot", return_value={}),
            patch("pyisolate._internal.socket_utils.has_af_unix", return_value=False),
            patch("os.name", "nt"),
            patch("sys.platform", "win32"),
        ):
            mock_listener = MagicMock()
            mock_listener.accept.return_value = (MagicMock(), None)
            mock_listener.getsockname.return_value = ("127.0.0.1", 43210)
            mock_socket.socket.return_value = mock_listener
            mock_socket.AF_INET = 2
            mock_socket.SOCK_STREAM = 1
            mock_socket.SOL_SOCKET = 1
            mock_socket.SO_REUSEADDR = 2

            cast(Any, ext)._launch_with_uds()

        child_env = mock_popen.call_args.kwargs["env"]
        assert child_env["PYISOLATE_ARTIFACT_DIR"] == r"C:\artifacts"




class TestCondaCudaIpcForced:

    @patch("pyisolate._internal.host.create_conda_env")
    @patch("pyisolate._internal.host.validate_backend_config")
    def test_conda_forces_cuda_ipc_false(
        self,
        mock_validate: MagicMock,
        mock_conda: MagicMock,
    ) -> None:
        from pyisolate._internal.host import Extension
        from pyisolate.shared import ExtensionBase

        config = _make_config(
            share_cuda_ipc=True,
            package_manager="conda",
            conda_channels=["conda-forge"],
            conda_dependencies=["numpy"],
        )

        ext = Extension.__new__(Extension)
        ext.name = "test_ext"
        ext.config = config
        ext.venv_path = Path("/fake/venv")
        ext.module_path = "/fake/module"
        ext.extension_type = ExtensionBase
        ext._cuda_ipc_enabled = True

        with patch.object(ext, "_launch_with_uds", return_value=MagicMock()):
            _call_private_launch(ext)

        assert ext._cuda_ipc_enabled is False
        assert config["share_cuda_ipc"] is False

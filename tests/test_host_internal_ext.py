import asyncio
import logging
import queue
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from pyisolate._internal.host import Extension
from pyisolate._internal.rpc_protocol import AsyncRPC, ProxiedSingleton
from pyisolate._internal.rpc_transports import JSONSocketTransport
from pyisolate.config import ExtensionConfig


class DummyProcess:
    def __init__(self) -> None:
        self.alive = False

    def start(self) -> None:
        self.alive = True

    def is_alive(self) -> Any:
        return self.alive

    def terminate(self) -> None:
        self.alive = False

    def join(self, timeout: Any = None) -> None:
        self.alive = False

    def kill(self) -> None:
        self.alive = False


class DummyContext:
    def __init__(self) -> None:
        self.q: queue.Queue[Any] = queue.Queue()

    def Queue(self) -> Any:  # noqa: N802 - matches multiprocessing API
        return queue.Queue()

    def Process(self, target: Any, args: Any) -> Any:  # noqa: N802 - matches multiprocessing API
        return DummyProcess()


class DummyMP:
    def __init__(self) -> None:
        self.ctx = DummyContext()
        self.executable = None

    def get_context(self, mode: Any) -> Any:
        return self.ctx

    def set_executable(self, exe: Any) -> None:
        self.executable = exe


class DummyExtension(Extension):
    def __init__(self, tmp_path: Path, config_overrides: Any = None) -> None:
        base_config: dict[str, Any] = {
            "name": "demo",
            "isolated": True,
            "dependencies": [],
            "share_torch": True,
            "share_cuda_ipc": False,
            "apis": [],
        }
        if config_overrides:
            base_config.update(cast(dict[str, Any], config_overrides))
        super().__init__(
            module_path="/tmp/mod.py",
            extension_type=SimpleNamespace,
            config=cast(ExtensionConfig, base_config),
            venv_root_path=str(tmp_path),
        )
        self.mp = DummyMP()


def test_initialize_process_requires_share_torch_for_cuda_ipc(tmp_path: Any) -> None:
    ext = DummyExtension(tmp_path, {"share_torch": False, "share_cuda_ipc": True})
    with pytest.raises(RuntimeError):
        ext._initialize_process()


def test_callable_roundtrip_shutdown_is_clean(caplog: Any, capsys: Any) -> Any:
    class HostCallbackAPI(ProxiedSingleton):
        async def invoke(self, handler: Any, payload: Any) -> Any:
            return await handler(payload)

    async def scenario() -> Any:
        left, right = socket.socketpair()
        host_transport = JSONSocketTransport(left)
        child_transport = JSONSocketTransport(right)
        host_rpc = AsyncRPC(transport=host_transport)
        child_rpc = AsyncRPC(transport=child_transport)
        HostCallbackAPI()._register(host_rpc)
        host_rpc.run()
        child_rpc.run()
        caller = child_rpc.create_caller(HostCallbackAPI, HostCallbackAPI.get_remote_id())
        try:

            def handler(payload: Any) -> Any:
                return {"value": payload["value"] + 1}

            return await asyncio.wait_for(caller.invoke(handler, {"value": 41}), timeout=5)
        finally:
            child_rpc.shutdown()
            host_rpc.shutdown()
            await asyncio.sleep(0)
            child_transport.close()
            host_transport.close()
            await asyncio.sleep(0)

    with caplog.at_level(logging.ERROR):
        result = asyncio.run(scenario())

    captured = capsys.readouterr()
    assert result == {"value": 42}
    assert "InvalidStateError" not in captured.err
    assert "RPC recv failed" not in caplog.text

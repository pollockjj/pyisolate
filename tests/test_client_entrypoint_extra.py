import importlib
import sys
from types import ModuleType
from typing import Any, cast

import pytest

from pyisolate._internal import client, uds_client
from pyisolate._internal.rpc_protocol import ProxiedSingleton
from pyisolate.config import ExtensionConfig
from pyisolate.shared import ExtensionBase


class DummyExtension(ExtensionBase):
    def __init__(self) -> None:
        super().__init__()
        self.before_called = False
        self.loaded_called = False

    async def before_module_loaded(self) -> None:
        self.before_called = True

    async def on_module_loaded(self, module: ModuleType) -> None:
        self.loaded_called = True
        assert hasattr(module, "VALUE")


class FakeRPC:
    def __init__(self, recv_queue: Any = None, send_queue: Any = None) -> None:  # noqa: ARG002
        self.registered: list[tuple[Any, Any]] = []
        self.running = False

    def register_callee(self, obj: Any = None, object_id: Any = None) -> None:
        self.registered.append((obj, object_id))

    def run(self) -> None:
        self.running = True

    async def run_until_stopped(self) -> Any:
        return None


def _config(**overrides: Any) -> ExtensionConfig:
    base: dict[str, Any] = {
        "name": "demo",
        "isolated": True,
        "dependencies": [],
        "share_torch": False,
        "share_cuda_ipc": False,
        "apis": [],
    }
    return cast(ExtensionConfig, {**base, **overrides})


def _make_module(tmp_path: Any, name: str, value: int) -> Any:
    module_dir = tmp_path / name
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text(f"VALUE = {value}\n")
    return module_dir


async def _run(module_dir: Any, ext: Any, config: ExtensionConfig) -> None:
    await client.async_entrypoint(
        module_path=str(module_dir),
        extension_type=lambda: ext,  # type: ignore[arg-type]
        config=config,
        to_extension=None,
        from_extension=None,
        log_queue=None,
    )


@pytest.mark.asyncio
async def test_async_entrypoint_runs_hooks_and_registers(tmp_path: Any, monkeypatch: Any) -> Any:
    module_dir = _make_module(tmp_path, "ext", 42)
    monkeypatch.setattr(client, "AsyncRPC", FakeRPC)
    ext = DummyExtension()
    await _run(module_dir, ext, _config())
    assert ext.before_called is True
    assert ext.loaded_called is True


@pytest.mark.asyncio
async def test_async_entrypoint_rejects_missing_dir(tmp_path: Any) -> None:
    bogus = tmp_path / "notadir"
    with pytest.raises(ValueError):
        await client.async_entrypoint(
            module_path=str(bogus),
            extension_type=DummyExtension,
            config=_config(),
            to_extension=None,
            from_extension=None,
            log_queue=None,
        )


@pytest.mark.asyncio
async def test_async_entrypoint_uses_inference_mode(monkeypatch: Any, tmp_path: Any) -> Any:
    module_dir = _make_module(tmp_path, "ext2", 1)
    entered = {"count": 0}

    class DummyInference:
        def __enter__(self) -> Any:
            entered["count"] += 1
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
            return False

    class DummyTorch:
        def inference_mode(self) -> Any:
            return DummyInference()

    monkeypatch.setitem(sys.modules, "torch", DummyTorch())
    monkeypatch.setattr(client, "AsyncRPC", FakeRPC)
    ext = DummyExtension()
    await _run(module_dir, ext, _config(name="demo2", share_torch=True))
    assert entered["count"] == 1


@pytest.mark.asyncio
async def test_async_entrypoint_registers_apis_with_adapter(monkeypatch: Any, tmp_path: Any) -> Any:
    module_dir = _make_module(tmp_path, "ext3", 3)

    class DummyAPI(ProxiedSingleton):
        last_rpc: Any = None

        @classmethod
        def use_remote(cls, rpc: Any) -> None:
            cls.last_rpc = rpc

    class DummyAdapter:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, Any]] = []

        def handle_api_registration(self, api_instance: Any, rpc: Any) -> None:
            self.calls.append((api_instance, rpc))

    dummy_adapter = DummyAdapter()
    monkeypatch.setattr(client, "_adapter", dummy_adapter)
    monkeypatch.setattr(client, "AsyncRPC", FakeRPC)
    ext = DummyExtension()
    await _run(module_dir, ext, _config(name="demo3", apis=[DummyAPI]))
    assert DummyAPI.last_rpc is not None
    assert dummy_adapter.calls


def test_sealed_worker_skips_api_class_import(monkeypatch: Any) -> Any:
    config = _config(name="demo-sealed", execution_model="sealed_worker", apis=["forbidden.module.ForbiddenAPI"])
    real_import_module = importlib.import_module

    def _forbidden_import(name: str, package: str | None = None) -> Any:
        if name == "forbidden.module":
            raise AssertionError("sealed worker must not import API classes from config")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _forbidden_import)
    resolved = uds_client._resolve_api_classes_from_config(config)
    assert resolved == []

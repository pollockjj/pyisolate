from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pyisolate._internal.serialization_registry import SerializerRegistry


def _shared_memory_registry() -> SerializerRegistry:
    from pyisolate._internal.serialization_registry import SerializerRegistry
    from pyisolate._internal.tensor_serializer import register_tensor_serializer

    registry = SerializerRegistry()
    register_tensor_serializer(registry, mode="shared_memory")
    return registry


def test_serialize_for_isolation_defers_tensor_to_per_channel_transport() -> None:
    torch = pytest.importorskip("torch")

    from pyisolate._internal.model_serialization import _serialize_for_isolation_impl
    from pyisolate._internal.remote_handle import RemoteObjectHandle

    out = _serialize_for_isolation_impl(
        torch.zeros(2, 3),
        registry=_shared_memory_registry(),
        torch_module=torch,
        remote_handle_type=RemoteObjectHandle,
    )

    assert isinstance(out, torch.Tensor)
    assert not (isinstance(out, dict) and out.get("__type__") == "TensorRef")


def test_serialize_for_isolation_reads_cuda_ipc_env_at_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    from pyisolate._internal.model_serialization import _serialize_for_isolation_impl
    from pyisolate._internal.remote_handle import RemoteObjectHandle
    from pyisolate._internal.serialization_registry import SerializerRegistry

    class FakeCudaTensor:
        is_cuda = True

        def cpu(self) -> str:
            return "DOWNGRADED_TO_CPU"

    class FakeTorch:
        Tensor = FakeCudaTensor

    tensor = FakeCudaTensor()
    registry = SerializerRegistry()

    def _serialize() -> object:
        return _serialize_for_isolation_impl(
            tensor,
            registry=registry,
            torch_module=FakeTorch,
            remote_handle_type=RemoteObjectHandle,
        )

    import sys

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("PYISOLATE_ENABLE_CUDA_IPC", "1")
    assert _serialize() is tensor

    monkeypatch.setattr(sys, "platform", "win32")
    assert _serialize() == "DOWNGRADED_TO_CPU"

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("PYISOLATE_ENABLE_CUDA_IPC", raising=False)
    assert _serialize() == "DOWNGRADED_TO_CPU"


def test_prepare_for_rpc_defers_tensor_to_per_channel_transport() -> None:
    torch = pytest.importorskip("torch")

    from pyisolate._internal.rpc_serialization import _prepare_for_rpc_impl

    out = _prepare_for_rpc_impl(
        torch.zeros(2, 3),
        registry=_shared_memory_registry(),
        torch_module=torch,
    )

    assert isinstance(out, torch.Tensor)
    assert not (isinstance(out, dict) and out.get("__type__") == "TensorRef")


def test_serialize_tensor_honors_cuda_ipc_env_at_chokepoint(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")

    import sys

    from pyisolate._internal import tensor_serializer

    calls: list[str] = []

    class FakeCudaTensor:
        is_cuda = True

        def detach(self) -> FakeCudaTensor:
            return self

        def cpu(self) -> str:
            return "CPU_COPY"

    def _fake_cuda(t: object) -> dict[str, str]:
        calls.append("cuda")
        return {"__type__": "TensorRef", "device": "cuda"}

    def _fake_cpu(t: object) -> dict[str, str]:
        calls.append(f"cpu:{t}")
        return {"__type__": "TensorRef", "device": "cpu"}

    monkeypatch.setattr(tensor_serializer, "_serialize_cuda_tensor", _fake_cuda)
    monkeypatch.setattr(tensor_serializer, "_serialize_cpu_tensor", _fake_cpu)

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("PYISOLATE_ENABLE_CUDA_IPC", "1")
    assert tensor_serializer.serialize_tensor(FakeCudaTensor())["device"] == "cuda"

    calls.clear()
    monkeypatch.setattr(sys, "platform", "win32")
    out = tensor_serializer.serialize_tensor(FakeCudaTensor())
    assert out["device"] == "cpu"
    assert calls == ["cpu:CPU_COPY"]

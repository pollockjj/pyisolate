from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pyisolate._internal.serialization_registry import SerializerRegistry


def _shared_memory_registry() -> SerializerRegistry:
    """A registry whose Tensor serializer is pinned to shared_memory, simulating a
    co-resident share_torch extension that registered last on the process-global
    SerializerRegistry singleton."""
    from pyisolate._internal.serialization_registry import SerializerRegistry
    from pyisolate._internal.tensor_serializer import register_tensor_serializer

    registry = SerializerRegistry()
    register_tensor_serializer(registry, mode="shared_memory")
    return registry


def test_serialize_for_isolation_defers_tensor_to_per_channel_transport() -> None:
    """serialize_for_isolation must not pre-encode torch tensors via the registry's
    mode-bound "Tensor" serializer; the tensor must be deferred so the per-channel
    transport decides the wire format.

    Regression: a host running a shared_memory (share_torch) extension alongside a json
    (sealed/conda) extension pinned the global "Tensor" serializer to shared_memory
    (last-writer-wins). serialize_for_isolation then emitted a shared-memory TensorRef
    onto the json channel; the torch-free sealed worker raised KeyError('data') decoding
    it and its RPC recv thread died ("Socket closed").
    """
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
    """The CUDA-IPC decision in serialize_for_isolation must be evaluated at call time,
    not captured at import.

    Regression: a module-level _cuda_ipc_enabled snapshot, consulted before the registry
    serializer, was stale because the host sets PYISOLATE_ENABLE_CUDA_IPC during
    _initialize_process -- after this module is imported. With CUDA IPC configured it
    therefore sent obj.cpu() instead of deferring the on-device tensor to the per-channel
    CUDA IPC transport, silently losing CUDA transport. A fake on-device tensor isolates
    the env decision without requiring a GPU.
    """
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

    # CUDA IPC configured at runtime -> defer the on-device tensor (do NOT downgrade).
    monkeypatch.setenv("PYISOLATE_ENABLE_CUDA_IPC", "1")
    assert _serialize() is tensor

    # CUDA IPC not configured -> fall back to CPU.
    monkeypatch.delenv("PYISOLATE_ENABLE_CUDA_IPC", raising=False)
    assert _serialize() == "DOWNGRADED_TO_CPU"


def test_prepare_for_rpc_defers_tensor_to_per_channel_transport() -> None:
    """_prepare_for_rpc_impl (the RPC argument pre-pass) must likewise defer torch tensors
    instead of pre-encoding them with the global registry mode. This is the path that
    serializes execute_node arguments; the per-channel transport must choose the format.
    """
    torch = pytest.importorskip("torch")

    from pyisolate._internal.rpc_serialization import _prepare_for_rpc_impl

    out = _prepare_for_rpc_impl(
        torch.zeros(2, 3),
        registry=_shared_memory_registry(),
        torch_module=torch,
    )

    assert isinstance(out, torch.Tensor)
    assert not (isinstance(out, dict) and out.get("__type__") == "TensorRef")

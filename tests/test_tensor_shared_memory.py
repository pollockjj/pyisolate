from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pyisolate._internal.tensor_serializer import (  # noqa: E402
    _reset_shm_check,
    deserialize_tensor,
    serialize_tensor,
)


def test_cpu_torch_share_roundtrip_is_zero_copy() -> None:
    """torch_share CPU transport must stay shared-memory, never a value copy.

    Runs single-process so it executes on Windows, where the multi-process
    torch_share RPC tests are skipped (extension loading needs Unix sockets).
    Guards the regression where a /dev/shm gate degrades CPU sharing to a
    file-based value copy: a copy still passes ``torch.equal`` but breaks the
    shared-storage contract that callers depend on for zero-copy transfer.
    """
    _reset_shm_check()
    original = torch.arange(25, dtype=torch.float32).reshape(5, 5)

    payload = serialize_tensor(original, mode="shared_memory")

    assert payload["__type__"] == "TensorRef", "CPU tensor degraded to a value copy"
    assert payload["device"] == "cpu"
    assert payload["strategy"] in ("file_system", "file_system_borrowed")

    rebuilt = deserialize_tensor(payload, mode="shared_memory")
    assert torch.equal(rebuilt, original)

    original[0, 0] = 999.0
    assert float(rebuilt[0, 0]) == 999.0, "receiver did not observe sender mutation"

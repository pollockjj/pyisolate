from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pyisolate._internal.tensor_serializer import (  # noqa: E402
    _reset_shm_check,
    deserialize_tensor,
    serialize_tensor,
)


def test_cpu_torch_share_roundtrip_is_zero_copy() -> None:
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

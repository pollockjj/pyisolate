import gc
from typing import Any

import pytest
import torch  # noqa: E402

try:
    import numpy as np  # noqa: F401

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@pytest.mark.asyncio
async def test_tensor_roundtrip_cpu(reference_host: Any) -> None:
    print("\n[TEST] Starting CPU tensor roundtrip")
    ext = reference_host.load_test_extension("tensor_cpu", isolated=True)
    proxy = ext.get_proxy()

    t = torch.ones(5, 5)
    print(f"[TEST] Created tensor: {t.shape}")

    print("[TEST] Sending tensor...")
    result = await proxy.echo_tensor(t)
    print("[TEST] Tensor echoed back.")

    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, t)
    print("[TEST] CPU tensor verification passed.")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.asyncio
async def test_tensor_roundtrip_cuda(reference_host: Any) -> None:
    print("\n[TEST] Starting CUDA IPC roundtrip")
    ext = reference_host.load_test_extension("tensor_cuda_ipc", isolated=True, share_cuda=True)
    proxy = ext.get_proxy()

    t = torch.ones(5, 5, device="cuda")
    print(f"[TEST] Created CUDA tensor: {t.shape}, device={t.device}")

    print("[TEST] Sending tensor...")
    result = await proxy.echo_tensor(t)
    print(f"[TEST] Received tensor: device={result.device}")

    assert isinstance(result, torch.Tensor)
    assert result.device.type == "cuda"
    assert torch.equal(result.cpu(), t.cpu())
    del result
    del t
    gc.collect()
    torch.cuda.synchronize()
    print("[TEST] CUDA IPC verified.")

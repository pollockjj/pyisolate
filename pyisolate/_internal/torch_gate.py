from __future__ import annotations

import importlib
import os
import sys
from typing import Any


def cuda_ipc_active() -> bool:
    """True only where torch CUDA-IPC handle sharing is supported AND enabled.

    CUDA-IPC handle import is Linux-only -- importing a handle off-Linux faults in c10
    (cudaErrorDeviceUninitialized). The host sets PYISOLATE_ENABLE_CUDA_IPC during init
    from probe_cuda_ipc_support(), but the serialization chokepoints gate on sys.platform
    here too so the CPU fallback is deterministic regardless of when the env is observed.
    """
    return sys.platform == "linux" and os.environ.get("PYISOLATE_ENABLE_CUDA_IPC") == "1"


def get_torch_optional() -> tuple[Any | None, Any | None]:
    """Return (torch, torch.multiprocessing.reductions) when available.

    PyTorch is optional for base pyisolate usage. Callers that need tensor
    features should use `require_torch(...)` for explicit errors.
    """
    try:
        torch = importlib.import_module("torch")
        reductions = importlib.import_module("torch.multiprocessing.reductions")
        return torch, reductions
    except Exception:
        return None, None


def require_torch(feature_name: str) -> tuple[Any, Any]:
    """Return torch modules or raise a clear feature-scoped error."""
    torch, reductions = get_torch_optional()
    if torch is None or reductions is None:
        raise RuntimeError(f"{feature_name} requires PyTorch. Install 'torch' to use this feature.")
    return torch, reductions

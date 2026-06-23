import logging
import os
import sys
from typing import Any

from pyisolate import flush_tensor_keeper
from pyisolate.shared import ExtensionBase

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)



class ReferenceTestExtension(ExtensionBase):

    async def initialize(self) -> None:
        logger.info("[TestPkg] Initialized.")
        sys.modules["_test_ext_initialized"] = True  # type: ignore

    async def prepare_shutdown(self) -> None:
        logger.info("[TestPkg] Preparing shutdown.")

    async def stop(self) -> None:
        try:
            flush_tensor_keeper()
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
        finally:
            await super().stop()

    async def ping(self) -> str:
        return "pong"

    async def echo_tensor(self, tensor: Any) -> Any:
        if not HAS_TORCH:
            return "NO_TORCH"

        if not isinstance(tensor, torch.Tensor):
            logger.error(f"Expected Tensor, got {type(tensor)}")
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

        logger.info(f"[TestPkg] Echoing tensor: shape={tensor.shape}, device={tensor.device}")
        return tensor

    async def allocate_cuda(self, size_mb: int) -> dict[str, Any]:
        if not HAS_TORCH or not torch.cuda.is_available():
            raise RuntimeError("CUDA not available in child")

        numel = size_mb * 1024 * 1024 // 4  # float32 = 4 bytes
        t = torch.zeros(numel, device="cuda", dtype=torch.float32)

        return {
            "device": str(t.device),
            "allocated_bytes": torch.cuda.memory_allocated(),
            "tensor_shape": list(t.shape),
        }

    async def write_file(self, path: str, content: str) -> str:
        logger.info(f"[TestPkg] Attempting to write to {path}")
        with open(path, "w") as f:
            f.write(content)
        return "ok"

    async def read_file(self, path: str) -> str:
        logger.info(f"[TestPkg] Attempting to read from {path}")
        with open(path) as f:
            return f.read()

    async def crash_me(self) -> None:
        logger.info("[TestPkg] Goodbye cruel world!")
        os._exit(42)

    async def get_env_var(self, key: str) -> str | None:
        return os.environ.get(key)


def extension_entrypoint() -> ExtensionBase:
    return ReferenceTestExtension()

"""Memory leak tests for tensor lifecycle and cleanup.

These tests verify that:
1. TensorKeeper holds a tensor reference while retained
2. TensorKeeper releases tensors after timeout
3. Exceptions during __init__ do not leak a registry entry

Note: Uses weakref to verify objects are collected, not actual memory profiling.
For actual memory profiling, use tracemalloc in integration tests.
"""

import gc
import time
import weakref
from typing import Any

import pytest

from pyisolate._internal.rpc_protocol import ProxiedSingleton, SingletonMetaclass


class TestTensorKeeperCleanup:
    """Tests for TensorKeeper memory management."""

    @pytest.fixture(autouse=True)
    def fast_tensor_keeper(self, monkeypatch: Any) -> None:
        """Configure TensorKeeper with short retention for testing."""
        from pyisolate._internal.tensor_serializer import TensorKeeper

        def fast_init(self: Any, retention_seconds: float = 2.0) -> None:  # noqa: ARG001
            self.retention_seconds = 2.0
            self._keeper = __import__("collections").deque()
            self._lock = __import__("threading").Lock()

        # Use 2 second retention for fast testing
        monkeypatch.setattr(
            TensorKeeper,
            "__init__",
            fast_init,
        )

    def test_tensor_keeper_keeps_reference(self) -> None:
        """Verify TensorKeeper holds tensor reference."""
        pytest.importorskip("torch")
        import torch

        from pyisolate._internal.tensor_serializer import TensorKeeper

        keeper = TensorKeeper(retention_seconds=5.0)
        tensor = torch.zeros(10)
        weak_ref = weakref.ref(tensor)

        # Keep tensor
        keeper.keep(tensor)

        # Delete local reference
        del tensor

        # Should still exist via keeper
        gc.collect()
        assert weak_ref() is not None, "Tensor collected while keeper holds it"

    @pytest.mark.slow
    def test_tensor_keeper_releases_after_timeout(self) -> None:
        """Verify TensorKeeper releases tensors after retention period.

        Note: This test takes ~3 seconds due to retention timeout.
        """
        pytest.importorskip("torch")
        import torch

        from pyisolate._internal.tensor_serializer import TensorKeeper

        # Short retention for testing
        keeper = TensorKeeper(retention_seconds=1.0)
        tensor = torch.zeros(10)
        weak_ref = weakref.ref(tensor)

        # Keep tensor
        keeper.keep(tensor)
        del tensor

        # Should still exist immediately
        gc.collect()
        assert weak_ref() is not None

        # Wait for retention to expire
        time.sleep(2.0)

        # Trigger cleanup by adding another tensor
        keeper.keep(torch.zeros(1))

        # Force GC
        for _ in range(3):
            gc.collect()

        # Original tensor should be released
        assert weak_ref() is None, "Tensor not released after retention period"


class TestMemoryLeakScenarios:
    """Tests for specific memory leak scenarios."""

    def test_exception_during_init_no_leak(self) -> None:
        """Verify exceptions during __init__ don't leak memory."""

        class FailingService(ProxiedSingleton):
            def __init__(self) -> None:
                super().__init__()
                raise ValueError("Init failed")

        # Attempt to create (should fail)
        with pytest.raises(ValueError):
            FailingService()

        # Should not be in registry (init failed)
        assert FailingService not in SingletonMetaclass._instances

import gc
import time
import weakref
from typing import Any

import pytest

from pyisolate._internal.rpc_protocol import ProxiedSingleton, SingletonMetaclass


class TestTensorKeeperCleanup:
    @pytest.fixture(autouse=True)
    def fast_tensor_keeper(self, monkeypatch: Any) -> None:
        from pyisolate._internal.tensor_serializer import TensorKeeper

        def fast_init(self: Any, retention_seconds: float = 2.0) -> None:  # noqa: ARG001
            self.retention_seconds = 2.0
            self._keeper = __import__("collections").deque()
            self._lock = __import__("threading").Lock()

        monkeypatch.setattr(
            TensorKeeper,
            "__init__",
            fast_init,
        )

    def test_tensor_keeper_keeps_reference(self) -> None:
        pytest.importorskip("torch")
        import torch

        from pyisolate._internal.tensor_serializer import TensorKeeper

        keeper = TensorKeeper(retention_seconds=5.0)
        tensor = torch.zeros(10)
        weak_ref = weakref.ref(tensor)

        keeper.keep(tensor)

        del tensor

        gc.collect()
        assert weak_ref() is not None, "Tensor collected while keeper holds it"

    @pytest.mark.slow
    def test_tensor_keeper_releases_after_timeout(self) -> None:
        pytest.importorskip("torch")
        import torch

        from pyisolate._internal.tensor_serializer import TensorKeeper

        keeper = TensorKeeper(retention_seconds=1.0)
        tensor = torch.zeros(10)
        weak_ref = weakref.ref(tensor)

        keeper.keep(tensor)
        del tensor

        gc.collect()
        assert weak_ref() is not None

        time.sleep(2.0)

        keeper.keep(torch.zeros(1))

        for _ in range(3):
            gc.collect()

        assert weak_ref() is None, "Tensor not released after retention period"


class TestMemoryLeakScenarios:
    def test_exception_during_init_no_leak(self) -> None:

        class FailingService(ProxiedSingleton):
            def __init__(self) -> None:
                super().__init__()
                raise ValueError("Init failed")

        with pytest.raises(ValueError):
            FailingService()

        assert FailingService not in SingletonMetaclass._instances

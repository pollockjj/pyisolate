
import asyncio
from typing import Any, cast

from pyisolate._internal.rpc_protocol import ProxiedSingleton

from .fixtures.test_adapter import MockRegistry


class TestProxiedSingletonContract:

    def test_singleton_returns_same_instance(self) -> None:
        instance1 = MockRegistry()
        instance2 = MockRegistry()

        assert instance1 is instance2

    def test_different_singletons_are_independent(self) -> None:

        class AnotherRegistry(ProxiedSingleton):
            def __init__(self) -> None:
                super().__init__()
                self.data = "another"

        test_instance = MockRegistry()
        another_instance = AnotherRegistry()

        assert test_instance is not another_instance
        assert isinstance(test_instance, MockRegistry)
        assert isinstance(another_instance, AnotherRegistry)


class TestEventLoopResilience:

    def test_asyncrpc_constructs_without_current_event_loop(self) -> None:
        import queue

        from pyisolate._internal.rpc_protocol import AsyncRPC

        try:
            previous_loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            previous_loop = None

        asyncio.set_event_loop(None)
        rpc = None
        try:
            rpc = AsyncRPC(recv_queue=cast(Any, queue.Queue()), send_queue=cast(Any, queue.Queue()))
            assert isinstance(rpc.default_loop, asyncio.AbstractEventLoop)
            assert not rpc.default_loop.is_closed()
        finally:
            created = rpc.default_loop if rpc is not None else None
            asyncio.set_event_loop(previous_loop)
            if created is not None and created is not previous_loop:
                created.close()

    def test_asyncrpc_reuses_preset_thread_loop(self) -> None:
        import queue

        from pyisolate._internal.rpc_protocol import AsyncRPC

        try:
            previous_loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            previous_loop = None

        installed = asyncio.new_event_loop()
        asyncio.set_event_loop(installed)
        try:
            rpc = AsyncRPC(recv_queue=cast(Any, queue.Queue()), send_queue=cast(Any, queue.Queue()))
            assert rpc.default_loop is installed
        finally:
            asyncio.set_event_loop(previous_loop)
            installed.close()

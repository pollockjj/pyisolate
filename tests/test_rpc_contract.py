"""Tests for RPC behavior and ProxiedSingleton contracts.

These tests verify:
1. ProxiedSingleton instances are singletons
2. RPC method calls work correctly
3. Event loop recreation doesn't break RPC
4. Exceptions propagate correctly

Note: These are unit tests that verify RPC contracts at the boundary
without full process isolation. For full integration tests, see
original_integration/.
"""

import asyncio
from typing import Any, cast

import pytest

from pyisolate._internal.rpc_protocol import ProxiedSingleton

from .fixtures.test_adapter import MockRegistry


class TestProxiedSingletonContract:
    """Tests for ProxiedSingleton metaclass behavior."""

    def test_singleton_returns_same_instance(self) -> None:
        """Multiple instantiations return the same instance."""
        instance1 = MockRegistry()
        instance2 = MockRegistry()

        assert instance1 is instance2

    def test_different_singletons_are_independent(self) -> None:
        """Different ProxiedSingleton subclasses are independent."""

        class AnotherRegistry(ProxiedSingleton):
            def __init__(self) -> None:
                super().__init__()
                self.data = "another"

        test_instance = MockRegistry()
        another_instance = AnotherRegistry()

        assert test_instance is not another_instance
        assert isinstance(test_instance, MockRegistry)
        assert isinstance(another_instance, AnotherRegistry)


class TestRpcMethodContract:
    """Tests for RPC method call contract."""


class TestEventLoopResilience:
    """Tests for RPC resilience across event loop recreation.

    This is a critical contract: ProxiedSingleton instances must
    remain functional even when the event loop is closed and
    recreated (e.g., between workflow executions).
    """

    def test_singleton_survives_loop_recreation(self) -> None:
        """Singleton instance survives event loop recreation."""
        try:
            previous_loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            previous_loop = None

        loop1 = asyncio.new_event_loop()
        loop2: asyncio.AbstractEventLoop | None = None
        try:
            asyncio.set_event_loop(loop1)
            registry = MockRegistry()
            obj_id = registry.register("loop1_object")

            loop1.close()

            loop2 = asyncio.new_event_loop()
            asyncio.set_event_loop(loop2)

            result = registry.get(obj_id)
            assert result == "loop1_object"
        finally:
            asyncio.set_event_loop(previous_loop)
            if loop2 is not None:
                loop2.close()
            elif not loop1.is_closed():
                loop1.close()

    def test_asyncrpc_constructs_without_current_event_loop(self) -> None:
        """AsyncRPC must construct when no current event loop exists.

        The host launches extensions from a synchronous path (host._launch_with_uds),
        constructing AsyncRPC outside any running loop. Python >=3.12 removed implicit
        main-thread loop creation, so an eager asyncio.get_event_loop() in __init__
        raised "There is no current event loop". This guards that regression.
        """
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
        """AsyncRPC must adopt the thread's installed (set-but-not-running) loop.

        A synchronous caller may create a loop, install it via asyncio.set_event_loop(),
        construct AsyncRPC, then drive that loop. __init__ must adopt the installed loop
        (matching historical asyncio.get_event_loop() behavior) instead of creating a
        separate loop that rpc.run()/dispatch would schedule on but nobody runs.
        """
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


class TestRpcErrorHandling:
    """Tests for RPC error handling contract."""

    def test_method_exception_propagates(self) -> None:
        """Exceptions in RPC methods should propagate."""

        class FailingService(ProxiedSingleton):
            def fail(self) -> None:
                raise ValueError("Intentional failure")

        service = FailingService()

        with pytest.raises(ValueError, match="Intentional failure"):
            service.fail()

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

    def test_singleton_instance_persists(self) -> None:
        """Singleton instance persists across calls."""
        instance1 = MockRegistry()
        instance1.register("test_object")

        instance2 = MockRegistry()
        # Should see the object registered via instance1
        assert instance2.get("obj_0") == "test_object"

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

    def test_method_returns_value(self) -> None:
        """RPC method must return expected value."""
        registry = MockRegistry()
        obj = {"key": "value"}

        obj_id = registry.register(obj)
        result = registry.get(obj_id)

        assert result == obj

    def test_method_accepts_arguments(self) -> None:
        """RPC method must accept positional and keyword arguments."""
        registry = MockRegistry()

        # Positional
        id1 = registry.register("positional_arg")
        assert registry.get(id1) == "positional_arg"

    def test_method_handles_none_return(self) -> None:
        """RPC method can return None."""
        registry = MockRegistry()

        result = registry.get("nonexistent")
        assert result is None

    def test_method_handles_complex_objects(self) -> None:
        """RPC method can handle complex nested objects."""
        registry = MockRegistry()

        complex_obj = {
            "list": [1, 2, 3],
            "nested": {"a": {"b": {"c": 42}}},
            "mixed": [{"x": 1}, {"y": 2}],
        }

        obj_id = registry.register(complex_obj)
        result = registry.get(obj_id)

        assert result == complex_obj


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

    def test_asyncrpc_construction_emits_no_deprecation_warning(self) -> None:
        """AsyncRPC construction must not leak a 'no current event loop' DeprecationWarning.

        The fix for the >=3.12 get_event_loop() crash must not itself emit the very
        deprecation it works around. Treats DeprecationWarning as an error while
        constructing with no installed loop.
        """
        import queue
        import warnings

        from pyisolate._internal.rpc_protocol import AsyncRPC

        try:
            previous_loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            previous_loop = None

        asyncio.set_event_loop(None)
        rpc = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                rpc = AsyncRPC(recv_queue=cast(Any, queue.Queue()), send_queue=cast(Any, queue.Queue()))
            assert isinstance(rpc.default_loop, asyncio.AbstractEventLoop)
        finally:
            created = rpc.default_loop if rpc is not None else None
            asyncio.set_event_loop(previous_loop)
            if created is not None and created is not previous_loop:
                created.close()

    def test_run_rebinds_dispatch_to_running_loop(self) -> None:
        """run() binds default_loop to the running loop that services dispatch.

        AsyncRPC may be constructed before the loop that will run it exists (the
        synchronous host launch path), so __init__ can only install a placeholder
        loop. _recv_thread dispatches inbound calls via
        run_coroutine_threadsafe(default_loop), which executes only on a *running*
        loop; a placeholder nobody runs would hang every inbound child->host call.
        run() therefore adopts the running loop before starting the dispatch
        threads. Guards the regression where a never-run fallback loop is used as
        the dispatch target on the Python >=3.12 sync-host path.
        """
        import queue

        from pyisolate._internal.rpc_protocol import AsyncRPC

        try:
            previous_loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            previous_loop = None

        # Construct with no running/installed loop so default_loop is a placeholder
        # distinct from the loop run() will later execute under.
        asyncio.set_event_loop(None)
        recv_q: queue.Queue[Any] = queue.Queue()
        recv_q.put(None)  # makes _recv_thread exit cleanly right after run()
        rpc = AsyncRPC(recv_queue=cast(Any, recv_q), send_queue=cast(Any, queue.Queue()))
        placeholder = rpc.default_loop

        async def _run_inside_loop() -> asyncio.AbstractEventLoop:
            rpc.run()
            return asyncio.get_running_loop()

        try:
            running = asyncio.run(_run_inside_loop())
            assert rpc.default_loop is running
            assert rpc.default_loop is not placeholder
            # The created fallback loop is closed when superseded -- no leaked loop.
            assert placeholder.is_closed()
        finally:
            rpc.shutdown()
            asyncio.set_event_loop(previous_loop)
            if not placeholder.is_closed():
                placeholder.close()

    def test_singleton_data_persists_across_loops(self) -> None:
        """Data stored in singleton persists across event loops."""
        try:
            previous_loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            previous_loop = None

        loop1 = asyncio.new_event_loop()
        loop2: asyncio.AbstractEventLoop | None = None
        try:
            asyncio.set_event_loop(loop1)

            registry = MockRegistry()
            id1 = registry.register("first")
            id2 = registry.register("second")

            loop1.close()

            loop2 = asyncio.new_event_loop()
            asyncio.set_event_loop(loop2)

            assert registry.get(id1) == "first"
            assert registry.get(id2) == "second"
        finally:
            asyncio.set_event_loop(previous_loop)
            if loop2 is not None:
                loop2.close()
            elif not loop1.is_closed():
                loop1.close()


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

    def test_type_error_propagates(self) -> Any:
        """TypeError in RPC methods should propagate."""

        class TypedService(ProxiedSingleton):
            def typed_method(self, value: int) -> int:
                return value + 1

        service = TypedService()

        # Wrong type should raise TypeError
        with pytest.raises(TypeError):
            service.typed_method(cast(Any, "not an int"))

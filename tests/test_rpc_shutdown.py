"""Tests for RPC graceful shutdown behavior."""

import asyncio
from typing import Any

import pytest

from pyisolate._internal.rpc_protocol import AsyncRPC
from pyisolate._internal.rpc_transports import RPCTransport


class MockTransport(RPCTransport):
    """Mock transport that blocks on recv until closed."""

    def __init__(self) -> None:
        self.recv_future: asyncio.Future[Any] = asyncio.Future()
        self.sent_messages: list[Any] = []
        self.closed = False

    def send(self, obj: Any) -> None:
        if self.closed:
            raise RuntimeError("Transport closed")
        self.sent_messages.append(obj)

    def recv(self) -> Any:
        """Simulate blocking recv."""
        if self.closed:
            raise ConnectionError("Connection closed")
        # In a real thread this would block, but for test we
        # return a value or raise based on state
        return None  # Returning None signals end of stream in our loop

    def close(self) -> None:
        self.closed = True


class BlockingMockTransport(RPCTransport):
    """Transport that allows controlling recv blocking."""

    def __init__(self) -> None:
        self.recv_queue: asyncio.Queue[Any] = asyncio.Queue()
        self.closed = False

    def send(self, obj: Any) -> None:
        pass

    def recv(self) -> None:
        # This will be called in a thread
        if self.closed:
            raise ConnectionError("Closed")
        # Block until item available
        # Since we can't easily block in a non-async way without
        # actual threading primitives, we'll just simulate a quick
        # loop check or similar.
        # But actually, the RPC implementation calls transport.recv()
        # which is synchronous.
        import time

        while not self.closed:
            time.sleep(0.01)
        raise ConnectionError("Closed during block")

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_shutdown_cancels_run_until_stopped() -> None:
    """Test that shutdown unblocks run_until_stopped."""
    rpc = AsyncRPC(transport=MockTransport())

    # Create the future manually as run() would
    rpc.blocking_future = asyncio.Future()

    # Create a task that waits for stop
    stop_task = asyncio.create_task(rpc.run_until_stopped())

    # Give it a moment to suspend
    await asyncio.sleep(0.01)
    assert not stop_task.done()

    # Trigger shutdown
    rpc.shutdown()

    # Should be done now
    await asyncio.wait_for(stop_task, timeout=1.0)
    assert stop_task.done()


@pytest.mark.asyncio
async def test_recv_none_fails_pending_requests() -> None:
    """A recv-side sentinel should fail pending requests instead of leaving them dangling."""
    rpc = AsyncRPC(transport=MockTransport())
    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()
    rpc.pending[7] = {
        "kind": "call",
        "object_id": "obj",
        "parent_call_id": None,
        "calling_loop": loop,
        "future": future,
        "method": "ping",
        "args": (),
        "kwargs": {},
    }
    rpc.blocking_future = loop.create_future()

    rpc._recv_thread()
    await asyncio.sleep(0)

    assert future.done()
    with pytest.raises(ConnectionError, match="RPC connection closed"):
        future.result()

import asyncio
from typing import Any

import pytest

from pyisolate._internal.rpc_protocol import AsyncRPC
from pyisolate._internal.rpc_transports import RPCTransport


class MockTransport(RPCTransport):
    def __init__(self) -> None:
        self.recv_future: asyncio.Future[Any] = asyncio.Future()
        self.sent_messages: list[Any] = []
        self.closed = False

    def send(self, obj: Any) -> None:
        if self.closed:
            raise RuntimeError("Transport closed")
        self.sent_messages.append(obj)

    def recv(self) -> Any:
        if self.closed:
            raise ConnectionError("Connection closed")
        return None  # Returning None signals end of stream in our loop

    def close(self) -> None:
        self.closed = True


class BlockingMockTransport(RPCTransport):
    def __init__(self) -> None:
        self.recv_queue: asyncio.Queue[Any] = asyncio.Queue()
        self.closed = False

    def send(self, obj: Any) -> None:
        pass

    def recv(self) -> None:
        if self.closed:
            raise ConnectionError("Closed")
        import time

        while not self.closed:
            time.sleep(0.01)
        raise ConnectionError("Closed during block")

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_shutdown_cancels_run_until_stopped() -> None:
    rpc = AsyncRPC(transport=MockTransport())

    rpc.blocking_future = asyncio.Future()

    stop_task = asyncio.create_task(rpc.run_until_stopped())

    await asyncio.sleep(0.01)
    assert not stop_task.done()

    rpc.shutdown()

    await asyncio.wait_for(stop_task, timeout=1.0)
    assert stop_task.done()


@pytest.mark.asyncio
async def test_recv_none_fails_pending_requests() -> None:
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

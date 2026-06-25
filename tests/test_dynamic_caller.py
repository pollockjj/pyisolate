import asyncio
import socket
from typing import Any

from pyisolate._internal.rpc_protocol import AsyncRPC, ProxiedSingleton
from pyisolate._internal.rpc_transports import JSONSocketTransport


class PingService(ProxiedSingleton):
    async def ping(self) -> str:
        return "pong"

    async def echo(self, value: Any) -> Any:
        return value

    async def boom(self) -> None:
        raise ValueError("service exploded")


async def _paired_rpc() -> tuple[AsyncRPC, AsyncRPC, JSONSocketTransport, JSONSocketTransport]:
    left, right = socket.socketpair()
    host_transport = JSONSocketTransport(left)
    child_transport = JSONSocketTransport(right)
    host_rpc = AsyncRPC(transport=host_transport)
    child_rpc = AsyncRPC(transport=child_transport)
    PingService()._register(host_rpc)
    host_rpc.run()
    child_rpc.run()
    return host_rpc, child_rpc, host_transport, child_transport


def test_dynamic_caller_invokes_service_by_name_without_its_class() -> None:
    """The client passes only the string object_id — never imports PingService."""

    async def scenario() -> tuple[Any, Any]:
        host_rpc, child_rpc, host_t, child_t = await _paired_rpc()
        try:
            caller = child_rpc.create_dynamic_caller("PingService")
            pong = await asyncio.wait_for(caller.ping(), timeout=5)
            echoed = await asyncio.wait_for(caller.echo({"a": 1}), timeout=5)
            return pong, echoed
        finally:
            child_rpc.shutdown()
            host_rpc.shutdown()
            await asyncio.sleep(0)
            child_t.close()
            host_t.close()

    pong, echoed = asyncio.run(scenario())
    assert pong == "pong"
    assert echoed == {"a": 1}


def test_call_service_sync_from_worker_thread() -> None:
    """A synchronous caller on a worker thread (the sealed-node case) gets the result."""

    async def scenario() -> Any:
        host_rpc, child_rpc, host_t, child_t = await _paired_rpc()
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                None,
                lambda: child_rpc.call_service_sync("PingService", "ping", timeout_ms=5000),
            )
        finally:
            child_rpc.shutdown()
            host_rpc.shutdown()
            await asyncio.sleep(0)
            child_t.close()
            host_t.close()

    assert asyncio.run(scenario()) == "pong"


def test_dynamic_caller_propagates_remote_error() -> None:
    async def scenario() -> None:
        host_rpc, child_rpc, host_t, child_t = await _paired_rpc()
        try:
            caller = child_rpc.create_dynamic_caller("PingService")
            await asyncio.wait_for(caller.boom(), timeout=5)
        finally:
            child_rpc.shutdown()
            host_rpc.shutdown()
            await asyncio.sleep(0)
            child_t.close()
            host_t.close()

    import pytest

    with pytest.raises(Exception, match="service exploded"):
        asyncio.run(scenario())

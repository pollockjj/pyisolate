import asyncio
import contextlib
import socket
import struct
from collections.abc import Callable, Coroutine, Iterator
from typing import Any, cast
from unittest.mock import patch

import pytest

from pyisolate._internal.rpc_transports import JSONSocketTransport

MB = 1024 * 1024
GB = 1024 * MB


def _make_transport() -> JSONSocketTransport:
    a, b = socket.socketpair()
    b.close()
    return JSONSocketTransport(a)


def _header_then_empty(msg_len: int) -> Callable[[int], bytes]:
    header = struct.pack(">I", msg_len & 0xFFFFFFFF)
    call_count = 0

    def fake_recvall(n: int) -> bytes:
        nonlocal call_count
        call_count += 1
        return header if call_count == 1 else b""

    return fake_recvall


def _expect_recv_raises(side_effect: Any, exc: Any, match: str | None = None) -> None:
    transport = _make_transport()
    try:
        with patch.object(transport, "_recvall", side_effect=side_effect), pytest.raises(exc, match=match):
            transport.recv()
    finally:
        transport.close()


@pytest.fixture
def socket_pair() -> Iterator[tuple[JSONSocketTransport, JSONSocketTransport]]:
    a, b = socket.socketpair()
    transport_a = JSONSocketTransport(a)
    transport_b = JSONSocketTransport(b)
    try:
        yield transport_a, transport_b
    finally:
        transport_a.close()
        transport_b.close()


def test_small_message_roundtrip(socket_pair: tuple[JSONSocketTransport, JSONSocketTransport]) -> None:
    sender, receiver = socket_pair
    payload = {"kind": "call", "method": "test", "args": [1, 2, 3]}
    sender.send(payload)
    result = receiver.recv()
    assert result["kind"] == "call"
    assert result["method"] == "test"
    assert result["args"] == [1, 2, 3]


def test_callable_roundtrip_executes_via_bound_rpc(
    socket_pair: tuple[JSONSocketTransport, JSONSocketTransport],
) -> None:
    sender, receiver = socket_pair

    class FakeRPC:
        def __init__(self) -> None:
            self.callbacks: dict[str, Callable[..., Any]] = {}
            self.next_id = 0

        def register_callback(self, func: Any) -> Any:
            callback_id = f"cb-{self.next_id}"
            self.next_id += 1
            self.callbacks[callback_id] = func
            return callback_id

        async def call_callback(self, callback_id: str, *args: Any, **kwargs: Any) -> Any:
            func = self.callbacks[callback_id]
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result

    sender_rpc = FakeRPC()
    sender.bind_rpc(sender_rpc)
    receiver.bind_rpc(sender_rpc)

    def handler(payload: dict[str, int]) -> dict[str, int]:
        return {"value": payload["value"] + 1}

    sender.send({"handler": handler})
    result = receiver.recv()

    callback = cast(Callable[[dict[str, int]], Coroutine[Any, Any, dict[str, int]]], result["handler"])
    callback_result: dict[str, int] = asyncio.run(callback({"value": 41}))
    assert callback_result == {"value": 42}


def test_2gb_exact_not_rejected() -> None:
    _expect_recv_raises(_header_then_empty(2 * GB), ConnectionError)


def test_2gb_plus_1_raises_value_error() -> None:
    _expect_recv_raises(_header_then_empty(2 * GB + 1), ValueError, match="Message too large")


def test_incomplete_length_header_raises() -> None:
    _expect_recv_raises(lambda n: b"\x00\x00", ConnectionError, match="incomplete length header")


def test_incomplete_message_body_raises() -> None:
    call_count = 0

    def fake_recvall(n: int) -> bytes:
        nonlocal call_count
        call_count += 1
        return struct.pack(">I", 100) if call_count == 1 else b"short"

    _expect_recv_raises(fake_recvall, ConnectionError, match="Incomplete message")


def test_socket_closed_mid_header_raises() -> None:
    a, b = socket.socketpair()
    transport = JSONSocketTransport(a)
    b.close()
    try:
        with pytest.raises((ConnectionError, OSError)):
            transport.recv()
    finally:
        with contextlib.suppress(Exception):
            b.close()
        transport.close()

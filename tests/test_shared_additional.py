import asyncio
import queue
from typing import Any, cast

import pytest

from pyisolate._internal.rpc_protocol import (
    AsyncRPC,
)
from pyisolate._internal.rpc_serialization import (
    AttrDict,
    AttributeContainer,
    RPCPendingRequest,
    _tensor_to_cuda,
)


def test_tensor_to_cuda_attribute_container() -> None:
    obj = {
        "__pyisolate_attribute_container__": True,
        "data": {"x": {"__pyisolate_attrdict__": True, "data": {"z": 5}}},
    }
    out = _tensor_to_cuda(obj)
    assert isinstance(out, AttributeContainer)
    assert isinstance(out.x, AttrDict)
    assert out.x.z == 5


def test_async_rpc_send_thread_sets_exception_on_send_failure() -> None:
    previous_loop = None
    loop = asyncio.new_event_loop()
    try:
        previous_loop = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        previous_loop = None
    asyncio.set_event_loop(loop)

    class FailingQueue:
        def put(self, _: Any) -> None:
            raise RuntimeError("boom")

    recv_q: Any = queue.Queue()
    rpc = AsyncRPC(recv_queue=cast(Any, recv_q), send_queue=cast(Any, FailingQueue()))

    pending = RPCPendingRequest(  # type: ignore[call-arg]
        kind="call",
        object_id="obj",
        parent_call_id=None,
        calling_loop=loop,
        future=loop.create_future(),
        method="ping",
        args=(),
        kwargs={},
    )
    rpc.outbox.put(pending)
    rpc.outbox.put(None)

    try:
        rpc._send_thread()
        loop.run_until_complete(asyncio.sleep(0))
        assert pending["future"].done() is True
        with pytest.raises(RuntimeError):
            pending["future"].result()
    finally:
        asyncio.set_event_loop(previous_loop)
        loop.close()

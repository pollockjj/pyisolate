
import asyncio
from typing import Any, cast

import pytest

from pyisolate._internal.event_bridge import _EventBridge


class TestEventBridgeDispatch:

    def test_emit_event_dispatches_to_handler(self) -> None:
        bridge = _EventBridge()
        received = []

        def handler(payload: Any) -> None:
            received.append(payload)

        bridge.register_handler("progress", handler)
        asyncio.run(bridge.dispatch("progress", {"value": 5, "total": 10}))

        assert len(received) == 1
        assert received[0] == {"value": 5, "total": 10}

    def test_emit_unregistered_event_raises(self) -> None:
        bridge = _EventBridge()

        with pytest.raises(ValueError, match="No handler registered for event 'unknown_event'"):
            asyncio.run(bridge.dispatch("unknown_event", {}))

    def test_emit_event_rejects_non_json_payload(self) -> None:
        from pyisolate.shared import ExtensionLocal

        ext = ExtensionLocal()

        class NotSerializable:
            pass

        with pytest.raises(TypeError):
            ext.emit_event("progress", cast(Any, NotSerializable()))

    def test_dispatch_with_async_handler(self) -> None:
        bridge = _EventBridge()
        received = []

        async def async_handler(payload: Any) -> None:
            received.append(payload)

        bridge.register_handler("test", async_handler)
        asyncio.run(bridge.dispatch("test", {"key": "value"}))

        assert received == [{"key": "value"}]


class TestApiSurface:
    """Tests that the event channel API exists on the right classes."""

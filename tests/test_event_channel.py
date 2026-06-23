"""Tests for the pyisolate event channel (emit_event / register_event_handler).

Tests verify:
1. Events dispatch from child to host handler
2. Unregistered events raise
3. Non-JSON payloads are rejected
4. API surface exists on ExtensionBase and SealedNodeExtension
"""

import asyncio
from typing import Any, cast

import pytest

from pyisolate._internal.event_bridge import _EventBridge


class TestEventBridgeDispatch:
    """Tests for _EventBridge RPC callee behavior."""

    def test_emit_event_dispatches_to_handler(self) -> None:
        """emit_event("progress", payload) calls the registered handler with exact payload."""
        bridge = _EventBridge()
        received = []

        def handler(payload: Any) -> None:
            received.append(payload)

        bridge.register_handler("progress", handler)
        asyncio.run(bridge.dispatch("progress", {"value": 5, "total": 10}))

        assert len(received) == 1
        assert received[0] == {"value": 5, "total": 10}

    def test_emit_unregistered_event_raises(self) -> None:
        """emit_event("unknown_event", {}) raises ValueError, not silently dropped."""
        bridge = _EventBridge()

        with pytest.raises(ValueError, match="No handler registered for event 'unknown_event'"):
            asyncio.run(bridge.dispatch("unknown_event", {}))

    def test_emit_event_rejects_non_json_payload(self) -> None:
        """emit_event with non-JSON-serializable payload raises immediately."""
        from pyisolate.shared import ExtensionLocal

        ext = ExtensionLocal()

        # ExtensionLocal.emit_event does json.dumps(payload) before RPC call
        # Create a non-serializable object
        class NotSerializable:
            pass

        with pytest.raises(TypeError):
            ext.emit_event("progress", cast(Any, NotSerializable()))

    def test_dispatch_with_async_handler(self) -> None:
        """Async handlers are awaited correctly."""
        bridge = _EventBridge()
        received = []

        async def async_handler(payload: Any) -> None:
            received.append(payload)

        bridge.register_handler("test", async_handler)
        asyncio.run(bridge.dispatch("test", {"key": "value"}))

        assert received == [{"key": "value"}]


class TestApiSurface:
    """Tests that the event channel API exists on the right classes."""

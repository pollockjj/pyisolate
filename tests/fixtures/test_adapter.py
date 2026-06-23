"""Reference IsolationAdapter implementation for testing.

Provides a complete, working IsolationAdapter that lets pyisolate unit tests
run without a host application, and serves as the canonical example for host
implementers.

Example Usage::

    from tests.fixtures.test_adapter import MockHostAdapter, MockRegistry

    adapter = MockHostAdapter(root_path="/tmp/myapp")
    config = adapter.get_path_config("/tmp/myapp/extensions/myext/__init__.py")
    adapter.register_serializers(SerializerRegistry.get_instance())
    services = adapter.provide_rpc_services()
"""

from __future__ import annotations

from typing import Any

from pyisolate._internal.rpc_protocol import AsyncRPC, ProxiedSingleton
from pyisolate.interfaces import IsolationAdapter, SerializerRegistryProtocol


class MockTestData:
    """Example custom type for serialization testing."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MockTestData):
            return False
        return bool(self.value == other.value)

    def __repr__(self) -> str:
        return f"MockTestData({self.value!r})"


class MockRegistry(ProxiedSingleton):
    """Example ProxiedSingleton exposing a registry service to extensions via RPC."""

    def __init__(self) -> None:
        super().__init__()
        self._store: dict[str, Any] = {}
        self._counter = 0

    def register(self, obj: Any) -> str:
        """Register an object and return its unique string ID."""
        obj_id = f"obj_{self._counter}"
        self._counter += 1
        self._store[obj_id] = obj
        return obj_id

    def get(self, obj_id: str) -> Any:
        """Retrieve an object by ID, or None if not found."""
        return self._store.get(obj_id)

    def clear(self) -> None:
        """Clear all stored objects."""
        self._store.clear()
        self._counter = 0


class MockHostAdapter(IsolationAdapter):
    """Reference adapter implementation for testing and documentation.

    Args:
        root_path: The root directory for this host application.
    """

    def __init__(self, root_path: str = "/tmp/testhost") -> None:
        self._root = root_path
        self._extensions_dir = f"{root_path}/extensions"

    @property
    def identifier(self) -> str:
        """Return unique adapter identifier."""
        return "testhost"

    def get_path_config(self, module_path: str) -> dict[str, Any]:
        """Compute sys.path configuration for an extension.

        Returns a dict with ``preferred_root`` (the host root prepended to
        sys.path) and ``additional_paths`` (extra paths after the root).
        """
        return {
            "preferred_root": self._root,
            "additional_paths": [self._extensions_dir],
        }

    def setup_child_environment(self, snapshot: dict[str, Any]) -> None:
        """Configure the child process after sys.path reconstruction."""

    def register_serializers(self, registry: SerializerRegistryProtocol) -> None:
        """Register custom type serializers for RPC transport."""
        registry.register(
            "MockTestData",
            serializer=lambda d: {"__testdata__": True, "value": d.value},
            deserializer=lambda d: MockTestData(d["value"]) if d.get("__testdata__") else d,
        )

    def provide_rpc_services(self) -> list[type[ProxiedSingleton]]:
        """Return ProxiedSingleton classes to expose via RPC."""
        return [MockRegistry]

    def handle_api_registration(self, api: ProxiedSingleton, rpc: AsyncRPC) -> None:
        """Post-registration hook for API-specific setup."""
        return None

    def setup_web_directory(self, module: Any) -> None:
        return None

    def setup_child_event_hooks(self, rpc: AsyncRPC) -> None:
        return None

    def get_sandbox_system_paths(self) -> list[str]:
        return []

    def get_sandbox_gpu_patterns(self) -> list[str]:
        return []

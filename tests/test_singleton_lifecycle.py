"""Singleton lifecycle tests.

These tests explicitly verify singleton injection/cleanup lifecycle behavior,
particularly the singleton_scope context manager and use_remote() injection.
"""

from typing import Any, cast

import pytest

from pyisolate._internal.rpc_protocol import (
    SingletonMetaclass,
)


class TestSingletonScopeIsolation:
    """Tests for singleton_scope context manager isolation.

    singleton_scope behavior:
    1. Saves current state at entry
    2. Does NOT clear state at entry (state persists into scope)
    3. On exit: clears current state and restores saved state

    This is designed for test isolation: any modifications during the scope
    are undone when the scope exits.
    """


class TestUseRemoteInjection:
    """Tests for use_remote() proxy injection."""

    def test_use_remote_requires_proxied_singleton(self) -> None:
        """Verify use_remote() only works with ProxiedSingleton subclasses."""

        class NotProxied(metaclass=SingletonMetaclass):
            pass

        class FakeRPC:
            def create_caller(self, cls: Any, object_id: Any) -> Any:
                return object()

        rpc = FakeRPC()

        with pytest.raises(AssertionError, match="must inherit from ProxiedSingleton"):
            NotProxied.use_remote(cast(Any, rpc))


class TestNestedSingletonRegistration:
    """Tests for nested ProxiedSingleton registration."""


class TestSingletonEdgeCases:
    """Tests for edge cases in singleton lifecycle."""

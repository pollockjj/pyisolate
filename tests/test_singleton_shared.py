
from collections.abc import Generator
from typing import Any, cast

import pytest

from pyisolate._internal.rpc_protocol import (
    LocalMethodRegistry,
    ProxiedSingleton,
    SingletonMetaclass,
    local_execution,
)


@pytest.fixture(autouse=True)
def reset_singleton_state() -> Generator[None, None, None]:
    LocalMethodRegistry._instance = None
    yield
    LocalMethodRegistry._instance = None


class FakeCaller:

    def __init__(self, target_cls: Any, object_id: Any) -> None:
        self.target_cls = target_cls
        self.object_id = object_id
        self.child: Any = None


class FakeRPC:

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any, FakeCaller]] = []

    def create_caller(self, cls: Any, object_id: Any) -> Any:
        caller = FakeCaller(cls, object_id)
        self.calls.append((cls, object_id, caller))
        return caller


class BasicSingleton(ProxiedSingleton):
    async def ping(self) -> Any:  # pragma: no cover - method invoked via proxy
        return "pong"


class LocalMethodSingleton(ProxiedSingleton):
    def __init__(self) -> None:
        super().__init__()
        self.counter = 0

    @local_execution
    def increment(self) -> Any:
        self.counter += 1
        return self.counter


class ChildSingleton(ProxiedSingleton):
    async def child_call(self) -> Any:  # pragma: no cover
        return "child"


class ParentSingleton(ProxiedSingleton):
    child: ChildSingleton

    async def parent_call(self) -> Any:  # pragma: no cover
        return "parent"


class TestSingletonMetaclass:
    def test_inject_instance_after_instantiation_raises(self) -> None:
        BasicSingleton()
        with pytest.raises(AssertionError):
            SingletonMetaclass.inject_instance(BasicSingleton, object())


class TestUseRemote:
    def test_use_remote_sets_proxy_instance(self) -> None:
        rpc = FakeRPC()
        BasicSingleton.use_remote(cast(Any, rpc))

        assert BasicSingleton in SingletonMetaclass._instances
        proxy = SingletonMetaclass._instances[BasicSingleton]
        assert isinstance(proxy, FakeCaller)
        assert proxy.target_cls is BasicSingleton
        assert rpc.calls[0][1] == BasicSingleton.get_remote_id()

    def test_local_execution_methods_registered(self) -> None:
        rpc = FakeRPC()
        LocalMethodSingleton.use_remote(cast(Any, rpc))

        registry = LocalMethodRegistry.get_instance()
        assert registry.is_local_method(LocalMethodSingleton, "increment")

        local_impl = registry.get_local_method(LocalMethodSingleton, "increment")
        assert local_impl() == 1
        assert local_impl() == 2  # local state should be preserved per process

    def test_nested_singletons_receive_callers(self) -> None:
        rpc = FakeRPC()
        ParentSingleton.use_remote(cast(Any, rpc))

        parent_proxy = SingletonMetaclass._instances[ParentSingleton]
        assert isinstance(parent_proxy, FakeCaller)

        assert len(rpc.calls) == 2
        _, child_object_id, child_proxy = rpc.calls[-1]
        assert child_object_id == ChildSingleton.get_remote_id()
        assert isinstance(child_proxy, FakeCaller)

        assert parent_proxy.child is child_proxy


class TestLocalMethodRegistry:
    def test_get_local_method_requires_registration(self) -> None:
        registry = LocalMethodRegistry.get_instance()
        with pytest.raises(ValueError):
            registry.get_local_method(BasicSingleton, "ping")

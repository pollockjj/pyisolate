
import pytest

from pyisolate._internal.model_serialization import deserialize_from_isolation
from pyisolate._internal.serialization_registry import SerializerRegistry


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    SerializerRegistry.get_instance().clear()


class TestDictGuard:
    async def test_dict_with_registered_handler_calls_deserializer(self) -> None:
        registry = SerializerRegistry.get_instance()
        sentinel = object()
        registry.register("dict", lambda x: x, lambda x: sentinel)
        result = await deserialize_from_isolation({})
        assert result is sentinel

    async def test_non_dict_object_skips_deserializer(self) -> None:
        class Foo:
            pass

        called = False

        def bad_deserializer(x: object) -> object:
            nonlocal called
            called = True
            return x

        registry = SerializerRegistry.get_instance()
        registry.register("Foo", lambda x: x, bad_deserializer)

        foo = Foo()
        result = await deserialize_from_isolation(foo)
        assert result is foo
        assert not called


class TestRefTypeDeserialization:
    async def test_dict_ref_type_uses_registered_deserializer(self) -> None:
        registry = SerializerRegistry.get_instance()
        sentinel = object()
        registry.register("MyRef", lambda x: x, lambda x: sentinel)
        result = await deserialize_from_isolation({"__type__": "MyRef", "id": "abc"})
        assert result is sentinel

    async def test_nested_dict_ref_deserialization(self) -> None:
        registry = SerializerRegistry.get_instance()
        sentinel = object()
        registry.register("MyRef", lambda x: x, lambda x: sentinel)
        result = await deserialize_from_isolation({"a": {"__type__": "MyRef", "id": "xyz"}})
        assert isinstance(result, dict)
        assert result["a"] is sentinel


class TestContainerPassthrough:
    async def test_list_items_deserialized(self) -> None:
        registry = SerializerRegistry.get_instance()
        sentinel = object()
        registry.register("MyRef", lambda x: x, lambda x: sentinel)
        result = await deserialize_from_isolation([{"__type__": "MyRef"}, 1, "str"])
        assert result[0] is sentinel
        assert result[1] == 1
        assert result[2] == "str"

    async def test_tuple_preserved_as_tuple(self) -> None:
        result = await deserialize_from_isolation((1, 2, 3))
        assert isinstance(result, tuple)
        assert result == (1, 2, 3)


class TestOpaqueHandlePreservation:

    async def test_deserialize_preserves_opaque_handle_no_rpc(self) -> None:
        from unittest.mock import AsyncMock

        from pyisolate._internal.remote_handle import RemoteObjectHandle

        handle = RemoteObjectHandle("test-id", "UnregisteredType")
        mock_extension = AsyncMock()

        result = await deserialize_from_isolation(handle, extension=mock_extension)

        assert result is handle
        mock_extension.get_remote_object.assert_not_called()

    async def test_flush_clears_remote_objects(self) -> None:
        from pyisolate.sealed import SealedNodeExtension

        ext = SealedNodeExtension()
        for i in range(5):
            ext.remote_objects[f"obj-{i}"] = object()
        assert len(ext.remote_objects) == 5

        await ext.flush_transport_state()

        assert len(ext.remote_objects) == 0

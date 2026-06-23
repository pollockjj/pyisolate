import pytest

from pyisolate._internal.serialization_registry import SerializerRegistry


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    SerializerRegistry.get_instance().clear()


def test_register_and_lookup() -> None:
    registry = SerializerRegistry.get_instance()
    registry.register("Foo", lambda x: {"v": x}, lambda x: x["v"])

    assert registry.has_handler("Foo")
    serializer = registry.get_serializer("Foo")
    deserializer = registry.get_deserializer("Foo")

    payload = serializer(123) if serializer else None
    assert payload == {"v": 123}
    assert deserializer(payload) == 123 if deserializer else False


class TestDataTypeFlag:
    def test_data_type_cross_type_isolation(self) -> None:
        registry = SerializerRegistry.get_instance()
        registry.register("TypeA", lambda x: x, data_type=True)
        registry.register("TypeB", lambda x: x)
        assert registry.is_data_type("TypeA")
        assert not registry.is_data_type("TypeB")


from pyisolate.path_helpers import (
    build_child_sys_path,
    serialize_host_snapshot,
)


class TestSerializeHostSnapshot:

    def test_snapshot_contains_required_keys(self) -> None:
        snapshot = serialize_host_snapshot()

        assert "sys_path" in snapshot
        assert "sys_executable" in snapshot
        assert "sys_prefix" in snapshot
        assert "environment" in snapshot

        assert isinstance(snapshot["sys_path"], list)
        assert isinstance(snapshot["sys_executable"], str)
        assert isinstance(snapshot["sys_prefix"], str)
        assert isinstance(snapshot["environment"], dict)


class TestBuildChildSysPath:

    def test_preserves_host_order(self) -> None:
        host = ["/host/lib1", "/host/lib2", "/host/lib3"]
        extras: list[str] = ["/venv/lib"]

        result = build_child_sys_path(host, extras)

        assert result[:3] == host
        assert result[3] == extras[0]

    def test_removes_duplicates(self) -> None:
        host = ["/host/lib", "/host/lib2", "/host/lib"]
        extras = ["/venv/lib"]

        result = build_child_sys_path(host, extras)

        assert result.count("/host/lib") == 1
        assert result[0] == "/host/lib"

    def test_inserts_preferred_root_first_when_missing(self) -> None:
        host = ["/host/lib1", "/host/lib2"]
        extras = ["/venv/lib"]
        preferred = "/myapp/root"

        result = build_child_sys_path(host, extras, preferred)

        assert result[0] == preferred
        assert result[1:3] == host

    def test_filtered_subdirs_removes_named_dirs_when_provided(self) -> None:
        root = "/myapp/root"
        host = [f"{root}/comfy", f"{root}/app", "/host/lib"]
        extras = ["/venv/lib"]

        result = build_child_sys_path(host, extras, root, filtered_subdirs=["comfy", "app"])

        assert result[0] == root
        assert f"{root}/comfy" not in result
        assert f"{root}/app" not in result
        assert "/host/lib" in result


class TestIntegration:
    """Integration tests combining snapshot + path building."""

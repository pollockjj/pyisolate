import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from pyisolate._internal.sandbox_detect import detect_sandbox_capability

_SANDBOX_AVAILABLE = False
if sys.platform == "linux":
    _SANDBOX_AVAILABLE = detect_sandbox_capability().available

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        not _SANDBOX_AVAILABLE,
        reason="filesystem barrier checks require a working Linux bubblewrap sandbox",
    ),
]


@pytest.mark.asyncio
async def test_filesystem_barrier(reference_host: Any) -> None:
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("sensitive data")
        sensitive_path = f.name

    try:
        ext = reference_host.load_test_extension("fs_test", isolated=True)
        proxy = ext.get_proxy()

        try:
            await proxy.write_file("/etc/hosts", "hacked")
            write_succeeded = True
        except Exception:
            write_succeeded = False

        assert not write_succeeded, "Child should NOT be able to write to /etc/hosts"

    finally:
        if os.path.exists(sensitive_path):
            os.unlink(sensitive_path)


@pytest.mark.asyncio
async def test_module_path_ro(reference_host: Any) -> None:
    ext = reference_host.load_test_extension("ro_test", isolated=True)
    proxy = ext.get_proxy()

    test_file = f"{ext.module_path}/hacked.txt"
    try:
        await proxy.write_file(test_file, "hacked")
        write_success = True
    except Exception:
        write_success = False

    assert not write_success, "Module path should be mounted Read-Only"


@pytest.mark.asyncio
async def test_host_tmp_marker_hidden_from_child(reference_host: Any) -> None:
    host_marker = Path(tempfile.mkstemp(prefix="pyisolate_host_tmp_", dir="/tmp")[1])
    child_scratch = "/tmp/child_scratch.txt"

    try:
        host_marker.write_text("host-only", encoding="utf-8")

        ext = reference_host.load_test_extension("tmp_privacy", isolated=True)
        proxy = ext.get_proxy()

        with pytest.raises(Exception, match="No such file or directory"):
            await proxy.read_file(str(host_marker))

        assert await proxy.write_file(child_scratch, "child-only") == "ok"
        assert await proxy.read_file(child_scratch) == "child-only"
        assert not Path(child_scratch).exists(), "Child /tmp scratch leaked into host /tmp"
    finally:
        if host_marker.exists():
            host_marker.unlink()

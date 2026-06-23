from typing import Any

import pytest

from pyisolate._internal import bootstrap


def test_bootstrap_malformed_snapshot_fails(monkeypatch: Any) -> None:
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", "{invalid_json")

    with pytest.raises(ValueError, match="Failed to decode PYISOLATE_HOST_SNAPSHOT"):
        bootstrap.bootstrap_child()


def test_bootstrap_missing_adapter_ref_fails(monkeypatch: Any) -> None:
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", '{"sys_path": []}')

    adapter = bootstrap.bootstrap_child()
    assert adapter is None


def test_bootstrap_bad_adapter_ref_fails(monkeypatch: Any) -> None:

    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", '{"adapter_ref": "bad.module:BadClass"}')

    with pytest.raises(ValueError, match="Snapshot contained adapter info but adapter could not be loaded"):
        bootstrap.bootstrap_child()

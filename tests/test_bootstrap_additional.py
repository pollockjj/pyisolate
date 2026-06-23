from typing import Any

import pytest

from pyisolate._internal import bootstrap


def test_bootstrap_child_snapshot_file_errors(tmp_path: Any, monkeypatch: Any) -> None:
    snap_path = tmp_path / "bad.json"
    snap_path.write_text("not-json")
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", str(snap_path))
    with pytest.raises(ValueError):
        bootstrap.bootstrap_child()


def test_bootstrap_child_missing_file_graceful(tmp_path: Any, monkeypatch: Any) -> None:
    snap_path = tmp_path / "missing.json"
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", str(snap_path))
    assert bootstrap.bootstrap_child() is None

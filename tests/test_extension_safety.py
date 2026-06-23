
from typing import Any

import pytest

from pyisolate._internal import host


class TestValidatePathWithinRoot:
    def test_allows_path_inside_root(self, tmp_path: Any) -> None:
        root = tmp_path
        inside = root / "child" / "module"
        inside.mkdir(parents=True)
        host.validate_path_within_root(inside, root)  # should not raise

    def test_rejects_path_outside_root(self, tmp_path: Any) -> None:
        root = tmp_path / "root"
        other = tmp_path / "other"
        root.mkdir()
        other.mkdir()
        with pytest.raises(ValueError):
            host.validate_path_within_root(other, root)

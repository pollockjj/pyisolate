"""Unit tests for sandbox capability detection.

Tests cover:
- Sysctl file reading
- RHEL/Ubuntu restriction detection
- SELinux and hardened kernel checks
- bwrap binary invocation
- Error classification
- Full detection flow
"""

import sys
from unittest.mock import patch

import pytest

from pyisolate._internal.sandbox_detect import (
    RestrictionModel,
    detect_sandbox_capability,
)


class TestSysctlReaders:
    """Test low-level sysctl reading functions."""


class TestKernelChecks:
    """Test kernel feature detection."""


class TestBwrapInvocation:
    """Test bwrap binary invocation and error handling."""


class TestErrorClassification:
    """Test error message classification."""


class TestFullDetection:
    """Integration tests for full detection flow."""

    def test_platform_check_non_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that non-Linux platforms return PLATFORM_UNSUPPORTED."""
        monkeypatch.setattr(sys, "platform", "darwin")
        cap = detect_sandbox_capability()
        assert cap.available is False
        assert cap.restriction_model == RestrictionModel.PLATFORM_UNSUPPORTED
        assert "darwin" in cap.remediation

    def test_bwrap_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that missing bwrap binary returns BWRAP_MISSING."""
        monkeypatch.setattr(sys, "platform", "linux")
        with patch("shutil.which", return_value=None):
            cap = detect_sandbox_capability()
            assert cap.available is False
            assert cap.restriction_model == RestrictionModel.BWRAP_MISSING
            assert "bubblewrap" in cap.remediation.lower()

    def test_full_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test full detection success path."""
        monkeypatch.setattr(sys, "platform", "linux")
        with (
            patch("shutil.which", return_value="/usr/bin/bwrap"),
            patch(
                "pyisolate._internal.sandbox_detect._check_rhel_restriction",
                return_value=False,
            ),
            patch(
                "pyisolate._internal.sandbox_detect._test_bwrap",
                return_value=(True, ""),
            ),
        ):
            cap = detect_sandbox_capability()
            assert cap.available is True
            assert cap.restriction_model == RestrictionModel.NONE
            assert cap.bwrap_path == "/usr/bin/bwrap"
            assert cap.remediation == ""

    def test_ubuntu_apparmor_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Ubuntu AppArmor detection with degraded-mode fallback."""
        monkeypatch.setattr(sys, "platform", "linux")
        with (
            patch("shutil.which", return_value="/usr/bin/bwrap"),
            patch(
                "pyisolate._internal.sandbox_detect._check_rhel_restriction",
                return_value=False,
            ),
            patch(
                "pyisolate._internal.sandbox_detect._test_bwrap",
                return_value=(False, "Permission denied: uid map"),
            ),
            patch(
                "pyisolate._internal.sandbox_detect._check_ubuntu_apparmor_restriction",
                return_value=True,
            ),
            patch(
                "pyisolate._internal.sandbox_detect._test_bwrap_degraded",
                return_value=(True, ""),
            ),
        ):
            cap = detect_sandbox_capability()
            assert cap.available is True
            assert cap.restriction_model == RestrictionModel.UBUNTU_APPARMOR
            assert "apparmor" in cap.remediation.lower()
            assert cap.raw_error == "Permission denied: uid map"


class TestRestrictionModelEnum:
    """Test RestrictionModel enum values."""


import pytest

from pyisolate._internal.host import normalize_extension_name, validate_dependency


class TestSecurityValidation:

    def test_normalize_extension_name_dangerous_chars(self) -> None:
        test_cases = [
            ("ext|pipe", "ext_pipe"),
            ("ext`backtick`", "ext_backtick"),
            ("ext$(command)", "ext_command"),
            ("ext&background", "ext_background"),
            ("ext>redirect", "ext_redirect"),
            ("ext<redirect", "ext_redirect"),
            ("ext'quote'", "ext_quote"),
            ('ext"quote"', "ext_quote"),
            ("ext!history", "ext_history"),
            ("ext{brace}", "ext_brace"),
            ("ext[glob]", "ext_glob"),
            ("ext*star", "ext_star"),
            ("ext?question", "ext_question"),
            ("ext#comment", "ext_comment"),
            ("ext=equals", "ext_equals"),
            ("ext,comma", "ext_comma"),
        ]
        for input_name, expected in test_cases:
            assert normalize_extension_name(input_name) == expected

    def test_normalize_extension_name_path_traversal(self) -> None:
        test_cases = [
            ("../evil", "evil"),  # Dots at start removed
            ("./hidden", "hidden"),  # Dots at start removed
            ("/absolute/path", "absolute_path"),  # Slashes replaced
            ("..\\windows\\path", "windows_path"),  # Backslashes replaced
            ("ext/../../../tmp/test", "ext_tmp_test"),
            ("...dots", "dots"),  # Leading dots removed
        ]
        for input_name, expected in test_cases:
            assert normalize_extension_name(input_name) == expected

    def test_validate_dependency_invalid(self) -> None:
        invalid_cases = [
            ("--extra-index-url", "cannot start with '-'"),
            ("--trusted-host", "cannot start with '-'"),
            ("-f http://example.com", "cannot start with '-'"),
            ("numpy && echo test", "dangerous character: '&&'"),
            ("numpy || echo test", "dangerous character: '||'"),
            ("numpy | echo test", r"dangerous character: '\|'"),
            ("numpy`echo test`", "dangerous character: '`'"),
            ("numpy$(echo test)", r"dangerous character: '\$'"),
            ("numpy\ntest-package", "dangerous character: '\\n'"),
            ("numpy\rtest", "dangerous character: '\\r'"),
            ("numpy\x00test", "dangerous character: '\\x00'"),
        ]

        for dep, expected_msg in invalid_cases:
            with pytest.raises(ValueError, match=expected_msg):
                validate_dependency(dep)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

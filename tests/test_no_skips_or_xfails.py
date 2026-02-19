"""
Anti-cheat guard: ensures no test is skipped or xfail-ed.

Scans all Python files under tests/ for forbidden patterns that would
hide real test failures. This test MUST pass before any merge.
"""

import re
from pathlib import Path

# Forbidden patterns that indicate test skipping
FORBIDDEN_PATTERNS = [
    r'pytest\.mark\.skip',
    r'pytest\.mark\.xfail',
    r'pytest\.skip\s*\(',
    r'@pytest\.mark\.xfail',
    r'unittest\.skip',
]

# This file is allowed to contain the patterns above (it's checking for them)
THIS_FILE = Path(__file__).resolve()


def test_no_skips_or_xfails_in_tests():
    """Fail if any test file contains skip/xfail markers."""
    tests_dir = Path(__file__).resolve().parent
    violations = []

    for py_file in tests_dir.rglob("*.py"):
        if py_file.resolve() == THIS_FILE:
            continue

        content = py_file.read_text(encoding="utf-8", errors="replace")
        for pattern in FORBIDDEN_PATTERNS:
            matches = list(re.finditer(pattern, content))
            for match in matches:
                line_no = content[:match.start()].count('\n') + 1
                violations.append(
                    f"{py_file.relative_to(tests_dir)}:{line_no}  "
                    f"found '{match.group()}'"
                )

    assert not violations, (
        "Found skip/xfail markers in test files. "
        "Fix the implementation instead of skipping tests.\n"
        + "\n".join(violations)
    )

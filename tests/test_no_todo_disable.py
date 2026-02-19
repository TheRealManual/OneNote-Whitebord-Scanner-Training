"""
Anti-cheat guard: ensures no test contains TODO disable, TEMP, HACK,
'commented out', or 'return True # fix' patterns that would mask failures.
"""

import re
from pathlib import Path

FORBIDDEN_PATTERNS = [
    r'TODO\s*disable',
    r'#\s*TEMP\b',          # Only match TEMP in comments (e.g. "# TEMP workaround")
    r'#\s*HACK\b',          # Only match HACK in comments (e.g. "# HACK fix later")
    r'commented\s+out',
    r'return\s+True\s*#\s*fix',
]

THIS_FILE = Path(__file__).resolve()


def test_no_todo_disable_patterns_in_tests():
    """Fail if any test file contains TODO disable / TEMP / HACK workarounds."""
    tests_dir = Path(__file__).resolve().parent
    violations = []

    for py_file in tests_dir.rglob("*.py"):
        if py_file.resolve() == THIS_FILE:
            continue

        content = py_file.read_text(encoding="utf-8", errors="replace")
        for pattern in FORBIDDEN_PATTERNS:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            for match in matches:
                line_no = content[:match.start()].count('\n') + 1
                violations.append(
                    f"{py_file.relative_to(tests_dir)}:{line_no}  "
                    f"found '{match.group()}'"
                )

    assert not violations, (
        "Found forbidden TODO/TEMP/HACK patterns in test files. "
        "Fix properly instead of disabling.\n"
        + "\n".join(violations)
    )

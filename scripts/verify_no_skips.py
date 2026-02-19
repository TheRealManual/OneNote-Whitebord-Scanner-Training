"""
Verify no test skipping exists anywhere in the test suite.
Run standalone: python scripts/verify_no_skips.py
"""

import re
import sys
from pathlib import Path

FORBIDDEN_PATTERNS = [
    r'pytest\.mark\.skip',
    r'pytest\.mark\.xfail',
    r'pytest\.skip\s*\(',
    r'@pytest\.mark\.xfail',
    r'unittest\.skip',
]

# Files that are allowed to reference these patterns (they scan for them)
ALLOWED_FILES = {
    'test_no_skips_or_xfails.py',
    'verify_no_skips.py',
}


def main():
    repo_root = Path(__file__).resolve().parent.parent
    tests_dir = repo_root / "tests"

    if not tests_dir.exists():
        print("ERROR: tests/ directory not found")
        sys.exit(1)

    violations = []
    for py_file in tests_dir.rglob("*.py"):
        if py_file.name in ALLOWED_FILES:
            continue

        content = py_file.read_text(encoding="utf-8", errors="replace")
        for pattern in FORBIDDEN_PATTERNS:
            matches = list(re.finditer(pattern, content))
            for match in matches:
                line_no = content[:match.start()].count('\n') + 1
                violations.append(
                    f"{py_file.relative_to(repo_root)}:{line_no}  "
                    f"found '{match.group()}'"
                )

    if violations:
        print("FAIL: Found skip/xfail markers in test files:")
        for v in violations:
            print(f"  {v}")
        sys.exit(1)
    else:
        print("OK: No skip/xfail markers found in tests.")
        sys.exit(0)


if __name__ == "__main__":
    main()

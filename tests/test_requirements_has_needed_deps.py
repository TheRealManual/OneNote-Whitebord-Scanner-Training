"""
Branch 0 test: verify requirements.txt contains all needed dependencies.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_DEPS = [
    "opencv-python",
    "matplotlib",
    "tqdm",
    "torch",
    "torchvision",
    "pillow",
    "numpy",
    "scipy",
]


def test_requirements_has_needed_deps():
    """requirements.txt must list all runtime dependencies."""
    req_path = PROJECT_ROOT / "requirements.txt"
    assert req_path.exists(), "requirements.txt not found"

    content = req_path.read_text(encoding="utf-8").lower()

    missing = []
    for dep in REQUIRED_DEPS:
        # Check for the dep name (case-insensitive), allowing version specifiers
        if dep.lower() not in content:
            missing.append(dep)

    assert not missing, (
        f"requirements.txt is missing these dependencies: {missing}\n"
        f"Current content:\n{content}"
    )

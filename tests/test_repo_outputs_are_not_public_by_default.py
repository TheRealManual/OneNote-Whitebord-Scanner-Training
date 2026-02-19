"""
Branch 0 test: verify that outputs are NOT written to public repo by default.

Runs a tiny training with --output-dir pointing to a temp directory and
asserts no files leak into the public repo root.
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_utf8_env():
    """Get env dict with PYTHONUTF8=1 for Windows subprocess Unicode safety."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    return env


def test_output_dir_override_writes_to_temp(tmp_path):
    """Training with --output-dir <tmp> writes output there, not under public repo."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()

    # Run 1 epoch with minimal settings on CPU using synthetic data
    # We create a tiny dataset inline
    ds_dir = tmp_path / "dataset"
    img_dir = ds_dir / "images"
    mask_dir = ds_dir / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    from tests.helpers.synthetic_data import create_rgb_image, create_random_stroke_mask

    for i in range(3):
        create_rgb_image(64, 64).save(img_dir / f"image_{i}.png")
        create_random_stroke_mask(64, 64, seed=i).save(mask_dir / f"image_{i}.png")

    result = subprocess.run(
        [
            sys.executable, str(PROJECT_ROOT / "train_segmentation.py"),
            "--data-dir", str(ds_dir),
            "--output-dir", str(output_dir),
            "--epochs", "1",
            "--batch-size", "1",
            "--img-height", "32",
            "--img-width", "32",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(PROJECT_ROOT),
        env=_get_utf8_env(),
    )

    assert result.returncode == 0, f"Training failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # Files should exist in temp output dir
    assert (output_dir / "whiteboard_seg_best.pt").exists(), "Best model not saved to temp dir"
    assert (output_dir / "training_history.json").exists(), "History not saved to temp dir"

    # No files should have leaked into public repo common output dirs
    forbidden_dirs = [
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "Research-Paper",
        PROJECT_ROOT / "results",
        PROJECT_ROOT / "experiments",
    ]
    for d in forbidden_dirs:
        if d.exists():
            assert not any(d.rglob("whiteboard_seg_best.pt")), (
                f"Model file leaked into public repo at {d}"
            )


def test_output_dir_override_does_not_write_to_private_repo(tmp_path):
    """When --output-dir is set to a temp dir, nothing goes to the private repo."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()

    # The private repo path (may or may not exist on the test machine)
    private_repo = PROJECT_ROOT.parent / "SegmentationResearchPaper"

    # Record files before (if private repo exists)
    files_before = set()
    if private_repo.exists():
        files_before = set(private_repo.rglob("*.pt"))

    ds_dir = tmp_path / "dataset"
    img_dir = ds_dir / "images"
    mask_dir = ds_dir / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    from tests.helpers.synthetic_data import create_rgb_image, create_random_stroke_mask

    for i in range(3):
        create_rgb_image(64, 64).save(img_dir / f"image_{i}.png")
        create_random_stroke_mask(64, 64, seed=i).save(mask_dir / f"image_{i}.png")

    result = subprocess.run(
        [
            sys.executable, str(PROJECT_ROOT / "train_segmentation.py"),
            "--data-dir", str(ds_dir),
            "--output-dir", str(output_dir),
            "--epochs", "1",
            "--batch-size", "1",
            "--img-height", "32",
            "--img-width", "32",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(PROJECT_ROOT),
        env=_get_utf8_env(),
    )

    assert result.returncode == 0, f"Training failed:\n{result.stderr}"

    # No new .pt files in private repo
    if private_repo.exists():
        files_after = set(private_repo.rglob("*.pt"))
        new_files = files_after - files_before
        assert not new_files, f"Files leaked to private repo: {new_files}"

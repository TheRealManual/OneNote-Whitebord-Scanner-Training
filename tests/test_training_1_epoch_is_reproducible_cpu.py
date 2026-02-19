"""
Branch 1 test: verify that 1-epoch training is reproducible on CPU with same seed.

Runs training twice using the ToySegModel with identical settings and asserts:
  - Final validation loss matches within epsilon
  - training_history.json key fields match (seed, epoch count)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_utf8_env():
    """Get env dict with PYTHONUTF8=1 for Windows subprocess Unicode safety."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    return env


def _run_training(ds_dir, output_dir, seed):
    """Run 1-epoch training on CPU with ToySegModel."""
    result = subprocess.run(
        [
            sys.executable, str(PROJECT_ROOT / "train_segmentation.py"),
            "--data-dir", str(ds_dir),
            "--output-dir", str(output_dir),
            "--epochs", "1",
            "--batch-size", "1",
            "--img-height", "32",
            "--img-width", "32",
            "--seed", str(seed),
            "--model", "toy",
            "--loss", "dice",
            "--patience", "100",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(PROJECT_ROOT),
        env=_get_utf8_env(),
    )
    return result


def _create_mini_dataset(base_dir):
    """Create a tiny dataset with 4 images for train/val split."""
    from tests.helpers.synthetic_data import create_rgb_image, create_random_stroke_mask

    ds_dir = base_dir / "dataset"
    img_dir = ds_dir / "images"
    mask_dir = ds_dir / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    for i in range(4):
        create_rgb_image(32, 32).save(img_dir / f"image_{i}.png")
        create_random_stroke_mask(32, 32, seed=i).save(mask_dir / f"image_{i}.png")

    return ds_dir


def test_training_1_epoch_reproducible_cpu(tmp_path):
    """Two runs with same seed on CPU must produce identical val_loss."""
    ds_dir = _create_mini_dataset(tmp_path)

    out1 = tmp_path / "run1"
    out1.mkdir()
    out2 = tmp_path / "run2"
    out2.mkdir()

    r1 = _run_training(ds_dir, out1, seed=42)
    assert r1.returncode == 0, f"Run 1 failed:\n{r1.stderr}"

    r2 = _run_training(ds_dir, out2, seed=42)
    assert r2.returncode == 0, f"Run 2 failed:\n{r2.stderr}"

    # Load histories
    h1 = json.loads((out1 / "training_history.json").read_text())
    h2 = json.loads((out2 / "training_history.json").read_text())

    # Seed and epoch count must match exactly
    assert h1["config"]["seed"] == h2["config"]["seed"] == 42
    assert h1["config"]["epochs"] == h2["config"]["epochs"] == 1

    # Val loss should be identical on CPU deterministic (within float epsilon)
    vl1 = h1["val_loss"][-1]
    vl2 = h2["val_loss"][-1]
    assert abs(vl1 - vl2) < 1e-6, (
        f"Val loss not reproducible: {vl1} vs {vl2} (diff={abs(vl1 - vl2)})"
    )

    # Train loss should also match
    tl1 = h1["train_loss"][-1]
    tl2 = h2["train_loss"][-1]
    assert abs(tl1 - tl2) < 1e-6, (
        f"Train loss not reproducible: {tl1} vs {tl2} (diff={abs(tl1 - tl2)})"
    )


def test_different_seeds_produce_different_results(tmp_path):
    """Two runs with different seeds should NOT produce identical val_loss."""
    ds_dir = _create_mini_dataset(tmp_path)

    out1 = tmp_path / "run_seed42"
    out1.mkdir()
    out2 = tmp_path / "run_seed123"
    out2.mkdir()

    r1 = _run_training(ds_dir, out1, seed=42)
    assert r1.returncode == 0, f"Run 1 failed:\n{r1.stderr}"

    r2 = _run_training(ds_dir, out2, seed=123)
    assert r2.returncode == 0, f"Run 2 failed:\n{r2.stderr}"

    h1 = json.loads((out1 / "training_history.json").read_text())
    h2 = json.loads((out2 / "training_history.json").read_text())

    vl1 = h1["val_loss"][-1]
    vl2 = h2["val_loss"][-1]

    # Very unlikely to be equal with different seeds
    assert vl1 != vl2, (
        f"Different seeds produced identical val_loss: {vl1}"
    )

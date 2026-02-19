"""
Branch 2 test: verify build_loss() factory returns a callable for all modes.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_segmentation import build_loss, LOSS_REGISTRY


def _make_args(**overrides):
    """Create a SimpleNamespace matching the argparse defaults."""
    defaults = dict(
        loss='dice_focal',
        dice_weight=0.6,
        focal_weight=0.4,
        focal_alpha=0.25,
        focal_gamma=2.0,
        tversky_alpha=0.3,
        tversky_beta=0.7,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_all_loss_modes_return_callable():
    """build_loss() must return (callable, str) for every registered mode."""
    for mode in LOSS_REGISTRY:
        args = _make_args(loss=mode)
        criterion, name = build_loss(args)
        assert callable(criterion), f"Loss for mode '{mode}' is not callable"
        assert isinstance(name, str) and len(name) > 0, (
            f"Loss name for mode '{mode}' is not a non-empty string"
        )


def test_unknown_loss_raises_valueerror():
    """build_loss() must raise ValueError for unknown loss types."""
    args = _make_args(loss='nonexistent')
    try:
        build_loss(args)
        assert False, "Expected ValueError for unknown loss type"
    except ValueError as e:
        assert 'nonexistent' in str(e)


def test_all_registered_modes_exist():
    """LOSS_REGISTRY must contain exactly: ce, dice, focal, dice_focal, tversky."""
    expected = {'ce', 'dice', 'focal', 'dice_focal', 'tversky'}
    assert set(LOSS_REGISTRY.keys()) == expected, (
        f"LOSS_REGISTRY keys mismatch: {set(LOSS_REGISTRY.keys())} != {expected}"
    )

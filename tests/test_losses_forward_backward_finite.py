"""
Branch 2 test: verify all loss functions produce finite scalar and valid gradients.

For each loss mode:
  - Creates synthetic logits [B,2,H,W] and binary targets [B,H,W]
  - Computes loss
  - Asserts: scalar, finite, backward produces finite gradients
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_segmentation import build_loss, LOSS_REGISTRY


def _make_args(**overrides):
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


def _run_forward_backward(mode):
    """Run forward + backward for a given loss mode. Returns (loss_val, grads_ok)."""
    args = _make_args(loss=mode)
    criterion, name = build_loss(args)

    B, C, H, W = 2, 2, 32, 32
    logits = torch.randn(B, C, H, W, requires_grad=True)
    targets = torch.randint(0, 2, (B, H, W))

    loss = criterion(logits, targets)

    # Check scalar
    assert loss.ndim == 0, f"[{mode}] Loss is not a scalar: shape={loss.shape}"

    # Check finite
    assert torch.isfinite(loss), f"[{mode}] Loss is not finite: {loss.item()}"

    # Backward
    loss.backward()

    # Check gradients exist and are finite
    assert logits.grad is not None, f"[{mode}] No gradient on logits after backward"
    assert torch.all(torch.isfinite(logits.grad)), (
        f"[{mode}] Gradients contain non-finite values"
    )

    return loss.item()


def test_ce_forward_backward():
    _run_forward_backward('ce')


def test_dice_forward_backward():
    _run_forward_backward('dice')


def test_focal_forward_backward():
    _run_forward_backward('focal')


def test_dice_focal_forward_backward():
    _run_forward_backward('dice_focal')


def test_tversky_forward_backward():
    _run_forward_backward('tversky')


def test_all_losses_produce_positive_loss():
    """Every loss on random data should produce a positive value."""
    for mode in LOSS_REGISTRY:
        args = _make_args(loss=mode)
        criterion, _ = build_loss(args)

        logits = torch.randn(2, 2, 32, 32)
        targets = torch.randint(0, 2, (2, 32, 32))

        loss = criterion(logits, targets)
        assert loss.item() > 0, f"[{mode}] Loss should be positive on random data, got {loss.item()}"

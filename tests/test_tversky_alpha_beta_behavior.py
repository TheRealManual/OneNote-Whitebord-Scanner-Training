"""
Branch 2 test: verify Tversky alpha/beta FP/FN penalty behavior.

Constructs a case with known FP/FN imbalance where the prediction misses
many stroke pixels (high FN). Compares loss with:
  - alpha=0.3, beta=0.7 (favor recall — penalizes FN more)
  - alpha=0.7, beta=0.3 (favor precision — penalizes FP more)

When FN dominates, the recall-favoring config (beta=0.7) should produce
higher loss because it penalizes the FN-heavy scenario more.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_segmentation import TverskyLoss


def test_tversky_favors_recall_penalizes_fn():
    """When FN dominates, alpha=0.3/beta=0.7 should give HIGHER loss than alpha=0.7/beta=0.3."""
    B, C, H, W = 1, 2, 16, 16

    # Target: significant foreground (bottom half is stroke)
    target = torch.zeros(B, H, W, dtype=torch.long)
    target[:, H // 2:, :] = 1  # bottom half is stroke

    # Prediction: mostly background (misses strokes = high FN)
    # Create logits strongly favoring background
    logits = torch.zeros(B, C, H, W)
    logits[:, 0, :, :] = 3.0  # strong background prediction everywhere
    logits[:, 1, :, :] = -3.0  # weak stroke prediction

    # Loss with favor-recall (beta > alpha → penalizes FN more)
    loss_recall = TverskyLoss(alpha=0.3, beta=0.7)
    val_recall = loss_recall(logits, target).item()

    # Loss with favor-precision (alpha > beta → penalizes FP more)
    loss_precision = TverskyLoss(alpha=0.7, beta=0.3)
    val_precision = loss_precision(logits, target).item()

    # When FN dominates, the recall-favoring config should give higher loss
    assert val_recall > val_precision, (
        f"Expected recall-favoring loss ({val_recall:.6f}) > precision-favoring loss "
        f"({val_precision:.6f}) when FN dominates"
    )


def test_tversky_favors_precision_penalizes_fp():
    """When FP dominates, alpha=0.7/beta=0.3 should give HIGHER loss than alpha=0.3/beta=0.7."""
    B, C, H, W = 1, 2, 16, 16

    # Target: mostly background (no strokes)
    target = torch.zeros(B, H, W, dtype=torch.long)
    target[:, 0:2, 0:2] = 1  # tiny stroke area

    # Prediction: predicts stroke everywhere (high FP)
    logits = torch.zeros(B, C, H, W)
    logits[:, 0, :, :] = -3.0  # weak background
    logits[:, 1, :, :] = 3.0   # strong stroke prediction everywhere

    loss_recall = TverskyLoss(alpha=0.3, beta=0.7)
    val_recall = loss_recall(logits, target).item()

    loss_precision = TverskyLoss(alpha=0.7, beta=0.3)
    val_precision = loss_precision(logits, target).item()

    # When FP dominates, the precision-favoring config should give higher loss
    assert val_precision > val_recall, (
        f"Expected precision-favoring loss ({val_precision:.6f}) > recall-favoring loss "
        f"({val_recall:.6f}) when FP dominates"
    )


def test_tversky_equals_dice_when_alpha_beta_equal():
    """Tversky with alpha=beta=0.5 should approximate Dice loss."""
    from train_segmentation import DiceLoss

    B, C, H, W = 2, 2, 16, 16
    logits = torch.randn(B, C, H, W)
    targets = torch.randint(0, 2, (B, H, W))

    tversky = TverskyLoss(alpha=0.5, beta=0.5)
    dice = DiceLoss()

    val_tversky = tversky(logits, targets).item()
    val_dice = dice(logits, targets).item()

    # They should be close (not exact due to smooth parameter differences)
    assert abs(val_tversky - val_dice) < 0.05, (
        f"Tversky(0.5, 0.5) = {val_tversky:.6f} should approximate Dice = {val_dice:.6f}"
    )

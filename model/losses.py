"""
Loss functions for medical image segmentation.

DiceBCELoss: combines Dice loss (handles class imbalance) with
BCE loss (stabilises gradients) — standard in medical imaging.
"""

import torch
import torch.nn as nn


def dice_score(pred, target, smooth=1e-6):
    """
    Compute Dice coefficient between prediction and ground truth.

    Args:
        pred:   binary tensor (B, 1, H, W) — already sigmoid + thresholded
        target: binary tensor (B, 1, H, W) — ground truth mask
        smooth: small constant to avoid division by zero

    Returns:
        scalar Dice score in [0, 1]
    """
    pred   = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice


def iou_score(pred, target, smooth=1e-6):
    """
    Intersection over Union (Jaccard index).
    """
    pred   = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


class DiceLoss(nn.Module):
    """Pure Dice loss — good for heavily imbalanced segmentation tasks."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        pred  = probs.contiguous().view(-1)
        tgt   = targets.contiguous().view(-1)
        inter = (pred * tgt).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (pred.sum() + tgt.sum() + self.smooth)


class DiceBCELoss(nn.Module):
    """
    Combined Dice + Binary Cross Entropy loss.

    - BCE provides stable gradients early in training
    - Dice handles class imbalance (tiny tumour in large background)
    - Combined loss converges faster and more reliably than either alone
    """

    def __init__(self, smooth=1e-6, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.smooth      = smooth
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce         = nn.BCEWithLogitsLoss()  # numerically stable sigmoid+BCE

    def forward(self, logits, targets):
        # BCE component
        bce_loss = self.bce(logits, targets)

        # Dice component
        probs = torch.sigmoid(logits)
        pred  = probs.contiguous().view(-1)
        tgt   = targets.contiguous().view(-1)
        inter = (pred * tgt).sum()
        dice_loss = 1.0 - (2.0 * inter + self.smooth) / (pred.sum() + tgt.sum() + self.smooth)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


if __name__ == "__main__":
    # Quick test
    logits  = torch.randn(4, 1, 256, 256)
    targets = torch.randint(0, 2, (4, 1, 256, 256)).float()

    criterion = DiceBCELoss()
    loss      = criterion(logits, targets)
    print(f"DiceBCE loss: {loss.item():.4f}  (expect ~1.2–1.8 at random init)")

    preds = (torch.sigmoid(logits) > 0.5).float()
    print(f"Dice score:   {dice_score(preds, targets).item():.4f}")
    print(f"IoU score:    {iou_score(preds, targets).item():.4f}")

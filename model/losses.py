import torch
import torch.nn as nn


def dice_score(pred, target, smooth=1e-6):

    pred   = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice


def iou_score(pred, target, smooth=1e-6):

    pred   = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


class DiceLoss(nn.Module):


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


    def __init__(self, smooth=1e-6, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.smooth      = smooth
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce         = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):

        bce_loss = self.bce(logits, targets)


        probs = torch.sigmoid(logits)
        pred  = probs.contiguous().view(-1)
        tgt   = targets.contiguous().view(-1)
        inter = (pred * tgt).sum()
        dice_loss = 1.0 - (2.0 * inter + self.smooth) / (pred.sum() + tgt.sum() + self.smooth)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


if __name__ == "__main__":

    logits  = torch.randn(4, 1, 256, 256)
    targets = torch.randint(0, 2, (4, 1, 256, 256)).float()

    criterion = DiceBCELoss()
    loss      = criterion(logits, targets)
    print(f"DiceBCE loss: {loss.item():.4f}  (expect ~1.2–1.8 at random init)")

    preds = (torch.sigmoid(logits) > 0.5).float()
    print(f"Dice score:   {dice_score(preds, targets).item():.4f}")
    print(f"IoU score:    {iou_score(preds, targets).item():.4f}")

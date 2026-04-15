"""
Main training script for U-Net liver segmentation.

Usage:
    python train.py                          # default settings
    python train.py --epochs 50 --lr 1e-4   # custom settings
    python train.py --data_dir slices --batch_size 8  # low VRAM

Outputs:
    checkpoints/unet_best.pt    ← best model by validation Dice
    checkpoints/unet_last.pt    ← final epoch checkpoint
    training_history.npy        ← loss/dice curves for plotting
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from model import UNet, DiceBCELoss, dice_score
from data  import get_dataloaders


# ── Training one epoch ─────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    loop = tqdm(loader, desc='  Train', leave=False)
    for imgs, masks in loop:
        imgs  = imgs.to(device,  non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, masks)
        loss.backward()

        # Gradient clipping — prevents exploding gradients in early training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=f'{loss.item():.4f}')

        # Free memory explicitly (important on low-VRAM GPUs)
        del imgs, masks, logits, loss

    return total_loss / len(loader)


# ── Validation one epoch ───────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    dice_scores = []

    for imgs, masks in tqdm(loader, desc='  Val  ', leave=False):
        imgs  = imgs.to(device,  non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)
        preds  = (torch.sigmoid(logits) > 0.5).float()
        dice_scores.append(dice_score(preds, masks).item())

    return float(np.mean(dice_scores))


# ── Main training loop ─────────────────────────────────────────────────────────

def train(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # Data
    print(f"\nLoading dataset from: {args.data_dir}")
    train_dl, val_dl = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    model = UNet(in_ch=1, out_ch=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model:  U-Net  ({total_params:,} parameters)")

    # Loss, optimiser, scheduler
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    os.makedirs('checkpoints', exist_ok=True)

    history   = {'train_loss': [], 'val_dice': []}
    best_dice = 0.0

    print(f"\nTraining for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_dl, criterion, optimizer, device)
        val_dice   = val_epoch(model, val_dl, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_dice'].append(val_dice)

        # Save best checkpoint
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_dice':    val_dice,
                'train_loss':  train_loss,
                'args':        vars(args),
            }, 'checkpoints/unet_best.pt')
            saved = ' ← saved best'
        else:
            saved = ''

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"loss {train_loss:.4f}  "
              f"dice {val_dice:.4f}  "
              f"best {best_dice:.4f}  "
              f"lr {lr_now:.2e}"
              f"{saved}")

    # Save last checkpoint + history
    torch.save({
        'epoch':       args.epochs,
        'model_state': model.state_dict(),
        'val_dice':    val_dice,
    }, 'checkpoints/unet_last.pt')
    np.save('training_history.npy', history)

    print(f"\nTraining complete!")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Checkpoints: checkpoints/unet_best.pt")
    print(f"Next: python evaluate.py")
    return history


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train U-Net for liver segmentation')
    parser.add_argument('--data_dir',    default='slices',  help='Path to extracted slices directory')
    parser.add_argument('--epochs',      default=30,  type=int,   help='Number of training epochs')
    parser.add_argument('--batch_size',  default=16,  type=int,   help='Batch size (reduce to 8 or 4 if OOM)')
    parser.add_argument('--lr',          default=1e-4, type=float, help='Initial learning rate')
    parser.add_argument('--num_workers', default=4,   type=int,   help='DataLoader workers (set 0 on Windows if errors)')
    args = parser.parse_args()

    # Print config
    print("=" * 50)
    print("U-Net Liver Segmentation — Training")
    print("=" * 50)
    for k, v in vars(args).items():
        print(f"  {k:<15}: {v}")

    history = train(args)

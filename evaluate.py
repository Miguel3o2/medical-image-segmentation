"""
Evaluation and visualisation for the trained U-Net.

Usage:
    python evaluate.py                                      # uses checkpoints/unet_best.pt
    python evaluate.py --checkpoint checkpoints/unet_best.pt
    python evaluate.py --visualise_only                     # skip metrics, just plot

Outputs:
    assets/predictions.png       ← CT + ground truth + prediction overlays
    assets/training_curves.png   ← loss and Dice curves over epochs
    assets/metrics_report.txt    ← full metrics summary
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from model import UNet, dice_score, iou_score
from data  import get_dataloaders


# ── Metrics ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_dl, device):
    """Compute Dice, IoU, Precision, Recall on the validation set."""
    model.eval()
    metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': []}

    for imgs, masks in tqdm(val_dl, desc='Evaluating'):
        logits = model(imgs.to(device))
        preds  = (torch.sigmoid(logits) > 0.5).float().cpu()

        p = preds.contiguous().view(-1).numpy()
        t = masks.contiguous().view(-1).numpy()

        inter = (p * t).sum()
        union = p.sum() + t.sum() - inter

        metrics['dice'].append(      (2 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6))
        metrics['iou'].append(       (inter + 1e-6)     / (union + 1e-6))
        metrics['precision'].append( (inter + 1e-6)     / (p.sum() + 1e-6))
        metrics['recall'].append(    (inter + 1e-6)     / (t.sum() + 1e-6))

    results = {k: float(np.mean(v)) for k, v in metrics.items()}

    print("\n" + "=" * 40)
    print("Evaluation Results")
    print("=" * 40)
    print(f"  Dice score : {results['dice']:.4f}")
    print(f"  IoU score  : {results['iou']:.4f}")
    print(f"  Precision  : {results['precision']:.4f}")
    print(f"  Recall     : {results['recall']:.4f}")
    print("=" * 40)

    return results


# ── Prediction visualisation ───────────────────────────────────────────────────

@torch.no_grad()
def visualise_predictions(model, val_dl, device, n=4, save_path='assets/predictions.png'):
    """Plot n CT slices with ground truth and predicted overlays side-by-side."""
    model.eval()
    imgs, masks = next(iter(val_dl))

    logits = model(imgs.to(device))
    preds  = (torch.sigmoid(logits) > 0.5).float().cpu()

    n = min(n, imgs.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(13, n * 4.2))

# Ensure axes is always 2D
    if n == 1:
      axes = np.expand_dims(axes, axis=0)
    fig.suptitle('U-Net Liver Segmentation — Validation Predictions', fontsize=14, y=1.01)

    for i in range(n):
        img_np  = imgs[i, 0].numpy()
        gt_np   = masks[i, 0].numpy()
        pred_np = preds[i, 0].numpy()

        # Per-slice Dice
        inter   = (pred_np * gt_np).sum()
        d       = (2 * inter + 1e-6) / (pred_np.sum() + gt_np.sum() + 1e-6)

        # CT slice
        axes[i, 0].imshow(img_np, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'CT slice {i+1}', fontsize=11)

        # Ground truth
        axes[i, 1].imshow(img_np, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].imshow(gt_np,  cmap='Greens', alpha=0.5, vmin=0, vmax=1)
        axes[i, 1].set_title('Ground truth', fontsize=11)

        # Prediction overlay
        axes[i, 2].imshow(img_np,   cmap='gray', vmin=0, vmax=1)
        axes[i, 2].imshow(pred_np,  cmap='Reds',  alpha=0.5, vmin=0, vmax=1)
        axes[i, 2].set_title(f'Prediction  (Dice = {d:.3f})', fontsize=11)

    for ax in axes.flat:
        ax.axis('off')

    # Legend
    gt_patch   = mpatches.Patch(color='green', alpha=0.5, label='Ground truth')
    pred_patch = mpatches.Patch(color='red',   alpha=0.5, label='Prediction')
    fig.legend(handles=[gt_patch, pred_patch], loc='lower center',
               ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ── Training curves ────────────────────────────────────────────────────────────

def plot_training_curves(history_path='training_history.npy',
                         save_path='assets/training_curves.png'):
    """Plot training loss and validation Dice over epochs."""
    if not os.path.exists(history_path):
        print(f"History file not found: {history_path}")
        return

    history = np.load(history_path, allow_pickle=True).item()
    epochs  = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('U-Net Training History', fontsize=14)

    # Loss curve
    ax1.plot(epochs, history['train_loss'], color='#185FA5', linewidth=2, label='Train loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Training loss (DiceBCE)')
    ax1.grid(alpha=0.3); ax1.legend()

    # Dice curve
    ax2.plot(epochs, history['val_dice'], color='#1D9E75', linewidth=2, label='Val Dice')
    ax2.axhline(0.85, color='gray', linestyle='--', alpha=0.6, label='Publication threshold (0.85)')
    ax2.axhline(0.90, color='#854F0B', linestyle=':', alpha=0.6, label='Excellent threshold (0.90)')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dice score')
    ax2.set_title('Validation Dice score')
    ax2.set_ylim(0, 1); ax2.grid(alpha=0.3); ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ── Sanity check ───────────────────────────────────────────────────────────────

@torch.no_grad()
def sanity_check(val_dl):
    """Visualise raw data — always run this before evaluating a new model."""
    imgs, masks = next(iter(val_dl))
    print(f"\nSanity check:")
    print(f"  Image shape:  {imgs.shape}")
    print(f"  Mask shape:   {masks.shape}")
    print(f"  Image range:  [{imgs.min():.3f}, {imgs.max():.3f}]")
    print(f"  Mask values:  {masks.unique().tolist()}")

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.suptitle('Data sanity check — images (top) and masks (bottom)', fontsize=12)
    for i in range(min(4, imgs.shape[0])):
        axes[0, i].imshow(imgs[i, 0].numpy(), cmap='gray')
        axes[0, i].set_title(f'CT slice {i+1}')
        axes[1, i].imshow(masks[i, 0].numpy(), cmap='hot')
        axes[1, i].set_title(f'Liver mask {i+1}')
    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/sanity_check.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("  Saved: assets/sanity_check.png")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained U-Net')
    parser.add_argument('--checkpoint',    default='checkpoints/unet_best.pt')
    parser.add_argument('--data_dir',      default='slices')
    parser.add_argument('--batch_size',    default=16,  type=int)
    parser.add_argument('--num_workers',   default=4,   type=int)
    parser.add_argument('--n_vis',         default=4,   type=int, help='Slices to visualise')
    parser.add_argument('--sanity_check',  action='store_true',   help='Only run data sanity check')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataloaders (no augmentation at eval time)
    _, val_dl = get_dataloaders(args.data_dir, batch_size=args.batch_size,
                                num_workers=args.num_workers)

    if args.sanity_check:
        sanity_check(val_dl)
    else:
        # Load model
        print(f"\nLoading checkpoint: {args.checkpoint}")
        model  = UNet(in_ch=1, out_ch=1).to(device)
        ckpt   = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f"Checkpoint from epoch {ckpt.get('epoch', '?')} | "
              f"val Dice {ckpt.get('val_dice', 0):.4f}")

        # Metrics
        results = evaluate(model, val_dl, device)

        # Save metrics report
        os.makedirs('assets', exist_ok=True)
        with open('assets/metrics_report.txt', 'w') as f:
            f.write("U-Net Liver Segmentation — Evaluation Results\n")
            f.write("=" * 45 + "\n")
            for k, v in results.items():
                f.write(f"{k:<12}: {v:.4f}\n")
        print("Saved: assets/metrics_report.txt")

        # Visualise predictions
        visualise_predictions(model, val_dl, device, n=args.n_vis)

        # Training curves
        plot_training_curves()

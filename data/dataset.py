"""
PyTorch Dataset for the pre-extracted liver CT slice dataset.

Expects the output of data/preprocess.py:
    slices/
        imgs/   00000.npy  00001.npy  ...
        masks/  00000.npy  00001.npy  ...

Usage:
    from data.dataset import LiverSliceDataset, get_dataloaders
    train_dl, val_dl = get_dataloaders('slices', batch_size=16)
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class LiverSliceDataset(Dataset):
    """
    Dataset for 2D liver CT slices extracted from NIfTI volumes.

    Each item is a (image, mask) pair:
        image: (1, H, W) float32 tensor, values in [0, 1]
        mask:  (1, H, W) float32 tensor, values in {0, 1}
    """

    def __init__(self, root_dir, augment=False):
        """
        Args:
            root_dir: directory containing imgs/ and masks/ subdirectories
            augment:  if True, apply random augmentations
        """
        self.imgs   = sorted(glob.glob(os.path.join(root_dir, 'imgs',  '*.npy')))
        self.masks  = sorted(glob.glob(os.path.join(root_dir, 'masks', '*.npy')))
        self.augment = augment

        assert len(self.imgs) == len(self.masks), \
            f"Mismatch: {len(self.imgs)} images vs {len(self.masks)} masks"
        assert len(self.imgs) > 0, \
            f"No .npy files found in {root_dir}/imgs/. Run data/preprocess.py first."

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img  = np.load(self.imgs[idx])   # (H, W) float32
        mask = np.load(self.masks[idx])  # (H, W) float32 binary

        if self.augment:
            img, mask = self._augment(img, mask)

        # Add channel dimension: (H, W) → (1, H, W)
        img  = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask

    def _augment(self, img, mask):
        """
        Conservative augmentations valid for medical imaging.
        - Horizontal flip only (no vertical — anatomical orientation matters)
        - Subtle gamma shift for intensity variation
        - Small Gaussian noise
        """
        # Horizontal flip
        if np.random.rand() > 0.5:
            img  = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        # Gamma correction (simulates scanner variability)
        if np.random.rand() > 0.5:
            gamma = np.random.uniform(0.8, 1.2)
            img   = np.power(np.clip(img, 0, 1), gamma)

        # Gaussian noise (very subtle)
        if np.random.rand() > 0.7:
            noise = np.random.normal(0, 0.01, img.shape).astype(np.float32)
            img   = np.clip(img + noise, 0, 1)

        return img.astype(np.float32), mask.astype(np.float32)


def get_dataloaders(root_dir, batch_size=16, val_fraction=0.15,
                    num_workers=4, seed=42):
    """
    Build train and validation DataLoaders from the slice dataset.

    Args:
        root_dir:     path to slices/ directory
        batch_size:   samples per batch
        val_fraction: fraction of data to use for validation
        num_workers:  parallel data loading workers (set 0 on Windows if issues)
        seed:         random seed for reproducible split

    Returns:
        (train_dl, val_dl): tuple of DataLoaders
    """
    full_ds = LiverSliceDataset(root_dir, augment=False)  # no augment yet
    n_val   = max(1, int(len(full_ds) * val_fraction))
    n_train = len(full_ds) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)

    # Enable augmentation on training split
    train_ds.dataset.augment = False  # will set per-sample in wrapper below

    # Wrap with augmentation-aware dataset
    train_aug_ds = _AugmentedSubset(train_ds, augment=True)
    val_aug_ds   = _AugmentedSubset(val_ds,   augment=False)

    train_dl = DataLoader(
        train_aug_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_dl = DataLoader(
        val_aug_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"Train: {len(train_aug_ds)} slices | Val: {len(val_aug_ds)} slices")
    print(f"Batch size: {batch_size} | Train batches: {len(train_dl)} | Val batches: {len(val_dl)}")
    return train_dl, val_dl


class _AugmentedSubset(Dataset):
    """Wraps a Subset and applies augmentation at the slice level."""

    def __init__(self, subset, augment=False):
        self.subset  = subset
        self.augment = augment

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, mask = self.subset[idx]
        if self.augment:
            img_np  = img.squeeze(0).numpy()
            mask_np = mask.squeeze(0).numpy()
            img_np, mask_np = LiverSliceDataset._augment(
                LiverSliceDataset.__new__(LiverSliceDataset), img_np, mask_np
            )
            img  = torch.from_numpy(img_np).unsqueeze(0)
            mask = torch.from_numpy(mask_np).unsqueeze(0)
        return img, mask


if __name__ == '__main__':
    # Quick test — run from project root: python data/dataset.py
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else 'slices'
    train_dl, val_dl = get_dataloaders(root, batch_size=4, num_workers=0)
    imgs, masks = next(iter(train_dl))
    print(f"Image batch:  {imgs.shape}   range [{imgs.min():.2f}, {imgs.max():.2f}]")
    print(f"Mask batch:   {masks.shape}  unique values: {masks.unique().tolist()}")
    print("Dataset OK!")

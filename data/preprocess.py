"""
Preprocessing pipeline for Medical Segmentation Decathlon Task03 (Liver CT).

Run this ONCE before training to extract 2D slices from 3D volumes:
    python data/preprocess.py --data_dir Task03_Liver --out_dir slices

Input:  Task03_Liver/imagesTr/*.nii.gz  (3D CT volumes)
        Task03_Liver/labelsTr/*.nii.gz  (3D segmentation masks)
Output: slices/imgs/00000.npy  ...      (2D axial slices, float32)
        slices/masks/00000.npy ...      (2D binary masks, float32)
"""

import os
import glob
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm


# ── Windowing ──────────────────────────────────────────────────────────────────

def apply_window(volume, window_center=-75, window_width=400):
    """
    Clamp CT HU values to a clinically relevant window and normalise to [0, 1].

    Default: soft tissue window for liver (WC=-75, WW=400 → range -275 to +125 HU)
    """
    lo = window_center - window_width / 2
    hi = window_center + window_width / 2
    volume = np.clip(volume, lo, hi)
    volume = (volume - lo) / (hi - lo)
    return volume.astype(np.float32)


# ── Resampling ─────────────────────────────────────────────────────────────────

def resize_slice(sl, mask, target_size=(256, 256)):
    """
    Resize a 2D CT slice and its mask to target_size.

    - Uses bilinear (order=1) for images  — smooth interpolation
    - Uses nearest  (order=0) for masks   — preserves binary labels
    """
    zoom_f = (target_size[0] / sl.shape[0], target_size[1] / sl.shape[1])
    sl_r   = zoom(sl,   zoom_f, order=1).astype(np.float32)
    mask_r = zoom(mask, zoom_f, order=0).astype(np.float32)
    return sl_r, mask_r


# ── Volume preprocessing ───────────────────────────────────────────────────────

def preprocess_volume(img_path, mask_path, target_size=(256, 256),
                      min_liver_pixels=500):
    """
    Load one NIfTI volume + mask, apply windowing, extract 2D axial slices.

    Skips slices with fewer than min_liver_pixels foreground voxels to
    avoid training on mostly-empty slices.

    Returns:
        List of (image_slice, mask_slice) tuples, both float32 numpy arrays.
    """
    img_nii  = nib.load(img_path)
    mask_nii = nib.load(mask_path)

    img  = img_nii.get_fdata().astype(np.float32)   # (H, W, D)
    mask = mask_nii.get_fdata().astype(np.float32)  # (H, W, D)

    # Binarise mask: 0=background, 1=liver+tumour
    mask = (mask > 0).astype(np.float32)

    # Apply soft tissue window
    # REPLACE this line in preprocess_volume():
    img = apply_window(img)

    img = (img - img.mean()) / (img.std() + 1e-6)   # z-score normalise
    img = np.clip(img, -3, 3)                         # clip outliers
    img = (img + 3) / 6.0                            # rescale to [0, 1]

    slices = []
    for i in range(img.shape[2]):
        sl   = img[:, :, i]
        msk  = mask[:, :, i]

        # Skip slices with too little liver
        if msk.sum() < min_liver_pixels:
            continue

        sl_r, msk_r = resize_slice(sl, msk, target_size)
        slices.append((sl_r, msk_r))

    return slices


# ── Main extraction loop ───────────────────────────────────────────────────────

def extract_all_slices(data_dir, out_dir, target_size=(256, 256)):
    img_paths  = sorted(glob.glob(os.path.join(data_dir, 'imagesTr', '*.nii.gz')))
    mask_paths = sorted(glob.glob(os.path.join(data_dir, 'labelsTr', '*.nii.gz')))

    assert len(img_paths) == len(mask_paths), \
        f"Mismatch: {len(img_paths)} images vs {len(mask_paths)} masks"

    os.makedirs(os.path.join(out_dir, 'imgs'),  exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'masks'), exist_ok=True)

    total_slices = 0
    for ip, mp in tqdm(zip(img_paths, mask_paths), total=len(img_paths),
                       desc='Extracting slices'):
        slices = preprocess_volume(ip, mp, target_size=target_size)
        for sl_img, sl_mask in slices:
            idx = f'{total_slices:05d}'
            np.save(os.path.join(out_dir, 'imgs',  f'{idx}.npy'), sl_img)
            np.save(os.path.join(out_dir, 'masks', f'{idx}.npy'), sl_mask)
            total_slices += 1

    print(f"\nDone! Extracted {total_slices} slices → {out_dir}/")
    print(f"Each slice: {target_size[0]}x{target_size[1]} float32")
    return total_slices


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract 2D slices from NIfTI volumes')
    parser.add_argument('--data_dir',   default='Task03_Liver', help='Path to MSD dataset folder')
    parser.add_argument('--out_dir',    default='slices',       help='Output directory for .npy files')
    parser.add_argument('--size',       default=256, type=int,  help='Target slice size (square)')
    parser.add_argument('--min_pixels', default=500, type=int,  help='Minimum liver pixels to keep slice')
    args = parser.parse_args()

    n = extract_all_slices(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        target_size=(args.size, args.size),
    )
    print(f"\nReady to train on {n} slices.")
    print(f"Next step: python train.py")

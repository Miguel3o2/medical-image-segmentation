"""
Preprocessing pipeline for Medical Segmentation Decathlon datasets.

Run this ONCE before training to extract 2D slices from 3D volumes:
    python data/preprocess.py --data_dir Task03_Liver --out_dir slices
    python data/preprocess.py --data_dir Task04_Hippocampus --out_dir slices_task04

Input:  <dataset>/imagesTr/*.nii.gz  (3D volumes)
        <dataset>/labelsTr/*.nii.gz  (3D segmentation masks)
Output: <out_dir>/imgs/00000.npy  ...   (2D axial slices, float32)
        <out_dir>/masks/00000.npy ...   (2D binary masks, float32)
"""

import os
import glob
import argparse
import json
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


def normalise_mri(volume):
    """
    Robust MRI normalisation using percentile clipping and z-score scaling.
    """
    volume = volume.astype(np.float32)
    nonzero = volume[np.abs(volume) > 1e-6]
    if nonzero.size == 0:
        nonzero = volume.reshape(-1)

    p1, p99 = np.percentile(nonzero, [1, 99])
    volume = np.clip(volume, p1, p99)
    mean = float(volume.mean())
    std = float(volume.std())
    volume = (volume - mean) / (std + 1e-6)
    volume = np.clip(volume, -3, 3)
    volume = (volume + 3) / 6.0
    return volume.astype(np.float32)


def load_dataset_profile(data_dir):
    """
    Infer lightweight preprocessing defaults from dataset.json when available.
    """
    profile = {
        'name': os.path.basename(os.path.abspath(data_dir)),
        'modality': None,
        'default_min_pixels': 500,
        'default_context_slices': 0,
    }

    dataset_json = os.path.join(data_dir, 'dataset.json')
    if os.path.exists(dataset_json):
        with open(dataset_json, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        profile['name'] = meta.get('name', profile['name'])
        modality = meta.get('modality', {})
        if isinstance(modality, dict):
            profile['modality'] = next(iter(modality.values()), None)
        elif isinstance(modality, str):
            profile['modality'] = modality

    name = (profile['name'] or '').lower()
    modality = (profile['modality'] or '').lower()

    if 'hippocampus' in name:
        profile['default_min_pixels'] = 20
        profile['default_context_slices'] = 2
    elif 'mri' in modality:
        profile['default_min_pixels'] = 50
        profile['default_context_slices'] = 1

    return profile


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
                      min_pixels=500, modality=None, context_slices=0):
    """
    Load one NIfTI volume + mask, apply windowing, extract 2D axial slices.

    Skips slices with fewer than min_pixels foreground voxels to
    avoid training on mostly-empty slices.

    Returns:
        List of (image_slice, mask_slice) tuples, both float32 numpy arrays.
    """
    img_nii  = nib.load(img_path)
    mask_nii = nib.load(mask_path)

    img  = img_nii.get_fdata().astype(np.float32)   # (H, W, D)
    mask = mask_nii.get_fdata().astype(np.float32)  # (H, W, D)

    # Binarise mask for binary segmentation.
    mask = (mask > 0).astype(np.float32)

    modality = (modality or '').lower()
    if modality == 'ct':
        img = apply_window(img)
        img = (img - img.mean()) / (img.std() + 1e-6)
        img = np.clip(img, -3, 3)
        img = (img + 3) / 6.0
    else:
        img = normalise_mri(img)

    slices = []
    keep_indices = set()
    positive_indices = []
    for i in range(img.shape[2]):
        if mask[:, :, i].sum() >= min_pixels:
            positive_indices.append(i)

    for idx in positive_indices:
        lo = max(0, idx - context_slices)
        hi = min(img.shape[2], idx + context_slices + 1)
        keep_indices.update(range(lo, hi))

    for i in sorted(keep_indices):
        sl = img[:, :, i]
        msk = mask[:, :, i]
        sl_r, msk_r = resize_slice(sl, msk, target_size)
        slices.append((sl_r, msk_r))

    return slices


# ── Main extraction loop ───────────────────────────────────────────────────────

def extract_all_slices(data_dir, out_dir, target_size=(256, 256),
                       min_pixels=None, context_slices=None):
    img_paths  = sorted(glob.glob(os.path.join(data_dir, 'imagesTr', '*.nii.gz')))
    mask_paths = sorted(glob.glob(os.path.join(data_dir, 'labelsTr', '*.nii.gz')))

    assert len(img_paths) == len(mask_paths), \
        f"Mismatch: {len(img_paths)} images vs {len(mask_paths)} masks"

    profile = load_dataset_profile(data_dir)
    modality = profile['modality']
    if min_pixels is None:
        min_pixels = profile['default_min_pixels']
    if context_slices is None:
        context_slices = profile['default_context_slices']

    os.makedirs(os.path.join(out_dir, 'imgs'),  exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'masks'), exist_ok=True)

    print(f"Dataset: {profile['name']} | Modality: {modality or 'unknown'}")
    print(f"Keeping slices with >= {min_pixels} foreground pixels")
    print(f"Context slices on each side: {context_slices}")

    total_slices = 0
    for ip, mp in tqdm(zip(img_paths, mask_paths), total=len(img_paths),
                       desc='Extracting slices'):
        slices = preprocess_volume(
            ip,
            mp,
            target_size=target_size,
            min_pixels=min_pixels,
            modality=modality,
            context_slices=context_slices,
        )
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
    parser.add_argument('--min_pixels', default=None, type=int,
                        help='Minimum foreground pixels to keep a slice; dataset-aware when omitted')
    parser.add_argument('--context_slices', default=None, type=int,
                        help='Extra neighboring slices to keep on each side of a positive slice')
    args = parser.parse_args()

    n = extract_all_slices(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        target_size=(args.size, args.size),
        min_pixels=args.min_pixels,
        context_slices=args.context_slices,
    )
    print(f"\nReady to train on {n} slices.")
    print(f"Next step: python train.py")

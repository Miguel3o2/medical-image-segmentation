# Medical Image Segmentation - U-Net (PyTorch)

Medical image segmentation using a U-Net trained from scratch in PyTorch.

This project started as a liver CT segmentation pipeline for Medical Segmentation Decathlon Task03 and was later extended to support Task04 Hippocampus MRI with dataset-aware preprocessing. The latest run reached a best validation Dice of about `0.90` on Task04.

## Results

| Metric | Score |
|--------|-------|
| Best validation Dice | 0.9027 |
| IoU | Run `python evaluate.py` to refresh |
| Precision | Run `python evaluate.py` to refresh |
| Recall | Run `python evaluate.py` to refresh |

### Predictions

![Predictions](assets/predictions.png)

### Training Curves

![Training curves](assets/training_curves.png)

## Architecture

U-Net with 4 encoder levels and a symmetric decoder:

```text
Input (1x256x256)
  -> Encoder: 64 -> 128 -> 256 -> 512 channels
  -> Bottleneck: 1024 channels
  -> Decoder: 512 -> 256 -> 128 -> 64 channels
  -> Output (1x256x256) binary segmentation mask
```

- Skip connections use concatenation to preserve spatial detail.
- Dice+BCE loss balances overlap quality with stable optimisation.
- The model has about 31M parameters.

## Supported Datasets

### Task03 Liver

- CT volumes with liver and tumour masks
- CT-style windowing and slice extraction
- Good fit when you have more disk space available

### Task04 Hippocampus

- MRI volumes with hippocampus labels
- MRI-aware normalisation for non-CT intensity distributions
- Small-structure-friendly slice filtering with neighboring context slices
- Better option for smaller local storage budgets

## Key Engineering Improvements

- Dataset-aware preprocessing based on `dataset.json`
- MRI normalisation path added for Task04
- Foreground thresholding now works through `--min_pixels`
- Context slices are retained around positive hippocampus slices
- Training no longer silently drops the only batch when the train split is small
- Clearer error message when a training loader is empty

## Setup

### Requirements

- Python 3.9+
- CUDA GPU recommended
- Enough disk space for your chosen dataset

### Install

```powershell
git clone https://github.com/YOUR_USERNAME/medical-image-segmentation.git
cd medical-image-segmentation

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Preprocess

```powershell
# Task03 Liver
python data/preprocess.py --data_dir Task03_Liver --out_dir slices

# Task04 Hippocampus
python data/preprocess.py --data_dir Task04_Hippocampus --out_dir slices_task04
```

Optional Task04 tuning:

```powershell
python data/preprocess.py --data_dir Task04_Hippocampus --out_dir slices_task04 --size 256 --min_pixels 10 --context_slices 2
```

### 2. Sanity Check

```powershell
python evaluate.py --data_dir slices_task04 --sanity_check --batch_size 8 --num_workers 0
```

### 3. Train

```powershell
# Task03 Liver
python train.py --data_dir slices --epochs 30 --batch_size 8 --lr 1e-4 --num_workers 0

# Task04 Hippocampus
python train.py --data_dir slices_task04 --epochs 60 --batch_size 8 --lr 1e-4 --num_workers 0
```

If VRAM is tight, reduce the batch size to `4`.

### 4. Evaluate

```powershell
python evaluate.py --data_dir slices_task04 --checkpoint checkpoints/unet_best.pt --batch_size 8 --num_workers 0
```

This writes:

- `assets/predictions.png`
- `assets/training_curves.png`
- `assets/metrics_report.txt`

## Project Structure

```text
medical-image-segmentation/
|-- model/
|   |-- unet.py
|   `-- losses.py
|-- data/
|   |-- preprocess.py
|   `-- dataset.py
|-- train.py
|-- evaluate.py
|-- assets/
`-- checkpoints/
```

## Key Implementation Details

**Why Dice loss?** Medical segmentation is highly imbalanced. Standard BCE can reward background-heavy predictions too easily, while Dice directly optimises overlap quality.

**Why 2D slices?** A full 3D U-Net is expensive on consumer hardware. Extracting 2D slices keeps training practical while preserving a strong baseline for both CT and MRI workflows.

**Why dataset-aware preprocessing?** Liver CT assumptions do not cleanly transfer to hippocampus MRI. Modality-aware normalisation and small-structure slice retention made the pipeline much more robust.

## Why This Project Is Interesting

- It turns 3D NIfTI medical volumes into a practical 2D training pipeline for consumer GPUs.
- It highlights how preprocessing assumptions can fail across modalities.
- It shows end-to-end ML engineering: preprocessing, training, evaluation, debugging, and presentation.

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/)

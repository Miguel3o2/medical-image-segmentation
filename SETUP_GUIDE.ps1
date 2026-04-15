# ============================================================
#  COMPLETE SETUP GUIDE — Medical Image Segmentation Project
#  Run these commands IN ORDER in Windows PowerShell
# ============================================================

# ── STEP 0: Check Python is installed ──────────────────────
python --version
# Need 3.9+. If missing: https://www.python.org/downloads/

# ── STEP 1: Install Git (if not installed) ─────────────────
# Download: https://git-scm.com/download/win
git --version    # verify it worked

# ── STEP 2: Navigate to where you want the project ─────────
cd C:\Users\YourName\Documents   # change this to your preferred folder

# ── STEP 3: Create project folder ──────────────────────────
# (skip if you're using the downloaded files directly)
# git clone https://github.com/YOUR_USERNAME/medical-image-segmentation.git
# cd medical-image-segmentation

# ── STEP 4: Create a virtual environment ───────────────────
python -m venv venv

# Activate it
venv\Scripts\Activate.ps1

# If you get an execution policy error, run this ONCE:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then try activating again:
venv\Scripts\Activate.ps1

# Your prompt should now show (venv) at the start

# ── STEP 5: Upgrade pip ────────────────────────────────────
python -m pip install --upgrade pip

# ── STEP 6: Install PyTorch ────────────────────────────────

# Option A: WITH CUDA (NVIDIA GPU) — check your CUDA version first:
# nvidia-smi  ← run this to see your CUDA version

# For CUDA 12.1 (most common on modern GPUs):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Option B: CPU only (slower but works everywhere)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Verify torch installed correctly:
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"

# ── STEP 7: Install remaining dependencies ─────────────────
pip install -r requirements.txt

# ── STEP 8: Download the dataset (~8 GB) ───────────────────
# Option A: via gdown (easiest)
pip install gdown
gdown --id 1jyVGUGyxKBXV6_9ivuZapQS8eUJXCIpu

# Option B: manual download
# Go to: http://medicaldecathlon.com/
# Click Task03_Liver → download Task03_Liver.tar

# Extract the tar file
# Option A — using Python (works everywhere):
python -c "import tarfile; tarfile.open('Task03_Liver.tar').extractall('.')"

# Option B — install 7-Zip and right-click → Extract

# Verify extraction:
dir Task03_Liver\imagesTr\   # should show .nii.gz files

# ── STEP 9: Preprocess — extract 2D slices (run once) ──────
python data\preprocess.py --data_dir Task03_Liver --out_dir slices

# This takes 5–15 minutes. Output:
#   slices\imgs\00000.npy  ...  (CT slices)
#   slices\masks\00000.npy ...  (liver masks)
# Expect ~4000–6000 slices total.

# ── STEP 10: Sanity check — verify data is correct ─────────
python evaluate.py --sanity_check --num_workers 0
# Opens assets/sanity_check.png
# Verify: CT slices look like CT, masks show liver region
# If masks look wrong — check preprocess.py threshold

# ── STEP 11: Quick model test ───────────────────────────────
python model\unet.py
# Should print:
#   Input:  torch.Size([2, 1, 256, 256])
#   Output: torch.Size([2, 1, 256, 256])
#   Params: 31,xxx,xxx
#   Sanity check passed!

# ── STEP 12: Train the model ────────────────────────────────

# Standard (16GB+ VRAM or CPU):
python train.py --epochs 30 --batch_size 16 --lr 1e-4

# Low VRAM (8GB GPU):
python train.py --epochs 30 --batch_size 8 --lr 1e-4

# Very low VRAM / CPU only:
python train.py --epochs 30 --batch_size 4 --lr 1e-4 --num_workers 0

# Training output per epoch:
#   Epoch 001/030  loss 1.4821  dice 0.1234  best 0.1234  lr 1.00e-04

# Expected milestones:
#   Epochs  1-3 : Dice 0.10–0.35  (model learning shapes)
#   Epochs  5-10: Dice 0.55–0.72  (liver region found)
#   Epochs 10-20: Dice 0.75–0.85  (boundaries sharpening)
#   Epochs 20-30: Dice 0.85–0.92  (fine refinement)

# ── STEP 13: Evaluate the best model ───────────────────────
python evaluate.py --checkpoint checkpoints\unet_best.pt --num_workers 0

# Generates:
#   assets/predictions.png       ← prediction overlays
#   assets/training_curves.png   ← loss/dice curves
#   assets/metrics_report.txt    ← your final scores

# ── STEP 14: Push to GitHub ────────────────────────────────

# Create a new repo on github.com (do NOT initialise with README)
# Then run:

git init
git add .
git commit -m "U-Net liver segmentation - Dice 0.XX on MSD Task03"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/medical-image-segmentation.git
git push -u origin main

# ── COMMON ERRORS & FIXES ──────────────────────────────────

# Error: "CUDA out of memory"
# Fix:  reduce --batch_size to 8 or 4

# Error: "RuntimeError: DataLoader worker ... exited unexpectedly"
# Fix:  add --num_workers 0  (Windows multiprocessing issue)

# Error: "ModuleNotFoundError: No module named 'nibabel'"
# Fix:  pip install nibabel  (make sure venv is activated)

# Error: "gdown: Cannot retrieve the public link"
# Fix:  download Task03_Liver.tar manually from medicaldecathlon.com

# Error: "FileNotFoundError: slices/imgs/*.npy"
# Fix:  run data/preprocess.py first (Step 9)

# Error: Dice stuck at 0.0 after 5+ epochs
# Fix:  run: python evaluate.py --sanity_check
#       check that masks show liver, not all-zeros

# ── DEACTIVATE VENV WHEN DONE ──────────────────────────────
deactivate

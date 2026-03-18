#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# setup.sh  —  Install deps & pre-download model weights
# Supports Ubuntu 22.04 and Ubuntu 24.04 (noble)
# ─────────────────────────────────────────────────────────────
set -euo pipefail

echo "========================================="
echo "  Video Understanding AI  —  Setup"
echo "========================================="

# ── 1. System packages ────────────────────────────────────────
if command -v apt-get &>/dev/null; then
  sudo apt-get update -q

  UBUNTU_VER=$(lsb_release -rs 2>/dev/null || echo "0")
  echo "Detected Ubuntu: $UBUNTU_VER"

  if [[ "$UBUNTU_VER" == "24.04" ]] || [[ "$UBUNTU_VER" > "23" ]]; then
    PY_PKGS="python3.12 python3.12-dev python3.12-venv"
    PY_BIN="python3.12"
    GLIB_PKG="libglib2.0-0t64"
  else
    PY_PKGS="python3.11 python3.11-dev python3.11-venv"
    PY_BIN="python3.11"
    GLIB_PKG="libglib2.0-0"
  fi

  sudo apt-get install -y \
    ffmpeg libgl1 $GLIB_PKG libsm6 libxext6 python3-pip $PY_PKGS

  echo "Python: $($PY_BIN --version)"
else
  PY_BIN="python3"
fi

# ── 2. Upgrade pip ────────────────────────────────────────────
$PY_BIN -m pip install --upgrade pip

# ── 3. PyTorch (CUDA 12.4) ───────────────────────────────────
echo ""
echo "Installing PyTorch (CUDA 12.4)…"
$PY_BIN -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# ── 4. Project requirements ───────────────────────────────────
echo ""
echo "Installing project requirements…"
$PY_BIN -m pip install -r requirements.txt

# ── 5. Pre-download model weights ────────────────────────────
echo ""
echo "Pre-downloading Qwen2.5-VL-7B-Instruct (~15 GB)…"

# Read .env manually to avoid dotenv frame assertion error in inline scripts
MODEL_NAME=$(grep '^MODEL_NAME' .env 2>/dev/null | cut -d'=' -f2 || echo "Qwen/Qwen2.5-VL-7B-Instruct")
MODEL_CACHE_DIR=$(grep '^MODEL_CACHE_DIR' .env 2>/dev/null | cut -d'=' -f2 || echo "./models_cache")

$PY_BIN - <<PYEOF
import os, torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

model_name = "${MODEL_NAME}"
cache_dir  = "${MODEL_CACHE_DIR}"
os.makedirs(cache_dir, exist_ok=True)
print(f"  Model : {model_name}")
print(f"  Cache : {cache_dir}")
print("  Downloading processor…")
AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
print("  Downloading model weights…")
Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16, trust_remote_code=True,
)
print("  ✓ Done.")
PYEOF

echo ""
echo "========================================="
echo "  Setup complete!"
echo "  Run:  python app.py"
echo "  Open: http://localhost:8000"
echo "========================================="
# ─────────────────────────────────────────────────────────────
# Video Understanding AI  —  Qwen2.5-VL-7B  on 2× NVIDIA L4
# ─────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── System deps ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    git curl wget && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3        1

WORKDIR /app

# ── Python deps (cached layer) ────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    pip install -r requirements.txt

# ── App source ────────────────────────────────────────────────
COPY . .

# ── Ports ─────────────────────────────────────────────────────
EXPOSE 8000

# ── Run ───────────────────────────────────────────────────────
CMD ["python", "app.py"]
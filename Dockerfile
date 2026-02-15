# ============================================================
# Energy-Aware Manipulation — GPU-Accelerated Dev Container
# Base: NVIDIA CUDA 12.1 + Ubuntu 22.04
# Includes: Python 3.10, PyTorch 2.3 (CUDA 12.1), MuJoCo,
#           robosuite, Stable-Baselines3, Sentence-BERT
# ============================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    unzip \
    # MuJoCo / rendering dependencies
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglew-dev \
    libglfw3 \
    libglfw3-dev \
    libosmesa6-dev \
    patchelf \
    # OpenCV dependencies (required by robosuite)
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # General build tools
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# ---- Ensure python3 -> python3.10, pip up to date ----
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && python -m pip install --upgrade pip setuptools wheel

# ---- PyTorch with CUDA 12.1 support ----
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# ---- MuJoCo rendering environment ----
# Use OSMesa for headless rendering (no display required)
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# ---- Python dependencies ----
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# ---- Working directory ----
WORKDIR /workspace

# ---- Default command ----
CMD ["bash"]

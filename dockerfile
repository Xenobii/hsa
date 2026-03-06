FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch (CUDA 12.8 build)
RUN pip install --no-cache-dir \
    torch \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
RUN pip install --no-cache-dir \
    pretty_midi \
    mir_eval \
    resampy \
    nnaudio \
    hydra-core \
    pandas \
    omegaconf \
    tqdm \
    matplotlib \
    h5py

WORKDIR /workspace
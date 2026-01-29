# HeartMuLa Studio - Docker Image
# Multi-stage build for optimized image size

# =============================================================================
# Stage 1: Build Frontend
# =============================================================================
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Copy package files first for better caching
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy frontend source
COPY frontend/ ./

# Build the frontend
RUN npm run build

# =============================================================================
# Stage 2: Final Image with Python + CUDA
# =============================================================================
# Use NVIDIA PyTorch container which includes CUDA and PyTorch pre-installed
# and is optimized for ARM64/GH100
FROM nvcr.io/nvidia/pytorch:24.02-py3

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# Note: Python, GCC, etc. are usually included in the base image, but we ensure
# specific runtime deps like ffmpeg are present.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    gcc \
    g++ \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 heartmula && \
    mkdir -p /app/backend/models /app/backend/generated_audio /app/backend/ref_audio /app/backend/db && \
    chown -R heartmula:heartmula /app

# Copy backend requirements first for better caching
COPY --chown=heartmula:heartmula backend/requirements.txt /app/backend/

# Install Python dependencies
# Layer 1: pip upgrade
RUN pip3 install --no-cache-dir --upgrade pip

# Force uninstall potential incompatible pre-installed versions
RUN pip3 uninstall -y numpy scipy pandas transformers accelerate soxr

# Ensure compatible NumPy version (avoid NumPy 2.x which breaks some libs)
RUN pip3 install --no-cache-dir "numpy<2"

# Install Triton and build soxr from source to fix NumPy ABI mismatch
RUN pip3 install --no-cache-dir triton && \
    pip3 install --no-cache-dir --no-binary soxr soxr

# Note: PyTorch, torchvision, and torchaudio are already installed in the base image.
# We skip installing them manually to avoid conflicts and leverage the optimized versions.

# Generate constraints file to lock system packages (torch, numpy, torchaudio, torchvision)
# We include all torch packages to prevent pip from upgrading them
RUN pip3 freeze | grep -E "^torch|^numpy" > /tmp/constraints.txt

# Layer 2: Other requirements (using constraints to protect system packages)
# We manually handle heartlib to patch its numpy requirement, so we remove it from requirements.txt first
RUN sed -i '/heartlib/d' /app/backend/requirements.txt && \
    pip3 install --no-cache-dir -r /app/backend/requirements.txt -c /tmp/constraints.txt

# Layer 2.5: Install patched heartlib (removing strict requirements to use system packages)
RUN git clone https://github.com/HeartMuLa/heartlib.git /tmp/heartlib && \
    # Remove strict dependency checks to prevent pip from conflicting with system packages
    sed -i '/numpy/d' /tmp/heartlib/pyproject.toml || true && \
    sed -i '/numpy/d' /tmp/heartlib/setup.py || true && \
    sed -i '/torch/d' /tmp/heartlib/pyproject.toml || true && \
    sed -i '/torch/d' /tmp/heartlib/setup.py || true && \
    sed -i '/torchaudio/d' /tmp/heartlib/pyproject.toml || true && \
    sed -i '/torchaudio/d' /tmp/heartlib/setup.py || true && \
    sed -i '/torchvision/d' /tmp/heartlib/pyproject.toml || true && \
    sed -i '/torchvision/d' /tmp/heartlib/setup.py || true && \
    sed -i '/bitsandbytes/d' /tmp/heartlib/pyproject.toml || true && \
    sed -i '/bitsandbytes/d' /tmp/heartlib/setup.py || true && \
    pip3 install --no-cache-dir /tmp/heartlib -c /tmp/constraints.txt && \
    rm -rf /tmp/heartlib

# Layer 3: Rebuild torchaudio/torchvision from source to match system torch ABI
# We use specific commits from Jan 2024 to match the PyTorch 2.3.0a0+ebedce2 version in the base image.
# This prevents ABI mismatches (undefined symbols) and ensures compatibility.
RUN pip3 uninstall -y torchaudio torchvision || true && \
    cd /tmp && \
    git clone https://github.com/pytorch/audio.git && \
    cd audio && \
    git checkout 258169e53f1c662a9b1c787031ebdc5d7dfb7935 && \
    USE_CUDA=0 pip3 install . --no-deps --no-build-isolation --no-cache-dir && \
    cd .. && rm -rf audio && \
    git clone https://github.com/pytorch/vision.git && \
    cd vision && \
    git checkout 71b27a00eefc1b169d1469434c656dd4c0a5b18d && \
    pip3 install . --no-deps --no-build-isolation --no-cache-dir && \
    cd .. && rm -rf vision

# Layer 4: Ensure core ML libs are consistent (using system constraints)
# We avoid force-reinstalling torch to preserve NVIDIA binaries
RUN pip3 install --upgrade --no-cache-dir transformers accelerate bitsandbytes tokenizers sentencepiece -c /tmp/constraints.txt

# Fix runtime linking for libtorch_cuda.so (required by torchaudio)
# We dynamically find the torch lib path and symlink libs to /usr/lib to ensure loader finds them
# CAUTION: Do NOT overwrite libtorch_cpu.so if it exists, as it breaks libshm.so
RUN export TORCH_LIB_PATH=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))") && \
    echo "$TORCH_LIB_PATH" > /etc/ld.so.conf.d/torch.conf && \
    ldconfig && \
    if [ -f "$TORCH_LIB_PATH/libtorch.so" ] && [ ! -f "$TORCH_LIB_PATH/libtorch_cuda.so" ]; then \
        ln -sf "$TORCH_LIB_PATH/libtorch.so" "$TORCH_LIB_PATH/libtorch_cuda.so"; \
    fi && \
    if [ -f "$TORCH_LIB_PATH/libtorch.so" ]; then \
        ln -sf "$TORCH_LIB_PATH/libtorch.so" /usr/lib/libtorch.so; \
    fi && \
    if [ -f "$TORCH_LIB_PATH/libtorch_cuda.so" ]; then \
        ln -sf "$TORCH_LIB_PATH/libtorch_cuda.so" /usr/lib/libtorch_cuda.so; \
    fi || true

# Copy backend code
COPY --chown=heartmula:heartmula backend/ /app/backend/

# Copy built frontend from Stage 1
COPY --from=frontend-builder --chown=heartmula:heartmula /app/frontend/dist /app/frontend/dist

# Copy startup script
COPY --chown=heartmula:heartmula start.sh /app/

# Environment variables for HeartMuLa
ENV PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HEARTMULA_4BIT=auto \
    HEARTMULA_SEQUENTIAL_OFFLOAD=auto \
    HF_HOME=/app/backend/models \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache \
    LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH

# Expose port
EXPOSE 8000

# Switch to non-root user
USER heartmula

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command - run the backend server
CMD ["python3", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

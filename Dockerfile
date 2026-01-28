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

# Generate constraints file to lock system packages (torch, torchaudio, etc.)
# This prevents pip from accidentally upgrading NVIDIA-optimized packages to incompatible PyPI wheels
RUN pip3 freeze > /tmp/constraints.txt

# Layer 2: Other requirements (using constraints to protect system packages)
RUN pip3 install --no-cache-dir -r /app/backend/requirements.txt -c /tmp/constraints.txt

# Layer 3: Force clean reinstall of core ML libs to fix 'GenerationMixin' errors
# We also use constraints here to ensure they link against the system torch
RUN pip3 install --force-reinstall --no-cache-dir transformers accelerate bitsandbytes tokenizers sentencepiece -c /tmp/constraints.txt

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
    TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache

# Expose port
EXPOSE 8000

# Switch to non-root user
USER heartmula

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command - run the backend server
CMD ["python3", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

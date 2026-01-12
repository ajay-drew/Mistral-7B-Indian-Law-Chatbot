# Dockerfile for Mistral Indian Law Chatbot
# Supports NVIDIA GPU on Windows host (WSL2/Docker Desktop with GPU support)

FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support first (large package)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Note: Flash Attention removed - using standard PyTorch attention

# Copy application code
COPY backend/ ./backend/
COPY mistral-indian-law-final/ ./mistral-indian-law-final/
COPY data/ ./data/
COPY offload_dir/ ./offload_dir/ 2>/dev/null || mkdir -p ./offload_dir

# Create necessary directories
RUN mkdir -p ./data/chroma_db ./offload_dir

# Expose backend port
EXPOSE 2347

# Set environment variables
ENV BASE_MODEL_NAME=mistralai/Mistral-7B-v0.1
ENV ADAPTER_PATH=./mistral-indian-law-final
ENV DEVICE_MAP=auto
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "2347"]

FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

WORKDIR /app

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    apt-utils \
    wget \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
        python3.12 \
        python3.12-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Clone SAM3 repository
RUN git clone https://github.com/facebookresearch/sam3.git /app/sam3

RUN pip3 install --upgrade pip setuptools wheel
# Install SAM3
RUN cd /app/sam3 && pip3 install -e .

# Install PyTorch with CUDA support
RUN pip3 install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Copy server code
COPY . /app/server

# Install server requirements
RUN pip3 install --no-cache-dir -r /app/server/requirements.txt

WORKDIR /app/server

# Create directories
RUN mkdir -p /tmp/sam3_uploads /tmp/sam3_outputs

# Expose ports
EXPOSE 8000 9090

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python3 -c "import requests; requests.get('http://localhost:8000/health')"

# Start server

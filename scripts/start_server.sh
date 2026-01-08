#!/bin/bash
# Start SAM3 Inference Server

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting SAM3 Inference Server...${NC}"

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found. Copying from .env.example${NC}"
    cp .env.example .env
    echo -e "${YELLOW}Please edit .env with your configuration${NC}"
    exit 1
fi

# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Start server
echo -e "${GREEN}Launching server...${NC}"
python server.py

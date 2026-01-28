#!/bin/bash
# Script to run agent with GPU support
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$(pwd)/.venv/lib/python3.12/site-packages/nvidia/cublas/lib
echo "Starting Agent with GPU Support..."
.venv/bin/python main.py

#!/bin/bash
# Cài đặt llama-cpp-python với hỗ trợ CUDA (GPU)
# Lưu ý: Cần cài đặt CUDA Toolkit trước khi chạy script này.

CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

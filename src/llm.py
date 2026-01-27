# src/llm.py
import sys
import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from .config import MODEL_REPO, MODEL_FILENAME, HF_TOKEN, logger

# --- 1. Model Loading (CPU / GGUF) ---
def load_model():
    """Download and load the GGUF model for CPU inference."""
    logger.info(f"Checking module {MODEL_FILENAME} from {MODEL_REPO}...")
    try:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME,
            token=HF_TOKEN
        )
        logger.info(f"Model downloaded to: {model_path}")
        
        # Initialize Llama (n_ctx=8192 for RAG context)
        llm = Llama(
            model_path=model_path,
            n_ctx=8192,      # Increased context window for RAG
            n_threads=6,
            n_threads_batch=6,
            verbose=False
        )
        logger.info("Qwen3 GGUF Model loaded successfully.")
        return llm
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

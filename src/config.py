# src/config.py
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
MODEL_REPO = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
MODEL_FILENAME = "Qwen3-4B-Instruct-2507-Q6_K.gguf"
DATA_DIR = "./data_investment"  # Directory containing .txt files for RAG
CACHE_DIR = "./.rag_cache"  # Directory to store RAG cache files
HF_TOKEN = os.getenv("HF_TOKEN")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- Logging Configuration ---
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logging()

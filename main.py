# main.py
import argparse
import sys
from src.config import DATA_DIR
from src.rag import InvestmentRAG
from src.llm import load_model
from src.agent import QwenAgent
from src.setup_mapping import download_and_process_mappings
from src.ui import create_ui

# Ensure we are running in a virtual environment
if sys.prefix == sys.base_prefix:
    print("❌ Error: You must run this application within the .venv environment.")
    print("👉 Please run: source .venv/bin/activate")
    sys.exit(1)


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Qwen3 Multi-Step Agent + Investment RAG")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild RAG cache, ignoring existing cache"
    )
    args = parser.parse_args()
    
    # 0. Setup/Update Mappings
    print("🔄 Ensuring ticker mappings are up-to-date...")
    download_and_process_mappings()

    # 1. Load Model
    llm_instance = load_model()
    
    # 2. Initialize RAG (with optional force rebuild)
    rag_instance = InvestmentRAG(DATA_DIR)
    rag_instance.initialize(llm=llm_instance, force_rebuild=args.rebuild)
    
    # 3. Initialize Agent
    agent_instance = QwenAgent(llm_instance, rag_instance)
    
    # 4. Launch UI
    print("🚀 Launching UI...")
    ui = create_ui(agent_instance, rag_instance, llm_instance)
    ui.launch(server_name="0.0.0.0", share=True)


if __name__ == "__main__":
    main()

# AI Agents Collection: RAG & Multi-Step Reasoning (Python Edition)

This repository hosts a modular Python application integrating advanced AI agents for financial analysis. It features a **Multi-Step Reasoning Agent** and a **Retrieval-Augmented Generation (RAG)** system optimized for investment documents.

## 🌟 Key Features

### 🤖 Multi-Step Agent (`src/agent/`)
- **Model**: `Qwen/Qwen3-4B-Instruct-2507` (GGUF format).
- **Architecture**: **Graph-based (LangGraph)**.
- **Workflow**:
  - **Intent Analysis**: Determines the user's goal and required output language.
  - **Planning**: Strategies for complex queries.
  - **Generation**: Produces tool calls or direct answers.
  - **Tool Execution**: Handles safe execution of tools.
  - **Synthesis**: Compiles final answers from tool outputs with citations.

### 📚 Investment RAG (`src/rag.py`)
- **Strategy**: **Summary Vector (Parent-Document Retrieval)**.
- **Parent Indexing**: Vectorizes Document Summaries for high-level semantic matching.
- **Child Retrieval**: Retrieves full document chunks associated with matched summaries.
- **Re-ranking**: Uses `BAAI/bge-reranker-base` to refine search results.
- **Smart Caching**: Caches embeddings and indices (FAISS) to speed up startup times.

### 🛠️ Tools (`src/tools/`)
Modular tools organized by domain:
- **Finance**: `get_stock_price`, `get_crypto_price`, symbol resolution.
- **Web**: `get_news` (Tavily), `crawl_url`, `scrape_web_page`.
- **Math**: `arithmetic_tool`.
- **RAG**: `query_knowledge_base`.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- [Optional] GPU support for faster inference (e.g., NVIDIA T4 on Colab).

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd "RAG agent"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   Create a `.env` file in the root directory:
   ```env
   HF_TOKEN=your_huggingface_token
   TAVILY_API_KEY=your_tavily_api_key
   ```

### 📂 Data Setup
Place your investment documents (text files, `.txt`) in the `data_investment/` directory. The RAG system will automatically index them on the first run.

## 🖥️ Usage

Run the main application:
```bash
python main.py
```

- **Force Rebuild RAG Cache**: If you added new documents and need to re-index:
  ```bash
  python main.py --rebuild
  ```

This will launch a **Gradio** web interface (local and public shareable link) where you can interact with the agent.

## 📂 Project Structure

```
├── main.py                 # Entry point
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
├── data_investment/        # Folder for RAG documents (.txt)
└── src/
    ├── agent/              # LangGraph Agent Logic
    │   ├── core.py         # Graph Definition & Compilation
    │   ├── nodes.py        # Graph Nodes (Intent, Plan, Generate, Tools, Synthesis)
    │   ├── state.py        # Agent State definition
    │   ├── intent_analyzer.py # Intent classification
    │   ├── planner.py      # Planning Logic
    │   └── ...
    ├── tools/              # Tool Definitions
    │   ├── finance.py      # Stock/Crypto
    │   ├── web.py          # News/Crawling
    │   └── ...
    ├── rag.py              # InvestmentRAG system (Summary Vector)
    ├── llm.py              # Model loading (llama-cpp-python)
    ├── config.py           # Configuration & logging
    └── setup_mapping.py    # Setup script
```

## 🛠️ Technologies
- **Inference**: `llama-cpp-python` (GGUF)
- **RAG**: `faiss-cpu`, `sentence-transformers`, `langchain-text-splitters`
- **Search & Data**: `yfinance`, `tavily-python`
- **UI**: `gradio`

---
*Created for Advanced Agentic Coding experiments.*

## 📜 License

This project is dual-licensed under the MIT and Apache 2.0 licenses. You may use this code under the terms of either license.

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

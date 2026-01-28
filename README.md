# AI Agents Collection: RAG & Multi-Step Reasoning (Python Edition)

This repository hosts a modular Python application integrating advanced AI agents for financial analysis. It features a **Multi-Step Reasoning Agent** and a **Retrieval-Augmented Generation (RAG)** system optimized for investment documents.

## 🌟 Key Features

### 🤖 Multi-Step Agent (`src/agent.py`)
- **Model**: `Qwen/Qwen3-4B-Instruct-2507` (GGUF format).
- **Capabilities**:
  - **Tool Use**: Autonomous execution of tools for real-time data and calculations.
  - **Contextual Memory**: Handles follow-up questions and maintains conversation history.
  - **Reasoning**: Breaks down complex queries into logical steps.

### 📚 Investment RAG (`src/rag.py`)
- **Hybrid Search**: Combines **FAISS** (Dense Vector Search) and **BM25** (Sparse Keyword Search) for robust retrieval.
- **Re-ranking**: Uses `BAAI/bge-reranker-base` to refine search results.
- **Contextual Retrieval**: Enhances chunks with document summaries for better context understanding.
- **Smart Caching**: Caches embeddings and indices to speed up startup times.

### 🛠️ Tools (`src/tools.py`)
- `get_stock_price(symbol)`: Real-time stock/crypto data via `yfinance`.
- `get_news(query)`: Latest news search via `Tavily` API.
- `arithmetic_tool(op, a, b)`: Precise mathematical operations.
- `query_knowledge_base(query)`: Access to internal investment documents.

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

## � Project Structure

```
├── main.py                 # Entry point
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
├── data_investment/        # Folder for RAG documents (.txt)
└── src/
    ├── agent.py            # QwenAgent logic & tool parsing
    ├── rag.py              # InvestmentRAG system (FAISS + BM25)
    ├── llm.py              # Model loading (llama-cpp-python)
    ├── tools.py            # Tool definitions (yfinance, tavily)
    └── config.py           # Configuration & logging
```

## 🛠️ Technologies
- **Inference**: `llama-cpp-python` (GGUF)
- **RAG**: `faiss-cpu`, `rank_bm25`, `sentence-transformers`
- **Search & Data**: `yfinance`, `tavily-python`
- **UI**: `gradio`

---
*Created for Advanced Agentic Coding experiments.*

# Architecture Guide

This document provides a high-level overview of the RAG Agent Architecture.

## 1. Project Structure

The project follows a modular structure within the `src/` directory, with `main.py` serving as the entry point.

```
RAG agent/
├── main.py                 # Entry point, Gradio UI, and orchestration
├── src/
│   ├── agent/              # Agent Logic Package
│   │   ├── core.py         # QwenAgent: Function calling loop & main orchestration
│   │   ├── planner.py      # Query analysis & planning logic
│   │   ├── parser.py       # Tool call parsing logic
│   │   └── summarizer.py   # Text summarization for crawled content
│   ├── tools/              # Tool Definitions Package
│   │   ├── finance.py      # Stock (yfinance) & Crypto (Binance) tools
│   │   ├── web.py          # News (Tavily), Crawling, Scraping tools
│   │   └── math.py         # Arithmetic tools
│   ├── rag.py              # InvestmentRAG: Parent-Document Retrieval implementation
│   ├── llm.py              # LlamaCpp model loading (CPU/GPU)
│   ├── config.py           # Configuration constants (Paths, Keys)
│   └── setup_mapping.py    # Utilities for stock symbol mapping
├── data_investment/        # Directory for RAG input text files
├── .rag_cache/             # Cache storage for RAG index and chunks
└── requirements.txt        # Python dependencies
```

## 2. Core Components

### 2.1. Agent (`src/agent/`)
- **Model**: Uses `Qwen/Qwen2.5-3B-Instruct` (or similar GGUF) via `llama-cpp-python`.
- **Logic**: Implements a ReAct-style loop (Reasoning + Acting).
- **Planner (`src/agent/planner.py`)**: Analyzes user requests to identify information gaps and propose search strategies ("Paranoid Query Analyzer").
- **Core (`src/agent/core.py`)**: Manages the agent loop, history, and integration with tools.
- **Parser (`src/agent/parser.py`)**: 
  - Primary: Checks for `<tool_call>...</tool_call>` XML tags.
  - Fallback: Checks for JSON-like structures.
- **Summarizer (`src/agent/summarizer.py`)**: Summarizes large text blocks from web crawling to manage context window.

### 2.2. Tools (`src/tools/`)
Tools are modularized by domain:
- **Finance**: Stock Price (yfinance), Crypto Price (Binance).
- **Web**: News Search (Tavily), URL Crawling, Web Scraping.
- **Math**: Basic arithmetic.
- **RAG**: Internal knowledge base search (invoked via Agent).

### 2.3. RAG System (`src/rag.py`)
- **Strategy**: **Summary Vector (Parent-Document Retrieval)**.
- **Mechanism**:
  1. **Ingestion**: Documents are split into chunks. A summary is generated for the whole document (Parent).
  2. **Indexing**: Only the **Parent Summary** is vectorized and stored in FAISS.
  3. **Storage**: Detailed **Child Chunks** are stored in a key-value store (dictionary/pickle), NOT vectorized.
  4. **Retrieval**:
     - Query matches against Parent Summaries.
     - Relevant Parent Documents are identified.
     - **All** Child Chunks from those parents are retrieved.
     - Chunks are re-ranked using a Cross-Encoder (`BAAI/bge-reranker-base`).
     - Top chunks are returned.
- **Caching**: Has rigorous caching mechanisms (hashing file contents) to avoid rebuilding the index unnecessarily.

### 2.4. LLM Layer (`src/llm.py`)
- **Engine**: `llama.cpp` (via `llama-cpp-python`).
- **Hardware**:
  - auto-detects CUDA (GPU) vs CPU.
  - Sets `n_gpu_layers=-1` for full GPU offloading if CUDA is available.
  - Configurable context window (default 8192+).

## 3. Data Flow

1.  **User Query** -> `main.py` -> `QwenAgent.run()`
2.  **Planning**: `planner.analyze_request()` injects search strategy if needed.
3.  **Agent Loop** (`src/agent/core.py`):
    -   Agent generates thought/plan.
    -   Agent outputs `<tool_call>`.
    -   `parser.parse_tool_calls()` extracts instructions.
    -   `QwenAgent` executes tool (e.g., `query_knowledge_base`, `get_stock_price`).
    -   **RAG Search**: Query -> Embedding -> FAISS (Summary Index) -> Re-ranking.
    -   **Web/Crawler**: Tool gets content -> `summarizer.summarize_text()` condenses it.
    -   Tool Result -> Agent History.
    -   Agent repeats or generates Final Answer.
4.  **Response** -> Gradio UI.

## 4. Key Design Decisions
-   **Modular Architecture**: Tools and Agent logic are split into sub-packages for easier maintenance and testing.
-   **GGUF Oriented**: Designed for local inference efficiency.
-   **Parent-Document Retrieval**: optimized for "investment" contexts where understanding the whole document context (Summary) is often better for retrieval than matching isolated keywords in small chunks.
-   **Gradio UI**: Provides a chat interface and a document upload tab for real-time RAG updates.

# Architecture Guide

This document provides a high-level overview of the RAG Agent Architecture.

## 1. Project Structure

The project follows a modular structure within the `src/` directory, with `main.py` serving as the entry point.

```
RAG agent/
├── main.py                 # Entry point
├── src/
│   ├── ui.py               # Gradio UI Implementation
│   ├── agent/              # Agent Logic Package
│   │   ├── core.py         # Graph Setup & Compilation
│   │   ├── nodes/          # Graph Node Implementations
│   │   │   ├── analyze_intent.py
│   │   │   ├── planning.py
│   │   │   ├── generate.py
│   │   │   ├── execute_tools.py
│   │   │   ├── synthesis.py
│   │   │   └── utils.py
│   │   ├── state.py        # TypedDict State Definition
│   │   ├── parser.py       # Tool call parsing
│   │   └── summarizer.py   # Text summarization
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
- **Framework**: Built using **LangGraph** for structured, stateful execution.
- **State (`src/agent/state.py`)**: Tracks conversation history, logs, intent, and plan.
- **Core (`src/agent/core.py`)**: Defines the graph topology (Nodes & Edges).
- **Core (`src/agent/core.py`)**: Defines the graph topology (Nodes & Edges).
- **Nodes (`src/agent/nodes/`)**:
  - **Intent** (`nodes/analyze_intent.py`): Analyzes user request for goal and language.
  - **Plan** (`nodes/planning.py`): Injects strategic hints for complex queries.
  - **Generate** (`nodes/generate.py`): **Pure Router/Searcher**. Decides whether to call tools or pass to synthesis. Does NOT generate final answers.
  - **Tools** (`nodes/execute_tools.py`): Executes requested tools and updates state with results.
  - **Synthesis** (`nodes/synthesis.py`): Final pass to consolidate tool outputs into a cohesive answer with citations.
  - **Utils** (`nodes/utils.py`): Helper functions for message history filtering and context management.
- **Parser (`src/agent/parser.py`)**: Handles extraction of `<tool_call>` XML tags.
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
1.  **User Query** -> `main.py` -> `Graph Agent`.
2.  **Intent Analysis Node**: Classifies intent (e.g., "market_data", "general_qa") and language.
3.  **Planning Node**: Checks if a breakdown is needed (e.g., for multi-part comparison).
4.  **Generation Node**: LLM receives State (Filtered History + Plan) -> Decides to call Tools or delegate to Synthesis.
5.  **Tools Node** (Conditional):
    -   Executes tools (`get_stock_price`, `get_news`, etc.).
    -   Updates State with Tool Outputs.
    -   *Logic loops back to Generation or proceeds to Synthesis*.
6.  **Synthesis Node**:
    -   Takes all history + Tool Outputs.
    -   Generates Final Answer with citations.
7.  **Response** -> Gradio UI.

## 4. Key Design Decisions
-   **Modular Architecture**: Tools and Agent logic are split into sub-packages for easier maintenance and testing.
-   **GGUF Oriented**: Designed for local inference efficiency.
-   **Parent-Document Retrieval**: optimized for "investment" contexts where understanding the whole document context (Summary) is often better for retrieval than matching isolated keywords in small chunks.
-   **Gradio UI (`src/ui.py`)**: Dedicated UI module with a modern design (Soft theme, Tabs) for chat and document management.

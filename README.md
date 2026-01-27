# AI Agents Collection: RAG & Multi-Step Reasoning

This repository contains advanced AI agents designed for financial analysis, investment insights, and complex multi-step reasoning.

## 🤖 1. Multi-Step Financial Agent (Qwen3 Edition)
**File**: `multi_step_agent_qwen.ipynb`
**Model**: `Qwen/Qwen3-4B-Instruct-2507`

A powerful agent capable of executing multi-step workflows, using external tools, and maintaining conversation context.

### ✨ Key Features
- **Advanced Tool Use**:
  - `get_price(symbol)`: Fetches real-time stock and crypto prices using `yfinance`.
  - `get_news(query)`: Searches for the latest news using `Tavily`.
  - `arithmetic_tool(op, a, b)`: Performs precise mathematical calculations.
- **Contextual Memory**: capable of handling follow-up questions (e.g., "What is BTC price?" -> "Multiply it by 2").
- **Interactive UI**: Built with **Gradio**, supporting a full chat interface with history.

### 🚀 Quick Start
1. Open `multi_step_agent_qwen.ipynb` in Google Colab (T4 GPU recommended).
2. Add your API keys to Colab Secrets:
   - `HF_TOKEN`: Hugging Face Token (with access to Qwen3).
   - `TAVILY_API_KEY`: Tavily Search API Key.
3. Run all cells. The notebook will install the latest `transformers` (required for Qwen3) and launch the UI.

---

## 🧠 2. RAG Investment Agent
**File**: `rag_investment.ipynb`
**Model**: `LiquidAI/LFM2.5-1.2B-Thinking`

A Retrieval-Augmented Generation system optimized for querying and analyzing investment documents.

### ✨ Key Features
- **Thinking Model**: Uses specific `<think>` tags to demonstrate reasoning before answering.
- **Hybrid Search**: FAISS (Dense) + BM25 (Sparse) for precise document retrieval.
- **Optimized Indexing**: Uses document summaries and metadata enrichment for better retrieval accuracy.

### 🚀 Quick Start
1. Open `rag_investment.ipynb` in Google Colab.
2. Place your investment documents (`.txt`) in the `data investment/` folder or link Google Drive.
3. Run the notebook to index data and start the Gradio chat interface.

---

## 🛠️ Requirements
- **Hardware**: T4 GPU (Google Colab Standard)
- **Libraries**: `transformers`, `accelerate`, `yfinance`, `tavily-python`, `gradio`, `faiss-cpu`, `rank_bm25`.

---
*Created for Advanced Agentic Coding experiments.*

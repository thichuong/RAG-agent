# RAG Investment Agent: Hybrid Search + Thinking Model

Advanced RAG (Retrieval-Augmented Generation) system optimized for financial investment data. This project leverages the **LiquidAI/LFM2.5-1.2B-Thinking** model to provide reasoned, context-aware answers.

## 🚀 Environment & Hardware
- **Platform**: Google Colab (Recommended)
- **Hardware**: **Tesla T4 GPU** (16GB VRAM)
- **Frameworks**: Transformers, PyTorch, FAISS, BM25, Gradio.

## 🧠 Key Technologies
- **Model**: `LiquidAI/LFM2.5-1.2B-Thinking` - A state-of-the-art 1.2B parameter model tuned for "System 2 Thinking" and logical reasoning using `<think>` tags.
- **Hybrid Search**: Combines **Dense Retrieval** (FAISS + BGE Embeddings) with **Sparse Retrieval** (BM25) for high-precision document matching.
- **Optimized Contextual Retrieval**:
    - **Document Summary Prepending**: Each document is summarized once by the LLM.
    - **Metadata Enrichment**: Filenames and broad summaries are prepended to every chunk.
    - **Performance**: 10x-20x faster indexing compared to traditional per-chunk contextualization.

## 🛠️ Setup Instructions

### 1. Requirements
Ensure you have a Hugging Face token with access to the LiquidAI models.

### 2. Google Colab Deployment
1. Upload `rag_investment.ipynb` to Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. (Optional) Mount your Google Drive to load data from a specific folder.
4. Add your `HF_TOKEN` to Colab Secrets (the key icon 🔑).

### 3. Data Preparation
Place your `.txt` files in a folder named `data investment` or specify a Google Drive path in the configuration cell.

## 🖥️ Usage
Run all cells in the notebook. A **Gradio UI** will be generated, providing a public URL to interact with your Investment Agent.

- **Ask about**: Value-at-Risk (VaR), Expected Shortfall, Portfolio Management, Currency Risk, etc.
- **Thinking Process**: The model will display its internal logic inside `<think>` tags before providing the final professional answer.

## 📂 Project Structure
- `rag_investment.ipynb`: Main application logic.
- `data investment/`: Directory for source documents.
- `README.md`: Project documentation.

---
*Created with ❤️ for Advanced Investment Analysis.*

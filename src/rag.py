# src/rag.py
import os
import glob
import logging
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import logger

class InvestmentRAG:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.documents = []
        self.chunks = []
        self.index = None
        self.bm25 = None
        self.embed_model = None
        self.reranker = None
        self.is_ready = False

    def initialize(self):
        """Load data, chunk, and build hybrid index."""
        logger.info("Initializing RAG Knowledge Base...")
        
        # 1. Load Data
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            logger.warning(f"Data directory '{self.data_dir}' created. Please add .txt files.")
            # Create a dummy file for demonstration if empty
            with open(os.path.join(self.data_dir, "sample_investment.txt"), "w") as f:
                f.write("Value at Risk (VaR) is a statistic that quantifies the extent of possible financial losses within a firm, portfolio, or position over a specific time frame.")
        
        files = glob.glob(os.path.join(self.data_dir, "*.txt"))
        if not files:
            logger.warning("No .txt files found in data directory.")
            return

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    self.documents.append({"filename": os.path.basename(file_path), "content": f.read()})
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

        # 2. Chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        for doc in self.documents:
            splits = text_splitter.split_text(doc["content"])
            for i, text in enumerate(splits):
                self.chunks.append({
                    "id": f"{doc['filename']}_{i}",
                    "content": text,
                    "filename": doc["filename"]
                })
        
        if not self.chunks:
            logger.warning("No text chunks created.")
            return

        # 3. Embeddings & FAISS
        logger.info("Loading Embedding Model & Re-ranker...")
        # Updated to use BGE-Base and CrossEncoder as per recent changes
        self.embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
        self.reranker = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
        
        logger.info("Encoding chunks...")
        embeddings = self.embed_model.encode([c["content"] for c in self.chunks], show_progress_bar=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        # 4. BM25
        tokenized_corpus = [c["content"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        self.is_ready = True
        logger.info(f"RAG System Ready. Indexed {len(self.chunks)} chunks.")

    def search(self, query: str, k: int = 3) -> str:
        if not self.is_ready:
            return "Knowledge base not initialized or empty."
        
        # Dense Search
        query_vec = self.embed_model.encode([query])
        D, I = self.index.search(np.array(query_vec).astype('float32'), k * 2)
        dense_indices = I[0].tolist()
        
        # Sparse Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:k * 2].tolist()
        
        # Hybrid Fusion (Simple Union)
        combined_indices = list(set(dense_indices) | set(bm25_indices))
        initial_results = [self.chunks[i] for i in combined_indices]
        
        # Re-ranking
        if not initial_results:
             return "No relevant documents found."
             
        logger.info(f"Re-ranking {len(initial_results)} candidates...")
        pairs = [[query, doc["content"]] for doc in initial_results]
        scores = self.reranker.predict(pairs)
        
        # Zip results with scores and sort
        ranked_results = sorted(zip(initial_results, scores), key=lambda x: x[1], reverse=True)
        
        # Select Top K
        results = [r[0] for r in ranked_results[:k]]
        
        # Format output
        context = "\n\n".join([f"[Source: {r['filename']}]\n{r['content']}" for r in results])
        return context

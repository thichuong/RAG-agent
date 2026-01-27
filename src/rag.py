# src/rag.py
import os
import glob
import json
import pickle
import hashlib
import logging
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import logger, CACHE_DIR


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
        self._cache_dir = CACHE_DIR

    def _compute_data_hash(self) -> str:
        """Compute hash of all txt files in data directory to detect changes."""
        hash_md5 = hashlib.md5()
        files = sorted(glob.glob(os.path.join(self.data_dir, "*.txt")))
        
        for file_path in files:
            # Include filename and modification time in hash
            hash_md5.update(file_path.encode())
            hash_md5.update(str(os.path.getmtime(file_path)).encode())
            
            # Also include file content hash for extra safety
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        
        return hash_md5.hexdigest()

    def _get_cache_paths(self):
        """Return paths for all cache files."""
        return {
            "meta": os.path.join(self._cache_dir, "cache_meta.json"),
            "faiss": os.path.join(self._cache_dir, "index.faiss"),
            "embeddings": os.path.join(self._cache_dir, "embeddings.npy"),
            "chunks": os.path.join(self._cache_dir, "chunks.pkl"),
            "bm25": os.path.join(self._cache_dir, "bm25.pkl"),
        }

    def _is_cache_valid(self) -> bool:
        """Check if cache exists and is valid (data hasn't changed)."""
        paths = self._get_cache_paths()
        
        # Check if all cache files exist
        for path in paths.values():
            if not os.path.exists(path):
                return False
        
        # Check if data hash matches
        try:
            with open(paths["meta"], "r") as f:
                meta = json.load(f)
            current_hash = self._compute_data_hash()
            return meta.get("data_hash") == current_hash
        except Exception:
            return False

    def _save_cache(self, embeddings: np.ndarray, tokenized_corpus: list):
        """Save all RAG components to cache files."""
        os.makedirs(self._cache_dir, exist_ok=True)
        paths = self._get_cache_paths()
        
        # Save metadata with data hash
        meta = {
            "data_hash": self._compute_data_hash(),
            "num_chunks": len(self.chunks),
        }
        with open(paths["meta"], "w") as f:
            json.dump(meta, f)
        
        # Save FAISS index
        faiss.write_index(self.index, paths["faiss"])
        
        # Save embeddings
        np.save(paths["embeddings"], embeddings)
        
        # Save chunks metadata
        with open(paths["chunks"], "wb") as f:
            pickle.dump(self.chunks, f)
        
        # Save BM25 tokenized corpus (for rebuilding BM25)
        with open(paths["bm25"], "wb") as f:
            pickle.dump(tokenized_corpus, f)
        
        logger.info(f"Cache saved to {self._cache_dir}")

    def _load_cache(self) -> bool:
        """Load RAG components from cache. Returns True if successful."""
        try:
            paths = self._get_cache_paths()
            
            # Load chunks
            with open(paths["chunks"], "rb") as f:
                self.chunks = pickle.load(f)
            
            # Load FAISS index
            self.index = faiss.read_index(paths["faiss"])
            
            # Load BM25 tokenized corpus and rebuild BM25
            with open(paths["bm25"], "rb") as f:
                tokenized_corpus = pickle.load(f)
            self.bm25 = BM25Okapi(tokenized_corpus)
            
            logger.info(f"Cache loaded: {len(self.chunks)} chunks")
            return True
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return False

    def _generate_summary(self, llm, text):
        """Generate a brief summary of the document using the LLM."""
        prompt = f"""<document>
{text}
</document>

You are an expert at financial document analysis. Please provide a brief, high-level summary (1-2 sentences) of the overall context and main topic of this document.
This summary will be used to help a search engine understand the broad context for small chunks of this document.

Answer format:
<think>
[Your reasoning here]
</think>
[Broad context summary here]
"""
        try:
            # Assuming llm is a Llama object from llama_cpp or similar interface
            response = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.1
            )
            content = response["choices"][0]["message"]["content"]
            if "</think>" in content:
                summary = content.split("</think>")[-1].strip()
            else:
                summary = content.strip()
            return summary.replace("\n", " ")  # Keep it on one line ideally
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return ""

    def initialize(self, llm=None, force_rebuild: bool = False):
        """Load data, chunk, and build hybrid index.
        
        Args:
            llm: Optional LLM instance for generating contextual summaries.
            force_rebuild: If True, ignore cache and rebuild from scratch.
        """
        logger.info("Initializing RAG Knowledge Base...")
        
        # Load models first (always needed for search)
        logger.info("Loading Embedding Model & Re-ranker...")
        self.embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
        self.reranker = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
        
        # Check cache validity
        if not force_rebuild and self._is_cache_valid():
            logger.info("Loading RAG from cache...")
            if self._load_cache():
                self.is_ready = True
                logger.info(f"RAG System Ready (from cache). {len(self.chunks)} chunks loaded.")
                return
            logger.warning("Cache loading failed, rebuilding...")
        elif force_rebuild:
            logger.info("Force rebuilding RAG cache...")
        else:
            logger.info("Cache not found or invalid, building from scratch...")
        
        # Build from scratch
        self._build_index(llm)

    def _build_index(self, llm=None):
        """Build RAG index from scratch and save to cache."""
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

        doc_summaries = {}
        if llm:
            logger.info("Generating document summaries for Contextual Retrieval...")

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    filename = os.path.basename(file_path)
                    self.documents.append({"filename": filename, "content": content})
                    
                    if llm:
                        logger.info(f"Summarizing {filename}...")
                        summary = self._generate_summary(llm, content)
                        doc_summaries[filename] = summary
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

        # 2. Chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        for doc in self.documents:
            splits = text_splitter.split_text(doc["content"])
            filename = doc["filename"]
            summary = doc_summaries.get(filename, "")
            
            for i, text in enumerate(splits):
                chunk_data = {
                    "id": f"{filename}_{i}",
                    "content": text,
                    "filename": filename
                }
                
                # Contextual Retrieval Logic
                if summary:
                    # Combine Method A (Metadata) and Method B (Summary)
                    prefix = f"[File: {filename}] [Context: {summary}]"
                    chunk_data["contextualized_content"] = f"{prefix}\n{text}"
                else:
                    chunk_data["contextualized_content"] = text
                    
                self.chunks.append(chunk_data)
        
        if not self.chunks:
            logger.warning("No text chunks created.")
            return

        # 3. Embeddings & FAISS
        logger.info("Encoding chunks...")
        # Use contextualized content for embeddings/search if available
        texts_to_embed = [c.get("contextualized_content", c["content"]) for c in self.chunks]
        embeddings = self.embed_model.encode(texts_to_embed, show_progress_bar=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        # 4. BM25
        # Also use contextualized content for BM25
        tokenized_corpus = [c.get("contextualized_content", c["content"]).lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 5. Save cache
        self._save_cache(embeddings, tokenized_corpus)
        
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

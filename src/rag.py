# src/rag.py
import os
import glob
import json
import pickle
import hashlib
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import logger, CACHE_DIR


class InvestmentRAG:
    """
    RAG implementation using Summary Vector (Parent-Document Retrieval) strategy.
    
    - Vector Index (Parent): Stores only summary vectors
    - Doc Store (Child): Stores all detailed chunks in memory (no vectorization)
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.documents = []
        
        # Parent-Document Retrieval structures
        self.doc_summaries = {}      # {doc_id: summary_text}
        self.doc_store = {}          # {doc_id: [chunk_dict1, chunk_dict2, ...]}
        
        self.summary_index = None    # FAISS index for summary vectors only
        self.summary_doc_ids = []    # Mapping: index position -> doc_id
        
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
            "faiss": os.path.join(self._cache_dir, "summary_index.faiss"),
            "doc_store": os.path.join(self._cache_dir, "doc_store.pkl"),
            "doc_summaries": os.path.join(self._cache_dir, "doc_summaries.pkl"),
            "summary_doc_ids": os.path.join(self._cache_dir, "summary_doc_ids.pkl"),
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

    def _save_cache(self, summary_embeddings: np.ndarray):
        """Save all RAG components to cache files."""
        os.makedirs(self._cache_dir, exist_ok=True)
        paths = self._get_cache_paths()
        
        # Save metadata with data hash
        meta = {
            "data_hash": self._compute_data_hash(),
            "num_docs": len(self.doc_store),
            "total_chunks": sum(len(chunks) for chunks in self.doc_store.values()),
        }
        with open(paths["meta"], "w") as f:
            json.dump(meta, f)
        
        # Save FAISS summary index
        faiss.write_index(self.summary_index, paths["faiss"])
        
        # Save doc store (child chunks)
        with open(paths["doc_store"], "wb") as f:
            pickle.dump(self.doc_store, f)
        
        # Save doc summaries (parent)
        with open(paths["doc_summaries"], "wb") as f:
            pickle.dump(self.doc_summaries, f)
        
        # Save summary doc ids mapping
        with open(paths["summary_doc_ids"], "wb") as f:
            pickle.dump(self.summary_doc_ids, f)
        
        logger.info(f"Cache saved to {self._cache_dir}")

    def _load_cache(self) -> bool:
        """Load RAG components from cache. Returns True if successful."""
        try:
            paths = self._get_cache_paths()
            
            # Load doc store
            with open(paths["doc_store"], "rb") as f:
                self.doc_store = pickle.load(f)
            
            # Load doc summaries
            with open(paths["doc_summaries"], "rb") as f:
                self.doc_summaries = pickle.load(f)
            
            # Load summary doc ids
            with open(paths["summary_doc_ids"], "rb") as f:
                self.summary_doc_ids = pickle.load(f)
            
            # Load FAISS summary index
            self.summary_index = faiss.read_index(paths["faiss"])
            
            total_chunks = sum(len(chunks) for chunks in self.doc_store.values())
            logger.info(f"Cache loaded: {len(self.doc_store)} docs, {total_chunks} chunks")
            return True
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return False

    def _generate_summary(self, llm, text: str) -> str:
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
            return summary.replace("\n", " ")
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return ""

    def initialize(self, llm=None, force_rebuild: bool = False):
        """Load data, chunk, and build summary vector index.
        
        Args:
            llm: Optional LLM instance for generating document summaries.
                 If not provided, will use document content directly.
            force_rebuild: If True, ignore cache and rebuild from scratch.
        """
        logger.info("Initializing RAG Knowledge Base (Summary Vector Strategy)...")
        
        # Load models first (always needed for search)
        logger.info("Loading Embedding Model & Re-ranker...")
        self.embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
        self.reranker = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
        
        # Check cache validity
        if not force_rebuild and self._is_cache_valid():
            logger.info("Loading RAG from cache...")
            if self._load_cache():
                self.is_ready = True
                total_chunks = sum(len(c) for c in self.doc_store.values())
                logger.info(f"RAG System Ready (from cache). {len(self.doc_store)} docs, {total_chunks} chunks.")
                return
            logger.warning("Cache loading failed, rebuilding...")
        elif force_rebuild:
            logger.info("Force rebuilding RAG cache...")
        else:
            logger.info("Cache not found or invalid, building from scratch...")
        
        # Build from scratch
        self._build_index(llm)

    def _build_index(self, llm=None):
        """Build RAG index from scratch using Summary Vector strategy."""
        # 1. Load Data
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            logger.warning(f"Data directory '{self.data_dir}' created. Please add .txt files.")
            with open(os.path.join(self.data_dir, "sample_investment.txt"), "w") as f:
                f.write("Value at Risk (VaR) is a statistic that quantifies the extent of possible financial losses within a firm, portfolio, or position over a specific time frame.")
        
        files = glob.glob(os.path.join(self.data_dir, "*.txt"))
        if not files:
            logger.warning("No .txt files found in data directory.")
            return

        # 2. Process each document
        logger.info("Processing documents...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    filename = os.path.basename(file_path)
                    doc_id = filename  # Use filename as doc_id
                    
                    self.documents.append({"doc_id": doc_id, "content": content})
                    
                    # Generate summary (Parent)
                    if llm:
                        logger.info(f"Generating summary for {filename}...")
                        summary = self._generate_summary(llm, content)
                    else:
                        # Fallback: use first 500 chars as summary
                        summary = content[:500] if len(content) > 500 else content
                    
                    self.doc_summaries[doc_id] = summary
                    
                    # Create chunks (Children) - stored in doc_store, NOT vectorized
                    splits = text_splitter.split_text(content)
                    chunks = []
                    for i, text in enumerate(splits):
                        chunks.append({
                            "id": f"{doc_id}_{i}",
                            "content": text,
                            "doc_id": doc_id,
                        })
                    self.doc_store[doc_id] = chunks
                    
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

        if not self.doc_summaries:
            logger.warning("No documents processed.")
            return

        # 3. Create Summary Vector Index (Parent Index)
        logger.info("Encoding document summaries for vector index...")
        self.summary_doc_ids = list(self.doc_summaries.keys())
        summary_texts = [self.doc_summaries[doc_id] for doc_id in self.summary_doc_ids]
        
        summary_embeddings = self.embed_model.encode(summary_texts, show_progress_bar=True)
        
        dimension = summary_embeddings.shape[1]
        self.summary_index = faiss.IndexFlatL2(dimension)
        self.summary_index.add(np.array(summary_embeddings).astype('float32'))
        
        # 4. Save cache
        self._save_cache(summary_embeddings)
        
        total_chunks = sum(len(chunks) for chunks in self.doc_store.values())
        self.is_ready = True
        logger.info(f"RAG System Ready. Indexed {len(self.doc_summaries)} doc summaries, {total_chunks} chunks in store.")

    def add_document(self, doc_id: str, content: str, llm=None):
        """
        Add a single document to the RAG system.
        
        Args:
            doc_id: Unique identifier for the document
            content: Document text content
            llm: Optional LLM for generating summary
        """
        if not self.embed_model:
            logger.error("RAG not initialized. Call initialize() first.")
            return False
        
        # Generate summary
        if llm:
            summary = self._generate_summary(llm, content)
        else:
            summary = content[:500] if len(content) > 500 else content
        
        self.doc_summaries[doc_id] = summary
        
        # Create chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_text(content)
        chunks = []
        for i, text in enumerate(splits):
            chunks.append({
                "id": f"{doc_id}_{i}",
                "content": text,
                "doc_id": doc_id,
            })
        self.doc_store[doc_id] = chunks
        
        # Add to summary index
        summary_embedding = self.embed_model.encode([summary])
        self.summary_index.add(np.array(summary_embedding).astype('float32'))
        self.summary_doc_ids.append(doc_id)
        
        logger.info(f"Added document '{doc_id}' with {len(chunks)} chunks.")
        return True

    def search(self, query: str, k: int = 3, k_docs: int = 2) -> str:
        """
        Search using Parent-Document Retrieval strategy.
        
        1. Search summary vectors to find relevant parent documents
        2. Retrieve ALL chunks from those documents
        3. Re-rank chunks and return top-k
        
        Args:
            query: Search query
            k: Number of final chunks to return
            k_docs: Number of parent documents to retrieve first
        """
        if not self.is_ready:
            return "Knowledge base not initialized or empty."
        
        # 1. Encode query
        query_vec = self.embed_model.encode([query])
        
        # 2. Search in summary index to find relevant parent documents
        num_docs = min(k_docs, len(self.summary_doc_ids))
        D, I = self.summary_index.search(np.array(query_vec).astype('float32'), num_docs)
        
        relevant_doc_ids = []
        for idx in I[0]:
            if 0 <= idx < len(self.summary_doc_ids):
                relevant_doc_ids.append(self.summary_doc_ids[idx])
        
        if not relevant_doc_ids:
            return "No relevant documents found."
        
        logger.info(f"Found {len(relevant_doc_ids)} relevant documents: {relevant_doc_ids}")
        
        # 3. Gather ALL chunks from relevant parent documents
        candidate_chunks = []
        for doc_id in relevant_doc_ids:
            if doc_id in self.doc_store:
                candidate_chunks.extend(self.doc_store[doc_id])
        
        if not candidate_chunks:
            return "No chunks found in relevant documents."
        
        # 4. Re-rank with CrossEncoder
        logger.info(f"Re-ranking {len(candidate_chunks)} candidate chunks...")
        pairs = [[query, chunk["content"]] for chunk in candidate_chunks]
        scores = self.reranker.predict(pairs)
        
        # 5. Sort and select top-k chunks
        ranked_results = sorted(zip(candidate_chunks, scores), key=lambda x: x[1], reverse=True)
        results = [r[0] for r in ranked_results[:k]]
        
        # Format output
        context = "\n\n".join([f"[Source: {r['doc_id']}]\n{r['content']}" for r in results])
        return context

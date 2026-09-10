"""
vectorstore.py — Vector Store management using FAISS and Sentence Transformers.

Handles text chunking, dense vector embedding generation, disk persistence,
and sub-millisecond semantic similarity search.
"""

import os
# Prevent OpenMP multiple runtime conflict warnings on Windows/Linux
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the FAISS vector store manager.

        Args:
            persist_dir: Directory where FAISS index and metadata pickle are saved.
            embedding_model: HuggingFace sentence-transformers model identifier.
        """
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index: Optional[faiss.IndexFlatL2] = None
        self.metadata: List[Dict[str, Any]] = []
        self._model_name = embedding_model
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load embedding model to save memory until required."""
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def build_from_documents(
        self,
        documents: List[Any],
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        batch_size: int = 64
    ):
        """
        Chunk documents, generate dense embeddings, and build the FAISS index.

        Args:
            documents: List of LangChain Document objects.
            chunk_size: Maximum character length per chunk.
            chunk_overlap: Overlapping character count between consecutive chunks.
            batch_size: Batch size for sentence-transformer encoding.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks")

        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype("float32")
        print(f"[INFO] Generated embeddings shape: {embeddings.shape}")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        self.metadata = [
            {"text": chunk.page_content, **chunk.metadata}
            for chunk in chunks
        ]

        self._save()
        print(f"[INFO] Successfully saved {self.index.ntotal} vectors to {self.persist_dir}/")

    def load(self):
        """Load an existing FAISS index and metadata from disk."""
        index_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Index files missing in '{self.persist_dir}'. Build the database first."
            )

        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded {self.index.ntotal} vectors from {self.persist_dir}/")

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search for a user query.

        Args:
            query_text: Search query string.
            top_k: Number of most similar chunks to return.

        Returns:
            List of result dicts containing distance, score, and metadata.
        """
        if self.index is None:
            if self.is_built():
                self.load()
            else:
                raise ValueError("Vector store index has not been loaded or built.")

        query_emb = self.model.encode([query_text], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(query_emb, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append({
                    "distance": float(dist),
                    "score": float(1.0 / (1.0 + float(dist))),
                    "metadata": self.metadata[idx]
                })
        return results

    def is_built(self) -> bool:
        """Check if pre-built vector index and metadata exist on disk."""
        return (
            os.path.exists(os.path.join(self.persist_dir, "faiss.index"))
            and os.path.exists(os.path.join(self.persist_dir, "metadata.pkl"))
        )

    def _save(self):
        """Internal helper to serialize index and metadata."""
        if self.index is None:
            raise ValueError("No index to save.")
        faiss.write_index(self.index, os.path.join(self.persist_dir, "faiss.index"))
        with open(os.path.join(self.persist_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

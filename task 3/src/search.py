"""
search.py — RAG generation engine for Pico-8 Lua code.

Combines FAISS vector retrieval with Groq high-speed LLM inference
using a tailored fantasy console system prompt and context conditioning.
"""

import os
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

PICO8_SYSTEM_PROMPT = """You are an expert Pico-8 fantasy console developer.
Pico-8 uses a subset of Lua with built-in functions:
- Game loop: _init(), _update(), _draw()
- Drawing: spr(), print(), pset(), line(), rect(), rectfill(), circ(), circfill(), map(), cls()
- Input: btn(), btnp()
- Audio: sfx(), music()
- Map/Sprites: mget(), mset(), fget(), fset()
- Math: flr(), ceil(), mid(), cos(), sin(), rnd(), abs(), sgn()

Technical Constraints:
- 8192 code tokens limit
- 128x128 screen resolution
- 16 color palette (0-15)
- 256 8x8 sprites

Use the retrieved game examples as reference to write authentic, bug-free, and well-structured Pico-8 Lua code."""


class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        llm_model: Optional[str] = None,
        vector_store: Optional[FaissVectorStore] = None,
    ):
        """
        Initialize the RAG generation pipeline.

        Args:
            persist_dir: Path to FAISS storage directory.
            llm_model: Model name for Groq inference. If omitted, checks GROQ_MODEL env var,
                       defaulting to 'llama-3.3-70b-versatile' or 'qwen/qwen3.8-27b'.
            vector_store: Optional pre-loaded FaissVectorStore instance to avoid reloading.
        """
        if vector_store is not None:
            self.store = vector_store
        else:
            self.store = FaissVectorStore(persist_dir)
            if not self.store.is_built():
                raise FileNotFoundError(
                    f"FAISS index not found at '{persist_dir}'. Build it first with: python build_rag.py --build"
                )
            self.store.load()

        model_name = llm_model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.model_name = model_name

        api_key = os.getenv("GROQ_API_KEY", "")
        self.api_key = api_key
        self._llm: Optional[ChatGroq] = None

    @property
    def llm(self) -> ChatGroq:
        """Lazy load the Groq client upon first query."""
        if self._llm is None:
            api_key = self.api_key or os.getenv("GROQ_API_KEY", "")
            if not api_key:
                raise ValueError("GROQ_API_KEY is not set. Add it to your .env file or sidebar.")
            self._llm = ChatGroq(
                groq_api_key=api_key,
                model_name=self.model_name,
                temperature=0.3,
                max_tokens=1000,
            )
        return self._llm

    def _build_context(self, results: List[Dict[str, Any]], max_chars: int = 5000) -> str:
        """Extract and format reference code snippets from retrieved chunks."""
        MAX_CODE_CHARS = 1500
        parts = []
        for r in results:
            meta = r.get("metadata", {})
            name = meta.get("game_name", "Unknown")
            author = meta.get("author", "Unknown")
            desc = meta.get("description", "")[:200]
            code = meta.get("game_code", meta.get("text", ""))[:MAX_CODE_CHARS]
            parts.append(
                f"--- Game: {name} (by {author}) ---\n"
                f"Description: {desc}\n"
                f"Code Sample:\n{code}"
            )
        return "\n\n".join(parts)[:max_chars]

    def query_with_sources(
        self,
        query: str,
        top_k: int = 3,
        retries: int = 3
    ) -> Dict[str, Any]:
        """
        Execute RAG retrieval and code generation, returning both answer and sources.

        Args:
            query: User's coding prompt or question.
            top_k: Number of reference games to retrieve.
            retries: Number of retry attempts on rate limit.

        Returns:
            Dict containing 'response' (str) and 'sources' (List[Dict]).
        """
        results = self.store.query(query, top_k=top_k)
        if not results:
            return {
                "response": "No relevant Pico-8 games found in the database.",
                "sources": []
            }

        sources = []
        for r in results:
            meta = r.get("metadata", {})
            sources.append({
                "game_name": meta.get("game_name", "Unknown"),
                "author": meta.get("author", "Unknown"),
                "like_count": meta.get("like_count", "0"),
                "description": meta.get("description", ""),
                "snippet": meta.get("text", "")[:400],
                "score": r.get("score", 0.0),
                "source_url": meta.get("source_url", ""),
            })

        context = self._build_context(results)
        prompt = (
            f"{PICO8_SYSTEM_PROMPT}\n\n"
            f"--- Retrieved Pico-8 Reference Games ---\n{context}\n\n"
            f"--- User Request ---\n{query}\n\n"
            f"Instructions: Write production-ready, clean Pico-8 Lua code that fulfills the request. "
            f"Include concise comments explaining the logic and Pico-8 specific APIs."
        )

        response_text = ""
        for attempt in range(retries):
            try:
                response = self.llm.invoke([prompt])
                response_text = response.content
                break
            except Exception as e:
                err_msg = str(e).lower()
                if "rate_limit" in err_msg or "429" in err_msg:
                    wait = 20 * (attempt + 1)
                    print(f"[WARN] Rate limited. Retrying in {wait}s ({attempt + 1}/{retries})...")
                    time.sleep(wait)
                else:
                    raise e
        else:
            response_text = "Rate limit reached on Groq API. Please wait a moment and try again."

        return {
            "response": response_text,
            "sources": sources
        }

    def query(self, query: str, top_k: int = 3, retries: int = 3) -> str:
        """Backward-compatible query method returning generated string directly."""
        result = self.query_with_sources(query, top_k=top_k, retries=retries)
        return result["response"]

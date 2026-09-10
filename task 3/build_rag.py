"""
build_rag.py — Command-line interface for building and querying the Pico-8 RAG database.

Usage Examples:
  # Build index from CSV
  python build_rag.py --build --csv data/games.csv

  # Query RAG database for Pico-8 code generation
  python build_rag.py --query "Create a simple platformer game with gravity"

  # Search and display retrieved reference chunks only (no LLM call required)
  python build_rag.py --retrieve-only --query "particle explosion"
"""

import os
# Suppress OpenMP duplicate runtime warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def build(csv_path: str, persist_dir: str = "faiss_store"):
    from src.data_loader import load_pico8_csv
    from src.vectorstore import FaissVectorStore

    print(f"\n[1/2] Loading games from: {csv_path}")
    docs = load_pico8_csv(csv_path)

    print(f"[2/2] Building FAISS index in: {persist_dir}/")
    store = FaissVectorStore(persist_dir)
    store.build_from_documents(docs)
    print(f"\n✅ Indexing complete! {store.index.ntotal} vectors indexed successfully.\n")


def query(query_text: str, top_k: int = 3, persist_dir: str = "faiss_store", model_name: str = None):
    from src.search import RAGSearch

    print(f"\nQuery: {query_text}")
    print("=" * 60)
    rag = RAGSearch(persist_dir=persist_dir, llm_model=model_name)
    result = rag.query_with_sources(query_text, top_k=top_k)

    print("\n🎮 Generated Pico-8 Code:\n")
    print(result["response"])
    print("\n" + "=" * 60)
    print(f"📚 Retrieved {len(result['sources'])} Reference Games:")
    for i, s in enumerate(result["sources"], 1):
        print(f"  [{i}] {s['game_name']} (by {s['author']}) - Score: {s['score']:.3f} | Likes: {s['like_count']}")
    print()


def retrieve_only(query_text: str, top_k: int = 5, persist_dir: str = "faiss_store"):
    from src.vectorstore import FaissVectorStore

    store = FaissVectorStore(persist_dir)
    if not store.is_built():
        print(f"Error: Vector store not found in '{persist_dir}'. Build it first.")
        sys.exit(1)
    store.load()

    print(f"\nSemantic Retrieval for: '{query_text}' (top {top_k})")
    print("=" * 60)
    results = store.query(query_text, top_k=top_k)
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        print(f"\n[{i}] {meta.get('game_name', 'Unknown')} (Score: {r['score']:.3f}, Distance: {r['distance']:.2f})")
        print(f"    Author: {meta.get('author', 'Unknown')} | Likes: {meta.get('like_count', '0')}")
        snippet = meta.get('text', '')[:200].replace('\n', ' ')
        print(f"    Excerpt: {snippet}...")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pico-8 RAG Database CLI")
    parser.add_argument("--build", action="store_true", help="Build the FAISS index from CSV")
    parser.add_argument("--csv", default="data/games.csv", help="Path to games CSV (default: data/games.csv)")
    parser.add_argument("--query", type=str, help="Query the RAG database to generate Pico-8 Lua code")
    parser.add_argument("--retrieve-only", action="store_true", help="Retrieve reference chunks without calling LLM")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved chunks (default: 3)")
    parser.add_argument("--persist-dir", default="faiss_store", help="FAISS index directory (default: faiss_store)")
    parser.add_argument("--model", type=str, default=None, help="Groq model name to use for generation")
    args = parser.parse_args()

    if not args.build and not args.query:
        parser.print_help()
        sys.exit(0)

    if args.build:
        build(args.csv, args.persist_dir)

    if args.query:
        if args.retrieve_only:
            retrieve_only(args.query, args.top_k, args.persist_dir)
        else:
            query(args.query, args.top_k, args.persist_dir, args.model)

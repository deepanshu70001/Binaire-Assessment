"""
data_loader.py — Loads Pico-8 game dataset from CSV into LangChain Document objects.

Parses metadata (game name, author, likes, description, source URL, comments)
and raw Lua code, creating structured documents ready for chunking and vector indexing.
"""

import csv
from pathlib import Path
from typing import List, Optional
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


def load_pico8_csv(csv_path: str) -> List[Document]:
    """
    Load the Pico-8 games CSV into LangChain Documents.
    Each valid row becomes one Document with structured page_content and rich metadata.

    Args:
        csv_path: Path to the CSV file containing game data.

    Returns:
        List of LangChain Document objects.
    """
    csv_file = Path(csv_path).resolve()
    if not csv_file.exists():
        # Fallback to case-insensitive search if path differs by case (e.g., Data vs data)
        alt = Path(str(csv_path).lower())
        if alt.exists():
            csv_file = alt.resolve()
        else:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

    documents: List[Document] = []

    with open(csv_file, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [col.strip().lower().replace(" ", "_") for col in reader.fieldnames]

        for row_idx, row in enumerate(reader):
            name = (row.get("game_name") or "").strip()
            author = (row.get("author") or "").strip()
            code = (row.get("game_code") or "").strip()
            license_info = (row.get("license") or "").strip()
            like_count = (row.get("like_count") or "0").strip()
            description = (row.get("description") or "").strip()
            source_url = (row.get("source_url") or "").strip()

            if not name and not code:
                continue

            # Combine up to 5 community comments if present
            comment_lines = []
            for i in range(1, 6):
                val = (row.get(f"comment_{i}") or "").strip()
                if val:
                    comment_lines.append(f"Comment {i}: {val}")
            comments = "\n".join(comment_lines)

            page_content = (
                f"=== Pico-8 Game: {name} ===\n"
                f"Author: {author}\n"
                f"Description: {description}\n"
                f"License: {license_info or 'Not specified'}\n"
                f"Likes: {like_count}\n\n"
                f"--- Game Code (Pico-8 Lua) ---\n{code}\n\n"
                f"--- Community Comments ---\n{comments}\n"
            )

            doc = Document(
                page_content=page_content,
                metadata={
                    "row_index": row_idx,
                    "game_name": name,
                    "author": author,
                    "game_code": code,
                    "license": license_info,
                    "like_count": like_count,
                    "description": description,
                    "comments": comments,
                    "source_url": source_url,
                    "artwork_url": (row.get("artwork_url") or "").strip(),
                },
            )
            documents.append(doc)

    print(f"[INFO] Loaded {len(documents)} Pico-8 games from {csv_file.name}")
    return documents
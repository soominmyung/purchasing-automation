"""
Store each history document whole. Documents ingested before 2026-10-19 were split into
1,000-character chunks overlapping by 200; history is retrieved by metadata, never by
similarity, so the chunks served no purpose, and joining them back repeated the overlaps.

Each chunked document is joined with the overlaps removed, stored again as one document with
the same metadata, and its chunks deleted. Run once:
     python scripts/unchunk_history_store.py
"""
import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from services.vector_store import HISTORY_COLLECTION, _add_docs, _get_or_create_store

MAX_OVERLAP = 300  # the splitter used an overlap of 200 characters


def join(chunks: list[str]) -> str:
    """The splitter trims whitespace at chunk edges, so overlaps are found by locating the start
    of the next chunk inside the end of the text so far, not by an exact suffix match."""
    text = chunks[0]
    for nxt in chunks[1:]:
        at = text.find(nxt[:50], max(0, len(text) - MAX_OVERLAP))
        text = text[:at] + nxt if at != -1 else text + "\n" + nxt
    return text


def main():
    store = _get_or_create_store(HISTORY_COLLECTION)
    raw = store._collection.get(include=["documents", "metadatas"])
    groups: dict[tuple, list] = defaultdict(list)
    for cid, text, meta in zip(raw["ids"], raw["documents"], raw["metadatas"]):
        groups[(meta.get("source_id") or meta.get("source"), meta.get("ingested_at"))].append((cid, text, meta))
    for key, parts in groups.items():
        if len(parts) == 1:
            continue
        whole = join([t for _, t, _ in parts])  # Chroma returns chunks in insertion order
        _add_docs(HISTORY_COLLECTION, [Document(page_content=whole, metadata=parts[0][2])], split=False)
        store._collection.delete(ids=[cid for cid, _, _ in parts])
        print(f"  {key[0]}: {len(parts)} chunks ({sum(len(t) for _, t, _ in parts)} chars) -> 1 document ({len(whole)} chars)")
    print("done")


if __name__ == "__main__":
    main()

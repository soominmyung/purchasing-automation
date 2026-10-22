"""
Copy documents from the earlier supplier_history / item_history collections into the single
history collection, where what a document is about comes from its header.

Chunks of one earlier document are joined back together (they share the supplier name or
item code they were filed under) and stored again as one document, so its header is read
once. Where the header does not name the supplier or item the document was filed under,
that is added, so nothing that used to be found is lost. The earlier collections are left
as they are.

Run once: python scripts/migrate_history_store.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from scripts.unchunk_history_store import join
from services.vector_store import (
    HISTORY_COLLECTION, _add_docs, _get_or_create_store, history_metadata, parse_history_header,
)

MIGRATED = "(migrated from the earlier store; file name not recorded)"


def main():
    target = _get_or_create_store(HISTORY_COLLECTION)
    if target._collection.get(where={"source": MIGRATED})["ids"]:
        raise SystemExit("Already migrated; nothing done.")
    for collection, key in (("supplier_history", "supplier_name"), ("item_history", "item_code")):
        raw = _get_or_create_store(collection)._collection.get(include=["documents", "metadatas"])
        groups: dict[str, list[str]] = {}
        for text, meta in zip(raw["documents"], raw["metadatas"]):
            groups.setdefault(meta.get(key) or "", []).append(text)
        for filed_under, chunks in groups.items():
            text = join(chunks)  # the earlier chunks overlap; join without repeating the overlaps
            header = parse_history_header(text)
            if collection == "supplier_history" and filed_under and not header["supplier"]:
                header["supplier"] = filed_under
            if collection == "item_history" and filed_under and filed_under not in header["item_codes"]:
                header["item_codes"].append(filed_under)
            _add_docs(HISTORY_COLLECTION, [Document(page_content=text, metadata=history_metadata(header, MIGRATED))],
                      split=False)
            print(f"  {collection:<17} {filed_under:<10} -> supplier={header['supplier'] or '-'} "
                  f"items={header['items'] or '-'} codes={header['item_codes'] or '-'} date={header['date'] or '-'}")


if __name__ == "__main__":
    main()

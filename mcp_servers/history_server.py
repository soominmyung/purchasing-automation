"""
MCP server exposing the project's purchasing-history retrieval.

The pipeline already looks this up before the analysis agent runs, but only as part of a
full run. Exposing it over MCP lets any client query the history on its own — useful
when someone wants to check one supplier's record without generating a whole report.

Retrieval is the pipeline's own (services/vector_store.search_history): one history store,
where each document says in its header which supplier and items it is about; documents are
matched on those and returned most recent first by the document's own date. No similarity
ranking is involved.

Run: python mcp_servers/history_server.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # config reads .env from the working directory; an MCP client starts us anywhere

from mcp.server.mcpserver import MCPServer

from services.vector_store import HISTORY_COLLECTION, _get_or_create_store, search_history

mcp = MCPServer(
    name="purchasing-history",
    instructions=(
        "Purchasing history for purchasing decisions. Each record says which supplier and "
        "items it is about: supplier-wide (e.g. a strike), a supplier's particular items "
        "(e.g. a defective lot), or items whoever supplies them (e.g. a demand spike). Look "
        "up a supplier before raising a purchase order. Browse the suppliers:// and items:// "
        "resources first if you do not know the exact name."
    ),
)


def _render(docs: list[dict], empty: str) -> str:
    if not docs:
        return empty
    parts = []
    for d in docs:
        items = ", ".join(d["items"] or d["item_codes"])
        about = (f"{d['supplier']} — {items}" if d["supplier"] and items else
                 f"{d['supplier']} — all items" if d["supplier"] else f"{items} — any supplier")
        parts.append(f"[{about}; {d['date'] or 'undated'}; {d['source'] or 'file name not recorded'}]\n{d['text']}")
    return "\n\n---\n\n".join(parts)


@mcp.tool(
    description=(
        "Look up past records about ONE supplier and, optionally, items bought from it: "
        "delivery delays, price changes, quality incidents, stock-outs, demand spikes. "
        "Returns records from the 12 months before `as_of` (YYYY-MM-DD, default today; the pipeline "
        "uses the stock snapshot date): up to `limit` about the supplier and up to `limit` about "
        "each item, most recent first."
    )
)
def purchasing_history(supplier_name: str = "", item_names: list[str] | None = None,
                       item_codes: list[str] | None = None, as_of: str = "", limit: int = 5) -> str:
    items = [("", n) for n in (item_names or [])] + [(c, "") for c in (item_codes or [])]
    if not supplier_name and not items:
        return "Give a supplier name, item names or item codes."
    from datetime import date
    from config import settings
    docs, _ = search_history(supplier_name, items, as_of=as_of or date.today().isoformat(),
                             months=settings.history_window_months, per_scope=limit)
    return _render(docs, "No history on record for that supplier or those items.")


def _distinct(field: str) -> list[str]:
    store = _get_or_create_store(HISTORY_COLLECTION)
    if store is None:  # no OpenAI key: the store cannot be opened
        return []
    raw = store._collection.get(include=["metadatas"])
    values = set()
    for m in raw.get("metadatas") or []:
        values.update(v.strip() for v in str((m or {}).get(field) or "").split(",") if v.strip())
    return sorted(values)


@mcp.resource(
    "suppliers://list",
    description="Suppliers named in history documents on file.",
    mime_type="text/plain",
)
def suppliers_list() -> str:
    names = _distinct("supplier_name")
    return "\n".join(names) if names else "(no history ingested yet)"


@mcp.resource(
    "items://list",
    description="Item names and codes named in history documents on file.",
    mime_type="text/plain",
)
def items_list() -> str:
    values = _distinct("items") + _distinct("item_codes")
    return "\n".join(values) if values else "(no history ingested yet)"


if __name__ == "__main__":
    mcp.run()

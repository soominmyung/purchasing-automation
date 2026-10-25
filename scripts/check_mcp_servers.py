"""
Start each MCP server as an MCP client would (stdio), list what it offers and call every tool
and resource once. The calls use the dev cases' figures, so the answers can be checked against
the pipeline's own runs of those cases.

Run: python scripts/check_mcp_servers.py
Out: dev/runs/<date_time>_mcp-servers-test/record.json
"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "dev" / "cases"

TOOL_CALLS = {
    "history_server": [
        # the store on this PC holds three records migrated from the earlier store: one about
        # SupplierA (undated), one about item 100004 ItemE (2024-11-28), one about item 100000
        ("purchasing_history", {"supplier_name": "SupplierA", "item_codes": ["100004", "100000"],
                                "as_of": "2025-04-05", "limit": 3}),
        ("purchasing_history", {"item_names": ["ItemE"], "as_of": "2026-10-25"}),
        ("purchasing_history", {"supplier_name": "NoSuchSupplier", "as_of": "2025-04-05"}),
        ("purchasing_history", {}),
    ],
    "replenishment_server": [
        # delayvale_low_becomes_medium: BushingRing is Low at 18 weeks, Medium at 22
        ("item_figures", {"snapshot_date": "2025-04-05", "item_code": "550001", "item_name": "BushingRing",
                          "available_stock": 3600, "weeks_to_out_of_stock": 36}),
        ("item_figures", {"snapshot_date": "2025-04-05", "item_code": "550001", "item_name": "BushingRing",
                          "available_stock": 3600, "weeks_to_out_of_stock": 36, "lead_time_weeks": 22}),
        # an order date already passed at the longer lead time
        ("item_figures", {"snapshot_date": "2025-04-05", "item_code": "X1", "available_stock": 200,
                          "weeks_to_out_of_stock": 20, "lead_time_weeks": 24}),
        ("stock_csv_figures", {"csv_text": json.loads((CASES / "lowfield_all_low.json").read_text(encoding="utf-8"))["csv"]}),
        ("replenishment_policy", {}),
    ],
}


def _content(result) -> object:
    texts = [getattr(c, "text", str(c)) for c in result.content]
    if getattr(result, "structured_content", None) is not None:
        return result.structured_content
    return texts[0] if len(texts) == 1 else texts


async def test_server(name: str) -> dict:
    # The client passes only a minimal environment unless told otherwise; the history store needs
    # OPENAI_API_KEY (its embeddings), as a desktop client's "env" entry would give it.
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp_servers" / f"{name}.py")],
                                   env=dict(os.environ))
    out: dict = {"server": name}
    started = time.perf_counter()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            out["server_info"] = {"name": init.server_info.name, "instructions": init.instructions}
            out["tools"] = [t.name for t in (await session.list_tools()).tools]
            out["resources"] = [str(r.uri) for r in (await session.list_resources()).resources]
            out["prompts"] = [p.name for p in (await session.list_prompts()).prompts]
            out["calls"] = []
            for tool, args in TOOL_CALLS[name]:
                t0 = time.perf_counter()
                result = await session.call_tool(tool, args)
                out["calls"].append({"tool": tool, "args": {k: (v if k != "csv_text" else "<lowfield_all_low csv>")
                                                            for k, v in args.items()},
                                     "is_error": bool(result.is_error), "seconds": round(time.perf_counter() - t0, 2),
                                     "result": _content(result)})
            for uri in out["resources"]:
                res = await session.read_resource(uri)
                out["calls"].append({"resource": uri, "result": [getattr(c, "text", "") for c in res.contents]})
    out["seconds"] = round(time.perf_counter() - started, 1)
    return out


async def main() -> None:
    record = {"label": "MCP servers test", "started": datetime.now().isoformat(timespec="seconds"),
              "servers": [await test_server(n) for n in TOOL_CALLS]}
    folder = ROOT / "dev" / "runs" / f"{datetime.now():%Y-%m-%d_%H%M%S}_mcp-servers-test"
    folder.mkdir(parents=True)
    (folder / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2, default=str))
    print("record:", folder)


if __name__ == "__main__":
    asyncio.run(main())

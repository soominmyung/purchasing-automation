# MCP servers: built, tested, not used by the pipeline

Two MCP servers in `mcp_servers/` give any MCP client (Claude Desktop, an IDE, an agent framework)
the two things the pipeline works out in code before the analysis agent runs: the purchasing
history for a supplier and its items, and the stock figures. They call the pipeline's own
functions, so a client gets the same answers a run would use. The pipeline itself does not call
them. Built and tested on 2026-10-25 (`mcp` 2.2.0, stdio).

## What they offer

| Server | Offers | Calls in the pipeline's code |
|---|---|---|
| `purchasing-history` | tool `purchasing_history(supplier_name, item_names, item_codes, as_of, limit)`; resources `suppliers://list`, `items://list` | `services/vector_store.search_history`: records naming the supplier or the items, from the 12 months before `as_of` (the pipeline uses the snapshot date), `limit` per scope, newest first |
| `purchasing-replenishment` | tools `item_figures(snapshot_date, available_stock, weeks_to_out_of_stock, item_code, item_name, lead_time_weeks)`, `stock_csv_figures(csv_text)`, `replenishment_policy()` | `services/item_grouping.build_item` and `group_by_supplier_and_recommend` (the 18-week figures), `services/documents.items_after_changes` and `with_lead_times` (risk level and latest order date at another lead time, with the snapshot-date floor) |

No MCP prompt is published. An earlier draft of the replenishment server published the analysis
agent's system prompt and told clients to "use these tools rather than working the dates or
quantities out yourself"; both were removed. The v5 analysis prompt is written for the pipeline's
input (figures already worked out, history already numbered), not for a conversation, and in v5
the model never works out a date or a quantity at all, so there is nothing to steer it away from.

## The test

`python scripts/check_mcp_servers.py` starts each server as a client would (stdio), lists what it
offers and calls every tool and resource once. Record: `dev/runs/2026-10-25_160908_mcp-servers-test/record.json`.

| Call | Result | Checked against |
|---|---|---|
| `item_figures`, BushingRing, 3,600 units, 36 weeks | Low; delivery by 2025-12-13; order by 2025-08-09 ("within 18 weeks"); 2,600 units | the 18-week figures of `dev/cases/delayvale_low_becomes_medium.json` |
| the same at `lead_time_weeks: 22` | Medium; order by 2025-07-12 ("within 14 weeks") | the pipeline's run of that case, where the analysis set 22 weeks and BushingRing went Low → Medium |
| `item_figures`, 20 weeks of stock, lead time 24 weeks | Critical; order date 2025-04-05 (the snapshot date), "overdue", `po_should_have` 2025-03-08; 4 weeks out of stock | delivery 2025-08-23 − 24 weeks = 2025-03-08, before the snapshot, so the floor applies as in `with_lead_times` |
| `stock_csv_figures`, the `lowfield_all_low` CSV | LowfieldSupplies, Low; FlangeGasket and ShaftCollar Low, order dates 2025-09-06 and 2025-11-29 | the pipeline's run of that case (report only, every item Low) |
| `replenishment_policy` | 18 weeks (16 import + 2 internal); Critical below the lead time, High below +8, Medium below +17; 26 weeks of cover | `services/item_grouping.py` |
| `purchasing_history`, SupplierA and items 100004, 100000, as of 2025-04-05 | the ItemE safety incident report of 2024-11-28 | the history store on this PC (three records migrated from the earlier store; the other two are undated and so outside any window, as in a run) |
| `purchasing_history`, ItemE, as of 2026-10-25 | no history: the report is more than 12 months old | the 12-month window |
| `purchasing_history` with no supplier or item | "Give a supplier name, item names or item codes." | |
| `suppliers://list`, `items://list` | SupplierA; ItemE, 100000, 100004 | |

Each server started and answered in about 5 s; the calls themselves took under 1.4 s.

One thing the test found: an MCP client passes the server only a minimal environment unless it is
told otherwise, so the history server could not open the store (it needs `OPENAI_API_KEY`, which
Chroma uses for embeddings) and every history call failed. The test now passes the environment,
and `mcp_servers/claude_desktop_config.example.json` has an `env` entry for the key. The servers
also change to the project folder on start, because `config.py` reads `.env` from the working
directory and a client may start them anywhere.

## Why the pipeline does not use them

1. **The pipeline must look everything up every time, the same way.** In a run, code retrieves the
   history (12 months, 5 records per scope) and works out every figure before the analysis agent is
   called, and the agent receives both in its input. Given tools instead, the model would decide
   whether to look up the history, for which items and for which dates. An analysis that skipped the
   lookup for one item would miss, for example, the ItemE safety incident above, and nothing would
   show it. Fixed input is also what the evaluation rests on: checks C1–C10 and the judge compare the
   analysis with the input it was given, which would differ from run to run.
2. **v5 took the arithmetic away from the model altogether.** The replenishment tools would hand the
   model the dates and risk levels to quote, but in v5 the model judges one figure, the lead time, and
   code recalculates the risk level and order date from it after the analysis (changelog item 21).
   A tool call in the middle of the analysis would put back the step that was removed.
3. **The people who would use it are not MCP users.** The buyers work from the documents the pipeline
   writes (the analysis report, the purchase request, the supplier email); they do not use Claude
   Desktop or an IDE. A question such as "what do we have on SupplierA?" is answered by the analysis
   report's findings, with the documents numbered.
4. **The company's data would leave by another route.** The stock and history live on the company's
   server, next to SAP; the pipeline sends them only to the models it calls, with LangSmith tracing
   set to hide content where real data runs (`docs/observability.md`). A desktop MCP client sends
   whatever a tool returns to the client's own model, outside those settings.
5. **It adds nothing the pipeline does not already do.** Both servers are thin wrappers over the
   pipeline's functions; their only use is a one-off lookup outside a run.

Where MCP would pay: if the analysis became an open-ended agent that chooses what to look up (for
example, querying SAP Business One for the last purchase price, which the purchase request lacks
today), or if the same tools were wanted by several assistants in the company. Neither is the case now.

## Run

```
python mcp_servers/history_server.py         # stdio; needs OPENAI_API_KEY for the history store
python mcp_servers/replenishment_server.py   # stdio; no keys
python scripts/check_mcp_servers.py           # the test above
```

For Claude Desktop, copy `mcp_servers/claude_desktop_config.example.json` into its config with the
real paths and key.

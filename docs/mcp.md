# MCP servers: built, tested, not used by the pipeline

**In short.** Two MCP servers were built for the v5 pipeline: one gives the purchasing history, the
other the stock figures (risk level, order dates, quantity), each by calling the pipeline's own code.
Both were tested once with an MCP client, and every answer matched the pipeline's. The pipeline does
not use them. In a run, code looks up the history for the supplier and every item and works out every
figure before the model is called, and in the right order. With tools, the model would decide
what to look up and when to work a figure out, and that is where it goes wrong. Built and tested on
2026-10-25 (`mcp` 2.2.0, stdio transport).

## 1. What MCP is, in this project's terms

MCP (Model Context Protocol) is a standard way to give a model tools. There are three parts:

- **Server:** a small program that offers tools (functions the model may call), resources (data it may
  read) and prompts. Here, `mcp_servers/history_server.py` and `mcp_servers/replenishment_server.py`.
- **Client:** the application the person talks to, such as Claude Desktop, an IDE or an agent
  framework. The client starts each server as a child process and talks to it over stdin/stdout
  (JSON-RPC messages). At start-up it asks the server which tools it has, and gets back each tool's name,
  description and parameters.
- **Model:** the client gives the model the tool list with the person's question. The model decides
  whether to call a tool, which one and with what arguments. The client runs the call on the server
  and sends the result back to the model. The model may then call another tool or answer.

The key point for this project is the last one: **with MCP, the model decides what is looked up
and what is calculated.** In the pipeline, code decides.

## 2. How the servers are designed

### Why these two

The analysis agent's input has two parts that code prepares before the model is called (`services/agents.py`):

1. **The history.** `search_history` returns the records about the supplier and about each item:
   - only records from the 12 months before the snapshot date;
   - at most 5 per scope (the supplier, and each item);
   - newest first.
2. **The figures.** For each item, the code works out:
   - the risk level;
   - the latest delivery date (when stock runs out) and the latest order date;
   - the weeks out of stock even if the item is ordered now;
   - the suggested quantity (26 weeks of cover).

   After the analysis proposes a different lead time, the same code works out the risk level and
   order date again (`with_lead_times`).

These are the two things an outside assistant would need to answer a purchasing question the way
the pipeline does. So each server exposes one of them.

### What each server offers

| Server | What it offers | What it runs |
|---|---|---|
| `purchasing-history` | tool `purchasing_history(supplier_name, item_names, item_codes, as_of, limit)`; resources `suppliers://list` and `items://list` (the names and codes in the store, for when the exact name is not known) | `services/vector_store.search_history`, the function a run uses |
| `purchasing-replenishment` | tool `item_figures(snapshot_date, available_stock, weeks_to_out_of_stock, item_code, item_name, lead_time_weeks)`: one item's figures, at 18 weeks or at any lead time given | `item_grouping.build_item` (the 18-week figures) and `documents.items_after_changes` / `with_lead_times` (the figures at another lead time) |
| | tool `stock_csv_figures(csv_text)`: every item of a stock CSV, grouped by supplier | `item_grouping.group_by_supplier_and_recommend`, the function that reads the CSV in a run |
| | tool `replenishment_policy()`: the lead time, risk rule and coverage | the constants in `services/item_grouping.py` |

### Design choices, and why

- **The pipeline's own functions, nothing reimplemented.** A client gets exactly what a run would use. If
  the risk rule changes in `item_grouping.py`, the server changes with it.
- **No model inside the servers.** They only look up and calculate. Each call is instant, and every
  result can be checked against the pipeline's.
- **`as_of` on the history tool.** A run takes the 12 months before the stock snapshot date, not before
  today. The first draft used today. With the sample data (snapshot 2025-04-05), that window drops a
  record that a run includes (see the test, §4).
- **`lead_time_weeks` on the figures tool.** In v5 the analysis judges one figure, the lead time, and
  code derives the rest from it. The tool takes the same input and gives the same output.
- **No MCP prompt.** The first draft did two things that are now removed:
  - it published the analysis agent's system prompt for clients to use;
  - it told them to "use these tools rather than working the dates or quantities out yourself".

  The v5 analysis prompt describes the pipeline's input: figures already worked out, history documents
  already numbered. A client conversation has no such input.
- **Read-only.** No tool writes to the history store or anywhere else.
- **Set-up details found in the test** (§4):
  - the servers change to the project folder on start, because `config.py` reads `.env` from the
    working directory and a client may start them anywhere;
  - the history server needs `OPENAI_API_KEY`, which Chroma uses to open the store, passed in the
    client's `env` entry.

## 3. How it is used, and what happens

### Set-up (Claude Desktop as the example)

Copy `mcp_servers/claude_desktop_config.example.json` into Claude Desktop's config file, with the real
paths and key:

```json
{
  "mcpServers": {
    "purchasing-history": {
      "command": "python",
      "args": ["C:/path/to/purchasing_automation/mcp_servers/history_server.py"],
      "env": {"OPENAI_API_KEY": "<the key the history store was built with>"}
    },
    "purchasing-replenishment": {
      "command": "python",
      "args": ["C:/path/to/purchasing_automation/mcp_servers/replenishment_server.py"]
    }
  }
}
```

When Claude Desktop starts, it launches both servers and learns their four tools and two resources.

### One question, step by step

This uses the figures of `dev/cases/delayvale_low_becomes_medium.json`, and the results are the ones the
servers returned in the test. The person asks: *"DelayvaleParts has sent a notice about lead times.
Do I need to order BushingRing now? We have 3,600 units, 36 weeks of stock, as of 2025-04-05."*

1. The model calls `purchasing_history(supplier_name="DelayvaleParts", item_names=["BushingRing"],
   as_of="2025-04-05")`. The server runs `search_history` and returns the supplier's notice: lead
   times four weeks longer from April 2025.
2. The model reads that 18 + 4 = 22 weeks. It calls `item_figures(snapshot_date="2025-04-05",
   available_stock=3600, weeks_to_out_of_stock=36, item_name="BushingRing", lead_time_weeks=22)`. It
   gets back:
   - **Medium** (Low at 18 weeks);
   - stock runs out on 2025-12-13;
   - latest order date **2025-07-12**, "within 14 weeks";
   - 2,600 units to cover 26 weeks.
3. The model answers in its own words: not today, but the order must be placed by 12 July 2025, four
   weeks earlier than it would have been.

Every figure in that answer came from the pipeline's code. Which lookups were made, and with what
arguments, was the model's choice.

### The same question in the pipeline

The buyer uploads the stock CSV. Then the steps are fixed:

1. **Code:** reads the CSV and works out the 18-week figures for every item.
2. **Code:** looks up the history for the supplier and for every item, by code and by name.
3. **Model (analysis, first call):** proposes the changes only, for example
   `{"items": ["550001"], "lead_time_weeks": 22, "reason": "..."}` with its finding.
4. **Code:** works out the risk level and order date again at 22 weeks: Medium, 2025-07-12.
5. **Model (analysis, second call):** writes the findings, questions and actions, from the figures
   after the change.
6. **Code and the judge:** check the analysis.
7. **Documents:** the analysis report, then the purchase request and the supplier email.

This case ran that way in `dev/runs/2026-10-25_155928_...`, with the same result as the MCP route.
That is expected, since both run the same code. The difference is who decides steps 2 and 4.

## 4. The test

`python scripts/check_mcp_servers.py` acts as an MCP client. It:

- starts each server over stdio;
- lists its tools, resources and prompts;
- calls every tool and resource once.

Record: `dev/runs/2026-10-25_160908_mcp-servers-test/record.json`.

| Call | Result | Checked against |
|---|---|---|
| `item_figures`: BushingRing, 3,600 units, 36 weeks | Low; stock runs out 2025-12-13; order by 2025-08-09 ("within 18 weeks"); 2,600 units | the 18-week figures of the Delayvale case |
| the same, `lead_time_weeks: 22` | **Medium**; order by **2025-07-12** ("within 14 weeks") | the pipeline's run of that case: the analysis set 22 weeks and BushingRing went Low → Medium |
| `item_figures`: 20 weeks of stock, lead time 24 | Critical; order date 2025-04-05 (the snapshot date), "overdue"; should have been ordered by 2025-03-08; 4 weeks out of stock | 2025-08-23 − 24 weeks = 2025-03-08, before the snapshot, so the order date is set to the snapshot date as in `with_lead_times` |
| `stock_csv_figures`: the `lowfield_all_low` CSV | LowfieldSupplies Low; FlangeGasket and ShaftCollar Low, order by 2025-09-06 and 2025-11-29 | the pipeline's run of that case (report only, every item Low) |
| `replenishment_policy` | 18 weeks (16 import + 2 internal); Critical below the lead time, High below +8, Medium below +17; 26 weeks of cover | `services/item_grouping.py` |
| `purchasing_history`: SupplierA and items 100004 and 100000, as of 2025-04-05 | the ItemE safety incident report of 2024-11-28 | the history store on this PC, which holds three records (the other two are undated, so outside any window, as in a run) |
| `purchasing_history`: ItemE, as of 2026-10-25 | no history: the report is more than 12 months old | the 12-month window, and why `as_of` was added |
| `purchasing_history` with nothing to look up | "Give a supplier name, item names or item codes." | |
| `suppliers://list`, `items://list` | SupplierA; ItemE, 100000, 100004 | |

Each server started and answered in about 5 s, and the calls themselves took under 1.4 s.

**What the test found.** On the first try, every history call failed. An MCP client passes a server
only a minimal set of environment variables unless it is told otherwise, so the server had no
`OPENAI_API_KEY` and could not open the store. The test now passes the environment, and the config
example has the `env` entry.

## 5. Why the pipeline does not use them

### 5.1 The model would decide what is looked up, and a missed lookup leaves no trace

The pipeline always looks up the supplier and every item, by code and by name. A model with the tool
decides the arguments itself, and the arguments decide what comes back. Checked on the store on this
PC:

| Call | Result |
|---|---|
| `purchasing_history(supplier_name="SupplierA", as_of="2025-04-05")` | "No history on record for that supplier or those items." |
| `purchasing_history(item_codes=["100004"], as_of="2025-04-05")` | the ItemE safety incident: off odour and discolouration in 4–6% of units tested, batch held |

The incident report is about the item, whoever supplies it, so it names no supplier. A model that
looked up only the supplier, which is a natural first call, would get "no history". It would then
write an analysis with no safety issue in it. Nothing in the output would show that a record was
missed.

The evaluation depends on the same fixed input. The judge checks that each finding is faithful to the
documents the analysis was given (J1), and code checks that the items and figures match the input
(C1–C10). Neither can check that the right documents were fetched in the first place. To check that,
code would have to do the full lookup anyway, and then it might as well give the result to the
model, which is what the pipeline does.

### 5.2 The figures must come after the model's own judgement, and a tool does not ensure that

In v5 the model never writes a figure it worked out itself. That was not the first design. Two
problems led to it:

- **Before item 21 (2026-10-23): the model moved dates itself.** It proposed order dates directly,
  and for a lead time four weeks longer it moved BearingCap's order date four weeks earlier, to
  2025-03-08. That date was already past: the order date was already the snapshot date, 2025-04-05
  (`dev/runs/2026-10-23_013618_...`).
- **Item 22: a figure written before its own change.** With the analysis in one call, the model wrote
  "6-week gap" in the same answer in which it raised the lead time, making the gap 10 weeks. The wrong
  figure reached the purchase request.

The fix was the order of the steps (§3, steps 3–5):

1. The model judges the lead time and nothing else.
2. Code works out the figures that follow from it.
3. Only then does the model write the text, from those figures.

A figures tool gives correct arithmetic, but only for the arguments the model passes, and only when
it calls the tool. In one answer the model could:

- write a gap before calling the tool;
- call it with 22 weeks and then settle on 24;
- not call it at all and work the date out in its head, as in v4.

The pipeline's order rules all of these out by construction. A tool makes them possible again.

### 5.3 Each tool call is another model call

With tools, every round works like this:

1. The model stops with a tool request.
2. The client runs the tool.
3. The model is called again with the whole conversation so far.

A supplier with four items needs history for the supplier and for each item, then the figures for any
item whose lead time changes. That could mean several rounds, each one a model call with a growing
conversation, in place of the pipeline's two analysis calls. This was not measured, because the
pipeline was not changed to try it.

### 5.4 The people who would use it do not use MCP clients

The buyers work from the documents the pipeline writes:

- the analysis report, with numbered findings and their documents;
- the purchase request for the approvers;
- the email for the supplier.

They do not use Claude Desktop or an IDE. A question such as "what do we have on SupplierA?" is
answered by the analysis report, with each finding linked to its document.

### 5.5 The company's data would go out by a route the pipeline does not control

The stock figures and history are on the company's own server. The pipeline sends them only to the
models it calls, and every call goes through `services/agents.py`, which:

- uses the company's own API keys;
- has a time limit on every call;
- is traced in LangSmith with prompt and response text hidden where real data runs
  (`docs/observability.md`).

A desktop MCP client sends whatever a tool returns to the client's own model, under the client's own
account and logging. None of those settings apply there.

### 5.6 It adds nothing the pipeline does not already do

Both servers are thin wrappers over the pipeline's functions. They can do nothing a run cannot. Their
only use is a one-off lookup outside a run, such as one item's figures at another lead time, which the
analysis report already gives for every item.

## 6. Where MCP would pay

MCP would pay where the model's choice of what to look up is the point, not a risk:

- **Open-ended questions** that a fixed run does not cover, such as "which suppliers have had quality
  incidents this year?" across all records.
- **A data source used only now and then.** Example: the last purchase price from SAP Business One,
  which the purchase request lacks today. It is needed only for the items in a request, and only once
  per request.
- **The same tools wanted by several assistants in the company**, so that one server serves them all.

None of these is the case now.

## 7. For the README, the portfolio and interviews

"MCP servers were built for the purchasing history and the stock figures. They call the pipeline's own
functions and were tested once with an MCP client, and every answer matched the pipeline's. The
pipeline does not use them, for two reasons:

- **Lookups.** Each analysis must be given the complete history and the figures in a fixed order.
  With tools, the model chooses what to look up. A supplier-only lookup returns nothing for an incident
  report that names only the item.
- **Figures.** The model must write figures only after code has recalculated them from its own
  judgement. A tool makes it possible again to write a figure before that, or with other arguments.

MCP would pay for open-ended questions or a rarely needed source such as SAP prices."

## 8. Files and commands

| File | What it is |
|---|---|
| `mcp_servers/history_server.py` | the history server |
| `mcp_servers/replenishment_server.py` | the figures server |
| `mcp_servers/claude_desktop_config.example.json` | client config example |
| `scripts/check_mcp_servers.py` | the test client |
| `dev/runs/2026-10-25_160908_mcp-servers-test/record.json` | the test record |

```
python mcp_servers/history_server.py         # stdio; needs OPENAI_API_KEY for the history store
python mcp_servers/replenishment_server.py   # stdio; no keys
python scripts/check_mcp_servers.py          # the test in §4
```

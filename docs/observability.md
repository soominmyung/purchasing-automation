# Observability: what goes to LangSmith, what stays in-house

Every LLM call in the pipeline sends company data (suppliers, items, available stock,
order dates, history documents) to the model provider. Tracing to LangSmith would send it
to a second outside service, which also keeps it. This page sets out which records are safe
to send there and which stay on the company's own server.

## Summary for the README and portfolio

With `langsmith_hide_content` on (the company deployment), what reaches LangSmith and what
does not, as verified against LangSmith on 2026-10-19:

**Hidden — never sent**
- Every prompt and every model response, at every node: the analysis agent's input (items,
  available stock, dates, history documents) and output, the report, the purchase request,
  the email draft, the email check, and the judge's prompt and verdicts.
- The supplier name in the trace's run name and metadata. Hiding inputs and outputs does not
  cover these two fields, so the pipeline leaves the name out itself.

**Sent — no company data**
- Run tree (pipeline → nodes → model calls) with fixed run names.
- Time per node, token counts per call (input and output), model and provider names, cost
  for OpenAI models.
- Metadata added in code: run id, environment, the judge's check ids and attempt number.

**Still sent, so kept free of data by rule**
- Error messages. Exceptions in traced code say what failed, never what the data was; the
  analysis agent and the judge return failures as results instead of raising them.

**Where the hidden content goes instead:** a run record per run on the company server
(`runs/<run_id>.json`, below), which also serves as the audit trail.

## Two settings

| Where | `langchain_tracing_v2` | `langsmith_hide_content` |
|---|---|---|
| Development and demo (synthetic data) | on | off — full traces for debugging |
| Company deployment (real data) | on | **on** — no prompt or response text leaves |

`langsmith_hide_content` sets `LANGSMITH_HIDE_INPUTS` and `LANGSMITH_HIDE_OUTPUTS`.

## What LangSmith keeps with content hidden

Tested on 2026-10-19 by tracing a GPT-4o-mini call through LangChain and a judge call
through `@traceable`, each carrying a marker string, then reading the runs back:

| Field | Kept | Contains company data |
|---|---|---|
| Run tree (pipeline → nodes → model calls), run names | yes | no — names are fixed in code |
| Start and end time, so latency per node | yes | no |
| Token counts, input and output, per call and summed per trace | **yes** | no |
| Model name and provider | yes | no |
| Cost | yes for OpenAI models; not for the Gemini judge, which LangSmith has no price for | no |
| Metadata added in code (`checks`, `attempt`) | yes | no — keep it that way |
| Inputs and outputs | **no** (`{}` / empty) | — |
| **Error messages** | **yes, in full** | **can** — see below |

The marker string appeared nowhere in the runs except in the error test.

**Error messages are sent even with content hidden.** An exception whose message quotes data
(for example "could not parse model output: …") sends that data. Rules:
- Exceptions raised inside traced code must not include prompt text, model output, supplier
  or item names. Say what failed, not what the data was.
- Already so in the pipeline: the analysis agent and the judge catch parse and API failures
  and return them as results (which are hidden), rather than raising them.

## What stays in-house

| Record | Why it cannot go out | Where it lives |
|---|---|---|
| Inputs to each model call (items, available stock, dates, history documents) | Purchasing position and supplier problems: leverage for suppliers and competitors | Run records, `runs/<run_id>.json` |
| Model outputs (analysis, report, purchase request, email draft) | Same, plus order plans | Run records; the generated documents themselves |
| Judge evidence (quotes from the analysis) | Quotes the analysis | Run records; the evaluation report |
| Data-issue lines, referenced-document paths | Item names, supplier folder names | In the report |
| Email drafts | Supplier contacts' names and addresses: personal data under UK GDPR | Generated output only |

The report itself is the audit record: its opening section lists rows excluded and why, its
closing section lists the history documents used, and the evaluation report shows which
checks passed.

## What LangSmith is used for

- **Development and demo:** debugging a single run from its full trace, and comparing runs
  before and after a prompt change.
- **Company deployment:** cost, latency and failure rates per node, from metadata only.
  Debugging a real run is done from its run record on the company server.

## Run records

`services/run_record.py` writes one JSON file per run to `runs/` (`run_records_dir`), also
when the run fails. Each holds:

- the run: id, start and finish time, status, client IP, prompt version, model per agent
- the input: the CSV exactly as received, the snapshot date and where it came from
- per supplier group: the items as preprocessed, data-issue lines, the history documents the
  model read (their text at the time), every output, the evaluation, each email check with
  its feedback, and per node the seconds, tokens per model (as the provider reports it,
  e.g. `gpt-4o-2024-08-06`) and any error

The client IP stands in for a user name (there is no login); on the company server it is the
requesting PC, matching how the IT manager tracks SAP use. Behind a proxy, set
`trust_forwarded_for`. An IP address can be personal data under UK GDPR.

`runs/` is git-ignored. On the company server it needs access limited to purchasing and the
operator, a retention period, and inclusion in the D-1 backup. A run with three suppliers
came to about 50 KB.

`scripts/summarize_runs.py` tabulates the records by prompt version: groups marked for
review, failures per check, checks the judge could not answer (counted apart), emails that
failed their check in the end, rewrites, tokens and time.

When content is hidden, the supplier name is also left out of the LangSmith run name and
metadata, which hiding inputs and outputs does not cover.

## Not covered by LangSmith

Server and network failures (the IT manager's monitoring), the shared-folder sync (its own
output and exit code), and anything outside LLM calls.

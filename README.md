# Multi-agent Purchasing AI Suite

A purchasing-decision pipeline: from a stock CSV and the company's supplier and item history
documents, it writes an **analysis report** for the purchasing officer (buyer), a **purchase
request** for the approvers and a **draft email to the supplier**, one set per supplier. Code does the arithmetic,
retrieval and document assembly; LLMs read the history and write; every draft is checked by code
and by an LLM judge before the buyer sees it.

> **Portfolio (Korean, 7 pages):** [docs/Purchasing_AI_Suite_ko.pdf](docs/Purchasing_AI_Suite_ko.pdf)

| | |
|---|---|
| Code and LLM roles | Code: risk, order dates and quantities, history retrieval, document assembly. LLM: reading the history, judgements, wording |
| History retrieval | Metadata-based RAG: documents found by their supplier, item and date headers, given to the model whole |
| Quality checks | 36 checks across the three documents (code and LLM judge); a failed analysis or email is sent back with the reasons, up to two rewrites |
| Model choice | Every agent's model and effort chosen on the same inputs, scored by the checks and read by hand (see [Model choice](#model-choice)) |
| Time and cost | About 80 s and $0.05 for one supplier with seven items, all checks included, no rewrite |

---

## Project history

- **Sep–Dec 2025, at Korea Foods.** Prototyped in n8n with the purchasing team, then built as a
  LangGraph multi-agent system on GPT-4o and deployed on the company's on-premises server, where a
  few members of the purchasing team used it on real data. A first fine-tuning study, Llama-3-8B
  with QLoRA SFT, as a self-hosted alternative to the API.
- **Feb 2026: reconstructed for this repository.** The company's code stayed with the company, so
  the system was rebuilt here from scratch, on synthetic data. The February commits record that
  reconstruction, not the first development.
- **Mar–Aug 2026.** The fine-tuning reproduced here on synthetic data, and the model served with vLLM.
- **Oct 2026: v5 overhaul.** The work between code and LLMs redrawn, the checks written and the
  judge validated, every agent's model chosen by measurement, and the self-hosting question
  measured again on the current pipeline.

---

## How a run works

1. **Input.** The buyer exports the stock position from an SAP Business One user query as a CSV and
   uploads it. The pipeline validates every row and groups the items by supplier, since ordering,
   negotiation and enquiries all happen per supplier.
2. **Figures in code.** Risk level from weeks to stockout against the 18-week lead time, latest
   order and delivery dates, and the suggested quantity (26 weeks of sales after arrival).
3. **History in code.** For each supplier and item, the documents of the 12 months before the
   snapshot, the latest five per scope, matched on their headers.
4. **Analysis (LLM).** `gpt-6.1-sol` states the facts in the history, each citing its document, and
   proposes lead-time or quantity changes, questions for the supplier and actions, each citing the
   facts it rests on. Code recalculates every figure after a proposed change.
5. **Checks.** Code checks first; then the judge (`claude-haiku-5-5`). A blocking failure sends the
   analysis back with the reasons, up to two rewrites; if it still fails, the attempt with the
   fewest failures is used and re-checked.
6. **Documents.** The analysis report is assembled in code. If anything needs ordering, the
   purchase request (`gpt-6.1-sol`) and the supplier email (`gpt-6.1-sol`) are written in parallel,
   each from only what it needs: the email agent never receives stock figures or risk levels.
7. **Document checks.** The purchase request is checked for added or altered facts; the email is
   checked for anything internal (code and Haiku, rewritten up to twice) and then for coverage and
   tone (judge, warnings).
8. **Output.** Word drafts for the buyer, each with a note at the top listing what is unresolved.
   The quality-check document and the run record stay on the server for the operator.

### Evidence trail

Source documents are numbered `[1], [2]…`; each fact cites its documents and is numbered
`[F1], [F2]…`; each proposed change, question and action cites the facts behind it. Facts drawn
from the stock figures carry no document number and are listed apart. Code checks that every
citation resolves; the judge checks that each one supports what rests on it.

---

## Quality checks

| Document | Checks | By | On failure |
|---|---|---|---|
| Analysis | C1–C10 (format, items, figures, citations, recalculated dates) and J1a–J8 (history stated, not distorted, applied to the right supplier and item; changes justified and sized; supplier questions free of internal data) | Code, LLM judge | Blocking: rewrite (up to 2). Warnings: note in the draft |
| Purchase request | P1a–c (nothing added, figures as given, meaning kept), P2 (length), P3 (citations) | LLM judge, code | Note in the draft |
| Supplier email | E1–E3 (no internal data, no past supplier problem, no item outside the order) | Code, Haiku | Rewrite (up to 2), then note |
| | E4–E7 (every point asked, no repeat, earliest dispatch asked, tone of a request) | LLM judge | Note in the draft |

Definitions: [`docs/eval_rubric.md`](docs/eval_rubric.md). Check ids were renumbered on 29 Oct 2026;
records written before use the old ids (mapping: `LEGACY_IDS` in `services/quality_checks.py`).

**The judge was validated before use** ([`docs/judge_selection.md`](docs/judge_selection.md)):
on 51 samples with planted faults it caught 93.9%, with 3.9% of verdicts changing between runs and
8.0% false alarms on 89 hand-scored verdicts. Letting Haiku reason before answering raised its
detection from 83.7%.

---

## Model choice

Each agent's model was chosen on fixed inputs, scored by the checks above and read by hand:

| Agent | Chosen | Compared with | Record |
|---|---|---|---|
| Analysis | `gpt-6.1-sol`, medium effort | `gpt-4o`, `gpt-6-luna`, `claude-sonnet-5-5`, Gemini Flash (7 models, 9 settings) | [`analysis_model_selection.md`](docs/analysis_model_selection.md) |
| Judge | `claude-haiku-5-5`, high effort | Haiku without reasoning, Haiku medium, `gpt-6-luna`, Gemini Flash-Lite | [`judge_selection.md`](docs/judge_selection.md) |
| Purchase request | `gpt-6.1-sol`, low effort | `gpt-4o`, `gpt-5.4-mini`, `gpt-6-luna` at several efforts | [`approval_model_selection.md`](docs/approval_model_selection.md) |
| Supplier email | `gpt-6.1-sol`, low effort | `gpt-5.4-mini`, `gpt-6-luna` at several efforts | [`email_model_selection.md`](docs/email_model_selection.md) |
| Email validator | `claude-haiku-5-5`, medium effort | `gpt-4o-mini`, Haiku low and high | [`email_model_selection.md`](docs/email_model_selection.md) |

---

## Self-hosting: measured, not adopted

Gemma 4 12B was served with vLLM on one RTX 5090 and fine-tuned with QLoRA SFT
([`docs/serving_gemma4_vllm.md`](docs/serving_gemma4_vllm.md),
[`docs/sft_gemma4.md`](docs/sft_gemma4.md): the training data and what one example is):

- **Serving is feasible:** in FP8 it takes 13.3 GiB (fits one L4), with 1.6–1.8× the throughput of
  bf16 and a quality difference within run-to-run variation.
- **The model is not good enough:** on the 25 evaluation cases the untuned model passed every
  blocking check in 14, against 24 for `gpt-6.1-sol`; every fine-tuned adapter did worse (3–10).
  Production stays on `gpt-6.1-sol`.

The earlier Llama-3-8B study scored outputs with a GPT-4o preference score and a
short placeholder prompt; its figures are not comparable with these and are no longer claimed.

---

## Data protection

- The email agent is not given stock figures, risk levels or internal questions, and its output is
  checked for them.
- LangSmith receives the run tree, times, tokens and model names only; prompt and response text and
  supplier names are kept out.
- Each run is recorded on the company server as JSON (inputs, history used, outputs, checks, times,
  tokens), with run and check statistics appended to CSV files.
- All data in this repository is synthetic.

---

## Running it

**Requirements:** Python 3.11, an OpenAI key, an Anthropic key (judge and email validator).

```bash
git clone https://github.com/soominmyung/purchasing-automation.git
cd purchasing-automation
pip install -r requirements.txt
cp .env.example .env        # set OPENAI_API_KEY, ANTHROPIC_API_KEY, API_ACCESS_TOKEN
uvicorn main:app --reload   # API docs at http://localhost:8000/docs
```

Or with Docker: `docker build -t purchasing-ai .` then `docker run -p 8080:8080 --env-file .env purchasing-ai`.

**History documents.** Upload PDFs to `POST /api/ingest/history` (or a ZIP to `/api/ingest/history/zip`).
Each document starts with header lines that say what it is about: `Supplier:`, `Item:`,
`ItemCode:`, `Date:`. A supplier only means the whole supplier; supplier and item, that item from
that supplier; item only, the item whoever supplies it. On the company server, a shared folder is
kept in sync instead (`/api/ingest/history/folder`).

**Stock CSV.** Upload to `POST /api/run` (or `/api/run/stream` for progress events), with the `X-API-Key`
header:

```
SnapshotDate,2025-04-05

ItemCode,ItemName,SupplierName,AvailableStock,WksToOOS
100000,ItemA,SupplierA,100,25
```

- `AvailableStock`: on hand + on order − committed (SAP Business One `OITW`).
- `WksToOOS`: available stock ÷ average weekly sales over the previous four months.
- The snapshot date comes from the `SnapshotDate` line, or else from a DDMMYY date in the file name
  (`050425.csv`). A snapshot more than three days old is refused (`max_snapshot_age_days`; the
  sample's run date is fixed in `config.py` so it passes).

A sample is in [`050425.csv`](050425.csv). Development cases with history documents are in
[`dev/cases/`](dev/cases/); `python scripts/run_dev_case.py dev/cases/kestrel_components.json --label "…"`
runs one through the whole pipeline and records it under `dev/runs/`; a sample record of that case:
[`dev/runs/2026-10-29_004531_judge-checks-renumbered-without-gaps-j3a-c-j4-j8/`](dev/runs/2026-10-29_004531_judge-checks-renumbered-without-gaps-j3a-c-j4-j8/).

**Main settings** (`config.py`, overridable by environment variables): `evaluator_model`,
`analysis_rewrites`, `history_window_months`, `history_docs_per_scope`, `prompt_version`,
`langsmith_hide_content`, `deliver_evaluation_report`. Agent models and efforts are in
`services/agents.py` (`AGENT_MODELS`, `AGENT_EFFORT`).

---

## Repository

| Path | Contents |
|---|---|
| `main.py`, `config.py`, `schemas.py` | FastAPI app, settings, request and response models |
| `routers/` | API: `pipeline` (run), `ingest` (history documents), `output` (downloads) |
| `services/` | `agents` (LangGraph graph and agents), `quality_checks` (code checks and judge), `documents` (document assembly), `item_grouping` (risk, dates, quantities), `vector_store` and `history_folder` (history store and shared-folder sync), `run_record`, `security` |
| `utils/` | CSV parsing and validation, PDF reading, Word output |
| `prompts/v5/` | The prompts in use (`prompt_version` in `config.py`); a version is frozen once results are recorded with it, and changes go into a new version folder |
| `scripts/` | Model comparisons, judge validation, dev runs, SFT and serving experiments |
| `eval_data/`, `eval_results/` | Evaluation scenarios, judge calibration samples and hand scores; raw results of every comparison |
| `dev/` | Development cases (kept apart from the evaluation set) and one recorded run |
| `docs/` | Quality checks, model selection records, serving measurement, portfolio (Korean PDF) |
| `mcp_servers/` | History and replenishment MCP servers, built and tested but not used in the pipeline |

## Tech stack

Python · FastAPI · LangGraph · LangSmith · OpenAI and Anthropic APIs · ChromaDB · MCP ·
QLoRA SFT (Unsloth, TRL) · vLLM · Docker · GitHub Actions · GCP Cloud Run

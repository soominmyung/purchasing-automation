# Changes, October 2026

Working record of the pipeline and evaluation changes made from 2026-10-15 to 2026-10-18,
kept so the README, the portfolio and the commit history can be updated from one place.
Each entry gives what changed, why, and what was measured where something was.

**Status:** committed on main from 88571b7 to the commit that adds this file. The version
before these changes is kept on branch `legacy/v1` (3866d60). Not yet committed:
`mcp_servers/`, and `scripts/eval_v2_score.py` until it is moved onto the shared checks.

## Pipeline

### 1. Risk level derived from weeks to stockout, with a Critical level

- Risk is computed in code from WksToOOS: **Critical** below the 18-week lead time (16 import
  + 2 internal), High below 26, Medium below 35, Low beyond. The source CSV no longer
  carries a RiskLevel column.
- Critical items carry `expected_stockout_weeks` (18 − weeks to stockout) and a fixed note
  written by code ("Critical: even if ordered today, a stockout of about N weeks is
  expected …"); their PO timing is "overdue".
- Why: the CSV's own RiskLevel could contradict its weeks to stockout, and nothing marked
  items that will run out before any order can arrive.
- Figures that are whole numbers stay whole, so reports show "450 units", not "450.0".

### 2. Input validation and the "Data Issues Requiring Attention" section

- Rows with a missing or invalid identity field, duplicate rows within a supplier, and rows
  whose CurrentStock or WksToOOS is missing, non-numeric, negative or implausibly large are
  excluded before analysis. Zero stock and zero weeks are valid; decimals are valid.
- Excluded rows are listed, by file line number, in a section placed first in every report.
- Why: the pipeline used to fill a missing figure with an 18-week assumption, so a bad input
  produced a confident analysis.

### 3. Snapshot date

- Read from a `SnapshotDate,YYYY-MM-DD` line at the top of the CSV; if that is absent or
  invalid, from a DDMMYY date in the file name; otherwise the upload is rejected (400).
  A rejected cell is reported in the data-issues section.
- The sample CSV (`050425.csv`) now starts with the reference line.

### 4. History retrieved in code, not through model tool calls

- The analysis agent used to be given two tools and told to write search queries, but the
  code ignored those queries and retrieved by metadata filter (supplier name, item codes)
  ranked by document date. The tool call only decided whether retrieval happened at all.
- Now the code always retrieves both histories first and passes them in the input JSON as
  `supplier_history` and `item_history`; the model is called once.
- Why: removes the case where the model skipped a lookup, saves one LLM call per supplier,
  makes production and evaluation inputs identical, and makes the prompt describe what the
  code actually does.
- Also fixed: an empty supplier name used to retrieve every supplier's documents.
- The previous prompts are kept in `services/legacy/prompts_tool_calling_20261018.py`; the
  previous tool-calling code is in git history.

### 5. "Referenced Documents" section

- Every report ends with a table of the history documents the analysis was based on
  (history type, file name, supplier or item code, document date), written by code from the
  retrieval result, or "No supplier/item history was found."
- File names are stored at ingest from now on. Documents ingested earlier show
  "(file name not recorded)" until they are re-ingested.

### 6. "No notable issues" kept apart from "no history"

- When history existed but described nothing notable, GPT-4o often wrote the fixed sentence
  "No historical supplier or item data available …", which tells a buyer the records are
  missing. The rule now says to write "History reviewed: no notable supplier or item issues."
  and to keep the fixed sentence for when both histories are empty.
- Measured on the 9 affected scenarios (38 items, 2 runs each): 7 items wrong with the
  committed prompt, 19 with the intermediate prompt. After the change, across all 25
  scenarios: 0 of 49 items with history; the two no-history scenarios still state it.
- Cause: the old rule only covered "positive findings", and the training scenarios all
  carried incidents, so the case had never been tested.

### 7. Evaluation node rewritten

- The node used to score four 1–10 criteria with GPT-4o judging its own analysis, one of
  them about email leaks in an analysis that contains no email.
- It now runs the same checks as the offline evaluation (`services/quality_checks.py`):
  code checks for format, items and figures, and yes/no judge checks by a model from a
  different provider (`evaluator_model`, provisionally `gemini:gemini-3.7-flash`). The
  analysis is marked ready only if every check passes; results are in the API response.
- `GEMINI_API_KEY` added to the deploy workflow; the GitHub secret still has to be created.

### 8. Prompt changes for Critical items

- Analysis: name every Critical item with its gap in the summary and recommend a mitigation;
  reproduce the code-written notes; never fill in a missing figure.
- Email: Critical items get the earliest dispatch date and a faster shipping option, without
  revealing internal urgency.

## Evaluation

- **Rubric** (`docs/eval_rubric.md`): fixed checklist replacing imitation-of-GPT-4o scoring;
  J2d, J4a, J4b dropped as the pipeline now handles them; J5 added for Critical items.
- **Scenarios**: 20 generated + 5 original holdout. Unattributed supplier histories relabelled
  as material. Weeks to stockout reset for 29 items (`scripts/set_eval_risk_mix.py`) so the
  set covers all risk levels: 23 Critical / 13 High / 10 Medium / 7 Low.
- **GPT-4o outputs** regenerated through the production preprocessing
  (`scripts/gen_eval_outputs_gpt4o.py`).
- **Judge validation** (`docs/judge_validation.md`): 33 seeded-error outputs built
  (`scripts/build_judge_calibration.py`); hand scoring and the judge run not yet done.
- **Judge costs** measured per call for 13 models (`docs/judge_model_costs.md`).

## MCP servers

`mcp_servers/` (history and replenishment servers, Claude Desktop config example). Docs
to be written after testing in Claude Desktop.

## Still to do

- Judge validation: hand-score about 10 originals, run 8 candidates × 3, choose the
  production judge and the offline panel.
- Refactor `scripts/eval_v2_score.py` onto `services/quality_checks.py`.
- Analysis-model benchmark (GPT-4o against 7 others on quality, cost, latency).
- **SFT retraining:** the fine-tuned model was trained on raw CSV rows, source risk levels
  with no Critical, and GPT-4o outputs from the old prompt. Regenerate the teacher data with
  the current pipeline and retrain before comparing it with GPT-4o.
- Email validator wrongly flags proposed purchase quantities as internal stock figures, and
  a failing email is still sent after three rewrites.
- Undecided: whether to call the model at all when every item is Low; whether the PR
  document also gets the data-issues section.
- Re-ingest history PDFs so file names appear in Referenced Documents.
- Update README, portfolio and resume: risk rule, input CSV format, evaluator, retrieval.
- Workload Identity Federation for deployment; delete branch `backup/before-coauthor-fix`.

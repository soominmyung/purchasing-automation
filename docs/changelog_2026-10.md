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
- The previous prompts are kept as `prompts/v2/`; the previous tool-calling code is in git
  history.

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

### 9. One kind of history, scoped by the document's header *(uncommitted)*

- Supplier history and item history are no longer separate. Many records sit across the
  boundary (a defect in one item from one supplier), and the split depended on whoever
  uploaded a document choosing the right folder. When a supplier-level record mentioned a
  quality problem without naming items, GPT-4o copied it into every item's notes (eval_08).
- Each document's header decides what it is about: `Supplier:` only = the supplier as a
  whole; `Supplier:` and `Item:` = those items from that supplier; `Item:` only = the items
  whoever supplies them (e.g. a demand spike). A document naming neither is refused.
- Retrieval: the 10 most recent documents whose supplier is this supplier or whose items
  include any item in the analysis, matched by item name or code, ignoring case and spacing.
- The model receives the documents with their header fields and applies each only to what
  it names: supplier-wide records once in the summary, item records in that item's notes,
  an unattributed quality problem as a question about which items were affected.
- J1 now also fails an analysis that attributes a document to a supplier or item it does
  not name.
- The earlier upload routes still work and are handled the same way. Existing documents
  were moved with `scripts/migrate_history_store.py`. The previous prompts are kept as
  `prompts/v3/`.

### 10. Sync from the shared folder *(uncommitted)*

- `scripts/sync_history_folder.py` walks the shared history folder (any layout: by year, by
  supplier, …) and sends only what changed: new files added, changed files replaced, deleted
  files removed. Documents are identified by their path, so a file sent twice is held once.
- The folder layout is never used to classify; a header that disagrees with a supplier
  folder is reported as a warning (usually a misfiled document or a header typo).
- Deletions are held back when they would remove more than half of the documents on file,
  which is what a disconnected drive looks like.
- Deployment as described for the company: the script runs `--local` on the on-prem server
  that is always on for SAP ERP and sees the network drive directly. The store is derived
  data: it can be rebuilt from the folder at any time, and the on-prem backup server keeps
  the previous day's copy. Schedule the sync outside the backup window.
- The Cloud Run deployment is a demo. Its disk is temporary, so ingested history is lost on
  restart and differs between instances.
- `/history` (single upload, for the homepage showcase) is kept; re-uploading a file name
  now replaces it.

### 11. `CurrentStock` renamed `AvailableStock` *(uncommitted)*

- The stock column has always been on hand + on order − committed, but its name read as
  the quantity in the warehouse. Renamed in the CSV, the item fields (`available_stock`),
  the prompts, the report table and the evaluation data; the prompt now defines it and
  tells the model never to call it "in stock" or "on hand". A CSV with the old column name
  is still read.
- The fixed no-history sentence now ends "analysis based on available stock and deadlines
  only".

### 12. Judge calls traced *(uncommitted)*

- The evaluation node's judge calls go to the providers' SDKs directly, so LangSmith showed
  the node but not what the judge was sent, what it answered or what it cost. They are now
  traced (`Judge` and `Judge_LLM` runs under the evaluation node) with token usage,
  including Gemini's thinking tokens. No effect when tracing is off.

### 13. Tracing without sending company data *(uncommitted)*

- New setting `langsmith_hide_content`, for wherever real data runs: traces keep the run
  tree, timings, token counts and model names but no prompt or response text.
- Tested against LangSmith: token counts survive hiding; a marker string in the data did not
  appear in any run. Error messages are still sent in full, so exceptions in traced code
  must not quote data. Details and the split between LangSmith and in-house records:
  `docs/observability.md`.

### 14. Prompt versions *(uncommitted)*

- Prompts now live in `prompts/<version>/`, one folder per version holding the whole set,
  including the email check and rewrite prompts that were written inline in the code.
  `prompt_version` in `config.py` picks the folder; run records and evaluation outputs store
  the version name (earlier, evaluation outputs stored a hash of the prompt, which a person
  could not read).
- v1 (branch `legacy/v1`) to v4 (current) reconstructed from the earlier prompts, verbatim;
  `prompts/CHANGES.md` says what each changed. `services/legacy/` removed.
- A version is not edited once results have been recorded with it.
- The Docker image now copies `prompts/`.

### 15. Run records *(uncommitted)*

- Every run writes `runs/<run_id>.json`, also when it fails: the CSV as received, client IP,
  prompt version and models, and per supplier the preprocessed items, the history documents
  read, all outputs, the evaluation, each email check, and per node time, tokens and errors.
  Details: `docs/observability.md`. Summary across runs: `scripts/summarize_runs.py`.
- Nodes are timed and their token use recorded (LangChain usage callback, plus the judge's
  own counts). Agent models are set in one table (`AGENT_MODELS`) that the record copies.
- Email checks are kept in a record-only list; the rewrite still receives only the latest
  feedback.
- Generated files carry the run id, so a rerun no longer overwrites an earlier run's
  documents. Evaluation reports can now be downloaded (they were missing from the download
  route). Responses include `run_id`.
- With content hidden, the supplier name is no longer sent in the LangSmith run name and
  metadata.
- First real run (sample CSV, three suppliers): the judge took 9–56 s per supplier, up to
  four times the analysis agent, with up to 7,032 output tokens; the email check rejected one
  draft for "internal stock quantities" that were proposed order quantities (the known
  validator fault), and the rewrite passed.

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
- Undecided: keep or drop `target` / `reason` on critical questions. No code reads them,
  downstream prompts use only the question text, and `reason` is sometimes wrong (H2: a
  stock-driven question marked `item_history`). Decide when the output format is settled
  for the model benchmark and SFT retraining.
- README and portfolio: explain what is hidden from LangSmith and what is still sent, from
  the summary at the top of `docs/observability.md`.
- The evaluation node runs between analysis and report, so the judge's time (9–56 s per
  supplier in the first run record) adds to every run, though nothing after it uses its
  result. Once the production judge is chosen, consider running evaluation alongside
  report → purchase request → email.
- Re-ingest history PDFs so file names appear in Referenced Documents.
- Demo: load sample history PDFs at startup when the store is empty, so a restarted
  Cloud Run container still shows history-based questions.
- Chroma embeds every document with OpenAI at ingest, but retrieval never uses similarity;
  either use the embeddings for something (e.g. similar past cases) or store history
  without them.
- Convert the evaluation scenarios to history documents and regenerate the GPT-4o outputs,
  seeded errors and hand-scoring sheet.
- Interview prep for the on-prem deployment (`docs/interview_prep_deployment.md`, not
  committed), separating what was done from what the design allows.
- README: the sentence "the in-house deployment implements per-user intranet ID/PW login"
  does not match the code, which has no login (only a shared API key and a per-IP limit).
  Replace it with what is true: runs are recorded with the requesting IP, which the IT
  manager already uses to track SAP use. No login is planned.
- Update README, portfolio and resume: risk rule, input CSV format, evaluator, retrieval.
  State the input definitions, which the documents do not give today:
  `AvailableStock` = on hand + on order − committed (from SAP Business One 9.3:
  `OITW.OnHand + OnOrder − IsCommited`), and `WksToOOS` = `AvailableStock` ÷ average
  weekly sales over the previous four months (invoice lines net of returns, in inventory
  units). Because both use the same stock, `AvailableStock ÷ WksToOOS` is the weekly sales
  the suggested quantity is built on. On-order stock is counted to avoid overstock on
  perishable food; known limits: on-order stock is counted as if already arrived, and
  weeks out of stock pull the average down.
- Workload Identity Federation for deployment; delete branch `backup/before-coauthor-fix`.

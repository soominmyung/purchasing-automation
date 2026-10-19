# Prompt versions

Each folder is the full set of prompts the pipeline uses. The version in use is
`prompt_version` in `config.py`, and run records and evaluation outputs store it.

Do not edit a version once runs or evaluation results have been recorded with it: copy it
to a new folder and change that. Typo-level fixes in the newest version, before anything has
been recorded with it, can be made in place.

**Prompt versions are not pipeline versions.** Branch `legacy/v1` is the whole pipeline
before October 2026; `prompts/v1` is the prompt set of that branch. From here on the prompt
version moves on its own whenever a prompt changes. Pipeline changes (code, data rules,
storage) are recorded in `docs/changelog_2026-10.md`.

| Prompt version | Pipeline state | Where the code is |
|---|---|---|
| v1 | Before October 2026 | branch `legacy/v1` (3866d60) |
| v2 | Critical risk level, input validation; history still by tool calls | never committed as a prompt set (the next commit, ddd4a06, already holds v3); kept only as this folder |
| v3 | History retrieved in code; Referenced Documents section | commit ddd4a06 to a475674 |
| v4 | One history store scoped by document headers; `AvailableStock` | uncommitted as of 2026-10-19 |

## Files

| File | Used by |
|---|---|
| `analysis.md` | Analysis agent |
| `report_doc.md` | Report agent |
| `pr_draft.md`, `pr_doc.md` | Purchase request agents |
| `email_draft.md` | Email draft agent |
| `email_revision_system_suffix.md`, `email_revision_request.md` | Email rewrite after a failed check (`{feedback}` is filled in) |
| `email_validator_system.md`, `email_validator.md` | Email check (`{email}` is filled in) |
| `evaluation.md` | v1 only: the evaluator before the checklist |

The email check and rewrite prompts were written inline in `services/agents.py` until
2026-10-19; their text is the same in every version.

---

## v4 — 2026-10-19 (current)

**What changed**
- History is one list of documents, each carrying the supplier and items named in its
  header. Each document is applied only to what it names: supplier-wide records once in the
  summary, item records in that item's notes; a document about an item of this supplier
  that is not in the analysis may be mentioned in the summary only.
- A quality problem in a supplier-wide record that names no item is not attributed to any
  item; the model asks which items were affected.
- `current_stock` renamed `available_stock` and defined in the prompt as on hand + on order −
  committed; the model must not call it "in stock" or "on hand". Report tables say
  `AvailableStock`.
- The fixed no-history sentence ends "analysis based on available stock and deadlines only".

**Why**
- The supplier/item split depended on which folder a document was uploaded to. A
  supplier-level record mentioning quality problems without naming items was copied into
  every item's notes (scenario eval_08), telling buyers all three items had quality issues.
- "CurrentStock" read as the quantity in the warehouse, while the figure includes stock on
  order and excludes stock committed to customers.

**Effect measured:** not yet; GPT-4o outputs are to be regenerated with v4 once the
evaluation scenarios are converted to history documents.

## v3 — 2026-10-18

**What changed**
- No tools. History is retrieved in code and passed as two input fields, `supplier_history`
  and `item_history`; the tool section and query-writing instructions removed. The prompt
  states that supplier history belongs to this supplier even when no company is named.
- History that exists but describes nothing notable is reported as "History reviewed: no
  notable supplier or item issues"; the fixed "No historical supplier or item data available"
  sentence is kept for when both histories are empty.

**Why**
- The model was told to write search queries, but the code ignored them and retrieved by
  metadata filter; the only effect of the tool call was that the model could skip it.
- The earlier rule summarised only "positive findings", so for unremarkable history the
  model wrote the no-history sentence, telling buyers records were missing when they were not.

**Effect measured** (GPT-4o, evaluation scenarios): items with history wrongly given the
no-history sentence went from 7 of 38 (v1 rule) and 19 of 38 (v2) on the 9 affected
scenarios, to 0 of 49 across all 25 scenarios; the two scenarios without history still
state it.

## v2 — 2026-10-18

**What changed**
- Lead time stated as 18 weeks (16 import + 2 internal). Every Critical item named in the
  summary with its expected stockout weeks and a mitigation (expedited shipping, alternative
  source); code-written notes reproduced at the start of the item's notes; missing or invalid
  figures shown as N/A, never filled in; 0 weeks treated as a real stockout.
- Email agent: Critical items get the earliest dispatch date and a faster shipping option,
  without revealing internal urgency.
- Report agent told not to write its own data-issues section (the pipeline adds it).
- Evaluation prompt removed: the evaluation node now runs the checklist in
  `services/quality_checks.py` with a judge from another provider.
- Still tool calling for history.

**Why**
- Risk level now comes from weeks to stockout in code, with a Critical level for items that
  will run out before any new order can arrive; the prompts had to use it.
- The old evaluator had GPT-4o scoring its own analysis, and scored an email-leak criterion
  on an analysis that contains no email.

## v1 — before October 2026

Prompts of branch `legacy/v1` (commit 3866d60): the model called `supplier_history` and
`item_history` tools with its own queries; risk level was taken from the CSV's RiskLevel
column (High / Medium / Low, no Critical); a GPT-4o evaluator scored four criteria from 1 to
10. The SFT and DPO models were trained on GPT-4o outputs generated in March 2026 with the
prompts of that time, which belong to this generation (tool calling, no Critical).

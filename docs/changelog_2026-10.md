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
- `GEMINI_API_KEY` added to the deploy workflow and as a GitHub secret (2026-10-22, with the
  new LangSmith key).

### 8. Prompt changes for Critical items

- Analysis: name every Critical item with its gap in the summary and recommend a mitigation;
  reproduce the code-written notes; never fill in a missing figure.
- Email: Critical items get the earliest dispatch date and a faster shipping option, without
  revealing internal urgency.

### 9. One kind of history, scoped by the document's header

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

### 10. Sync from the shared folder

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

### 11. `CurrentStock` renamed `AvailableStock`

- The stock column has always been on hand + on order − committed, but its name read as
  the quantity in the warehouse. Renamed in the CSV, the item fields (`available_stock`),
  the prompts, the report table and the evaluation data; the prompt now defines it and
  tells the model never to call it "in stock" or "on hand". A CSV with the old column name
  is still read.
- The fixed no-history sentence now ends "analysis based on available stock and deadlines
  only".

### 12. Judge calls traced

- The evaluation node's judge calls go to the providers' SDKs directly, so LangSmith showed
  the node but not what the judge was sent, what it answered or what it cost. They are now
  traced (`Judge` and `Judge_LLM` runs under the evaluation node) with token usage,
  including Gemini's thinking tokens. No effect when tracing is off.

### 13. Tracing without sending company data

- New setting `langsmith_hide_content`, for wherever real data runs: traces keep the run
  tree, timings, token counts and model names but no prompt or response text.
- Tested against LangSmith: token counts survive hiding; a marker string in the data did not
  appear in any run. Error messages are still sent in full, so exceptions in traced code
  must not quote data. Details and the split between LangSmith and in-house records:
  `docs/observability.md`.

### 14. Prompt versions

- Prompts now live in `prompts/<version>/`, one folder per version holding the whole set,
  including the email check and rewrite prompts that were written inline in the code.
  `prompt_version` in `config.py` picks the folder; run records and evaluation outputs store
  the version name (earlier, evaluation outputs stored a hash of the prompt, which a person
  could not read).
- v1 (branch `legacy/v1`) to v4 (current) reconstructed from the earlier prompts, verbatim;
  `prompts/CHANGES.md` says what each changed. `services/legacy/` removed.
- A version is not edited once results have been recorded with it.
- The Docker image now copies `prompts/`.

### 15. Run records

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

### 16. v5 design: which parts need an LLM *(decided 2026-10-19 to 2026-10-22; implemented 2026-10-22)*

What changed from v4 and the effect measured on the first runs, including run time: `prompts/CHANGES.md`, v5.

The decisions below are LLM-design choices to explain in the README and portfolio. Each has
the problem that prompted it.

**Order of work.** v5 is built before judge validation, because the judge must be validated on
the output format it will grade in production. Then judge validation, the analysis-model
benchmark, SFT retraining.

**One rule for where LLMs are used: judgement, and writing for a different reader.** Facts and
layout are code.

| Step | v4 | v5 | Why |
|---|---|---|---|
| Analysis | LLM writes judgement and a free-form markdown report | LLM writes judgement only, in fields | The report was written twice: once by the analysis agent and again by the report agent, so figures went through two LLMs and only the first was checked |
| Analysis report (for the buyer) | Report agent (LLM) rewrites the analysis markdown, in the style of exemplar reports from RAG | **Code template**; the report agent is removed | Same reader as the analysis; nothing to rewrite. One company layout that rarely changes, so a template file edited on request does the exemplar's job |
| Purchase request (for the approvers) | Two LLM agents: one restates the analysis as JSON, one writes the document | **One LLM agent** writes the purpose and justification for approvers; code renders the item table, timing and signature block | Approvers need a request ("this seeks approval to buy … because …"), a different reader and tone; the figures need no LLM |
| Supplier email | LLM agent ⇄ validator agent | unchanged | Outside reader; internal facts must be hidden |
| Evaluation | Judge from another provider | unchanged | Cross-checking judgement |

The multi-agent design is described by reader: buyer (code), approvers (LLM), supplier (LLM
with a validator), and a judge from another provider. A template-drafting agent (exemplar
document in, template draft out, human approval) was considered and not built: the company has
one layout and it rarely changes.

**Analysis output fields (prompt v5, revised 2026-10-22).**
`summary`; `findings` (numbered facts: `{id, items, text, basis: {stock, documents}}`);
`critical_questions` (`{question, ask: supplier | internal, items, findings}`);
`po_date_changes` and `quantity_changes` (each with `reason` and `findings`);
`recommended_actions` (`{action, items, findings}`).
Removed: `purchasing_report_markdown`, `replenishment_timeline` (it made the model copy the
item figures), the questions' `reason` (no code read it and it was sometimes wrong), and the
separate `supplier_findings` / `item_findings`, now one numbered list.

**Decisions, each with what led to it.**
- **No classification of findings (e.g. a "notable" yes/no).** Such a label needs a definition
  of what counts (is substitution notable?), and that definition would then decide which
  questions and report sentences appear. The model reports what the documents say, ordinary or
  not; code adds only the label.
- **"No documents" findings written by code, from the document headers.** When no document
  names an item, code adds the finding "<item> — no documents on this item."; the model never
  writes it. In v4 the model wrote such labels and used them inconsistently (as a heading before
  history text, or as a "no notable issues" ending after a demand surge, eval_16).
- **Figures are part of the analysis and stay in the text** ("a 16-week gap"), quoted as given
  and checked rather than avoided: code compares every figure in the text with the input, and
  the judge checks the rest.
- **Questions: what the buyer needs to find out to place this order well, from the stock
  position, the history or both; none whose answer would not change the order; [] if there is
  nothing, and the report then says "No open questions for this order."** v4 left the list empty
  in some untroubled scenarios (eval_01) and asked courtesy questions in others (eval_03). The
  section is read as "answers needed before ordering", so courtesy questions would bury the real
  ones; routine confirmations (availability, lead time, terms) are already requested in every
  supplier email. Sources are not restricted: a High item due for ordering within a week (eval_01,
  RubberSeal) needs a lead-time check although its history is unremarkable.
- **No "do not do this" examples in prompts.** A rule illustrated with the behaviour it forbids
  tended to bring that behaviour on (a "which items were affected" wording spread to unrelated
  questions; v4's example sentence was copied into item notes). Rules are stated as principles.
- **A supplier-wide problem that names no items is asked about in terms of this order**
  ("how does it affect the items in this order?"), whatever the problem is — quality, delay,
  price: a price rise from a raw-material shortage, or a strike at one warehouse, can hit some
  items and not others.
- **Actions as one list with no cap, each tagged with the items it concerns.** A cap would
  drop or merge actions for a supplier with many problems. Actions are not tied to fixed slots
  (per finding, per item): many actions span several items, or come from the stock position and
  the history together. Code checks that every Critical item has at least one action that cites
  a stock finding about it.
- **Summary: most urgent first, at most about 100 words; the first sentence names every Critical
  item and the longest gap; detail is left to the sections below.** A fixed sentence count was
  rejected (three sentences cannot hold a supplier with several problems); no limit was
  rejected (half a page). 100 words holds the worst case — a supplier has at most seven items,
  so naming all seven takes 15–20 words. The summary's word count is kept in the run record to
  see how often the limit is passed; nothing is cut.
- **The example in the prompt shows every case once and agrees with its own input** — a
  supplier-wide delay, an item with an incident, an item with ordinary history, an item with no
  document — after v4's example taught a wrong note (`prompts/CHANGES.md`).

- **Figures.** The pipeline's figures (available stock, weeks to stockout, expected stockout,
  suggested quantity, recommended dates) are quoted as given; other figures are not restricted.
  Not every figure the model writes can be verified, so they are handled by what an error would
  cost: the pipeline's figures are checked exactly by code; figures quoted from history are
  checked by the judge; incidental figures (counts, time phrases) are left to the judge.
- **Proposed order-date and quantity changes, in either direction.** A delay, a changed lead time
  or a change in demand can mean the pipeline's date or quantity should move, but the model must
  not quietly rewrite figures the code calculated. So a change is proposed in its own field with
  the pipeline's value, the shift in weeks (negative = earlier) or the factor, the proposed value,
  the reason, and the findings it rests on. Code checks the arithmetic
  (proposed = original moved by the shift, or original × factor rounded up), that the original is
  the pipeline's value, that the cited findings exist and concern the item, and flags a
  proposed date that has already passed. A new judge check (J6) asks whether the history supports
  the change and its size, and whether items the same reason applies to were left out. The report
  shows the changes in its Proposals section, original and proposed side by side, and marks the
  values in the tables. Changing an
  assumption (lead time, demand rate) and recomputing everything was considered; stating the
  original and proposed date or quantity is clearer for the buyer.
- **The prompt's example leaves the change fields empty; their format is described in the rules**
  ("-2 means two weeks earlier", "1.5 means 50% more"). How well a model proposes changes from
  rules alone depends on the model, and the analysis-model benchmark should measure that
  difference rather than have an example hide it.
- **Findings are numbered facts, and every question, change and action cites the findings it
  rests on (2026-10-22).** A finding states a fact from the documents or from the stock figures,
  without judgement; what it means for the order is in the questions, changes and actions that
  cite it. Facts from the stock figures are findings too, written by the model and marked
  "(stock position)" (e.g. "WidgetX runs out before a new order can arrive; the gap is 14
  weeks."), so an action that rests on the stock position has something to cite. Questions,
  changes and actions may overlap: a question can lead to a change, and a change can still need
  confirming with the supplier. Why: each judgement can be traced to its facts and each fact to
  its source, so the checks are in two steps — finding against the document (judge), and
  question, change or action against the findings it cites (judge; code checks that the cited
  ids exist and concern the items). Code can also list documents that no finding uses. Citing
  documents directly was considered; citing findings keeps the stock position citable and lets
  the report show one list of facts. A separate layer of "issues" between findings and
  questions, changes and actions was considered and not adopted: each judgement would be bound
  to one issue, and the issue sentences would repeat the summary.
- **Sources shown by code.** Each finding ends with the numbers of its documents ([1]) or
  "(stock position)"; each question, change and action ends with the findings it cites (F1,
  F2). The model gives ids; code renders them and resolves document ids to file names.

**Report layout (2026-10-22, revised the same day).** Facts first, then the judgements that
cite them:
- Header: "Snapshot date: <date> (N days before this report)" (the age only when older), then
  "Report date: <date>".
- Data Issues Requiring Attention, only when rows were excluded.
- Summary.
- Items: item, risk, weeks to stockout, order by, quantity; most urgent first.
- Findings: F1, F2, … — supplier-wide findings first, then items most urgent first; each line
  starts with the supplier or item name and ends with its source.
- Critical Questions: "Ask <supplier>:" then "Check internally:", numbered through, each ending
  with the findings it cites.
- Proposals: order-date and quantity changes, "pipeline value → proposed value", with the
  reason and the findings cited.
- Recommended Actions: numbered, each with the findings cited.
- Item Details: one Measure | Value table per item (available stock, weeks to stockout, lead
  time, order by, deliver by, suggested quantity), Low items included.
- Referenced Documents: the search window and limits, then a numbered table.
- Run id at the foot.
"Critical" is shown in bold red in tables and headings (not inside the model's sentences, where
the word can have other meanings). History is searched as "documents dated <from> to <to> (last
12 months), at most 5 per supplier and per item". Every proposed change is shown as "pipeline
value → proposed value", including the risk level (High → Critical), with the whole "pipeline
value → proposed value" highlighted in yellow (not red, which would read as risk and clash with
Critical; the highlight survives black-and-white printing as a grey band, and the arrow carries
the meaning without colour). Finding numbers, document numbers and all figures come from code.
The earlier layout put history, proposals, actions and questions in each item card; side by
side they did not show which answered which problem, so the sections now follow the chain
facts → judgements, linked by finding numbers.

**Purchase request (internal approval document; 2026-10-22).** All documents are in
English; the prompts only state the register ("a professional company analysis report", "a
professional internal approval document"). The approval agent writes two parts, Purpose and
Justification; everything else is code. In order:
- The red italic draft note, then Critical Questions and Recommended Actions in red italic
  (see "The pipeline produces drafts"), without finding numbers: the request has no findings
  list for them to point to, and the buyer works from the analysis report.
- Header: "Purchase Request — <supplier>", request date, "Requested by: *[Your name here]*"
  (italic, for the buyer's name), and the analysis run it comes from.
- Purpose, Justification (approval agent).
- "Items Requested (snapshot <date>)": item, code, risk, weeks to stockout, quantity, order by,
  deliver by, with the same arrows and highlighting as the analysis report; then "Unit price
  and total: to be confirmed with the supplier" (no price data exists, so none is invented).
- Supporting Documents: the documents Justification cites, numbered in the order first cited.
- Approval block: Purchasing Manager, Operations Director, Finance Officer.

Items requested: Critical, High and Medium — everything due to be ordered within about 17 weeks
— and not Low, whose order dates are months away and which the analysis report still shows.

The approval agent receives, chosen by code: the supplier; the items requested with their risk
level and figures; the findings about the supplier as a whole or these items, each with the
numbers of its documents; the changes and actions for them. Not the summary (written for the
buyer, and likely to be copied), the questions (code puts them in the note) or Low items.
Justification covers only what the approvers decide on: why these items must be bought now,
why at these quantities and order dates, and any action that adds cost or changes how the items
are sourced; findings that bear on none of these are left out (the analysis report keeps them).
Facts are stated as far as the input states them, figures exactly and with no word that judges
them; an action's importance may be stated as far as its findings support; a check to be made
before ordering is written as a condition the request rests on, for the buyer to revise once
answered. A statement that rests on a document ends with its number ([1]); facts from the stock
figures carry no number, since the items table shows them. The agent also receives each item's
order-date timing ("overdue"). Code renumbers the cited documents for Supporting Documents.
Checks on the text, in the quality-check document: P1 (judge) adds nothing to its input, P2
(code) Purpose 1–2 and Justification 2–4 sentences, P3 (code) cites only documents given. The
tuning that led here is in `prompts/CHANGES.md`, v5.

**Critical questions split by who answers them (2026-10-22).** Each question is marked
"supplier" (the supplier can answer it, and it reveals nothing internal: no stock levels,
urgency, budget or analysis) or "internal" (the company must settle it). A single question can
mix the two ("does the price rise apply, and can the budget absorb it?"), so the split is made
when the question is written. The report shows them under "Critical Questions" as "Ask
<supplier>:" and "Check internally:"; the supplier email receives only the supplier questions.

**Supplier email (2026-10-22).** The email agent turns what the company needs to find out into
polite, conversational requests to a business partner, in one or two short paragraphs rather
than a list, one question per point, without mentioning what was reviewed. A point that comes
from a problem the supplier had before is asked about the related steps of this order as an
ordinary question, with no reference to what happened before or when: the analyst's question
("Has the customs documentation issue that delayed shipments in November 2024 been fully
resolved …?") becomes, in the email, whether the customs documentation for this order is
expected to proceed normally. The analysis report keeps the analyst's wording, since it is read
inside the company. The email no longer mentions history on its own: the old rule to "mention
history gracefully" produced "We noted your recent notice on pricing…", which stated a notice the
documents did not contain, asked nothing, and repeated the request for prices.
- Its input is chosen by code: the supplier, the items (name, quantity, requested delivery date),
  `points_to_clarify` (the analyst's supplier questions, described as internal notes and not
  wording for the supplier), the names of the Critical items, for which it also asks the
  earliest dispatch date and a faster shipping option, and `other_items` (items of this
  supplier outside the email, to be left out even where a point names them). No snapshot date
  (when the company reviewed its stock is internal), risk levels, stock figures, findings or
  internal questions. Findings as background were not needed.
- Only text in quotes in the prompt is copied into the email (the greeting and the two-line
  signature); everything else describes what the email must do, so its wording is not copied.
- The item lines are written by code and placed after the purpose paragraph; the agent writes
  no item list and no placeholder (a `{ITEMS}` placeholder was dropped or filled in by the
  model). Lines: "<item> — purchase quantity: <n>; requested delivery date: <date>", the quantity
  after any proposed change and highlighted when changed, with no arrow. No item code: suppliers
  use their own codes, not the company's SAP ItemCode. "Requested delivery date" (formerly
  "target receipt date", which reads as an internal target) is the pipeline's latest delivery
  date; whether to send it is left to the buyer when reviewing the draft.
- The validator checks the agent's text before the item lines are added, so purchase quantities
  never reach it.
- Model: `gpt-5.4-mini` for the draft and its rewrite. `gpt-4o-mini` and `gpt-4o` carried the
  analyst's wording into the email whatever the rule said, although `gpt-4o` explained the rule
  correctly when asked about the point alone. The steps and what was hard: `prompts/CHANGES.md`, v5.

**The pipeline produces drafts.** One run produces all three documents. The purchase request
opens with a red italic note: "Draft — to be revised once the questions and actions below are
resolved, with current quantities, dates and figures. Delete this note and everything in red
before submitting." Below it, also in red italic, the Critical Questions and Recommended Actions;
the body has no questions section, so approvers receive the revised
request without them. The note asks for current figures because the request is usually
submitted days after the analysis, when the stock has moved. The supplier email opens with
"Supplier email draft — review and edit internally before use, then delete this note." The
buyer gets the supplier's answers and edits the drafts; the formal purchase order is then issued
in SAP, outside the pipeline.

**Planned check changes (step 4 of v5).** C3 becomes "every figure in the text matches the
input"; J5 becomes a code check (each Critical item has a tagged action); J1b and J3 are no
longer judge checks, since code writes the no-history sentences from the headers.

**Process note.** While drafting v5 the prompt was tried on evaluation scenarios eval_01, 03,
08, 09, 11, 12, 13, 16, H2 and H4. That tunes the prompt to the scenarios later used to compare
models. Still to decide: a separate set of development scenarios for prompt work, keeping the
evaluation set for measurement only.

**Still to decide.** The purchase request's tone and language (any fixed company phrasing;
English or Korean).

### 17. Which history the analysis sees; no chunking *(decided 2026-10-19)*

- **Rule (agreed with the purchasing team):** documents dated within the 12 months before the
  snapshot date; within that, the 5 most recent about the supplier and the 5 most recent about
  each item. A document in several scopes counts once. The window is measured from the
  snapshot date, not today, so re-running an old CSV sees the history of that time. An undated
  document is placed by its upload date. Settings: `history_window_months`,
  `history_docs_per_scope`.
- **Why not "the 10 most recent":** document volume varies a lot by supplier — about one every
  two months for large suppliers, one every four to six months for small ones. Ten most recent
  reached back about 20 months for a large supplier but 3–5 years for a small one, bringing in
  long-resolved problems; and one shared count let a busy scope push out another (a supplier with
  many recent supplier-wide documents could leave an item with documents shown as having none).
- **Shown in the report:** Referenced Documents opens with the window and the limit, and, if any
  documents in the window were left out by the limit, how many.
- **No chunking.** History documents are at most two A4 pages (about 1,100–1,400 tokens) and are
  retrieved by metadata, never by similarity, so each is stored and given to the model whole.
  They had been split into 1,000-character chunks overlapping by 200 and joined back for the
  model, which repeated every overlap (35 repeated passages in one incident report, about 20 %
  more tokens). `scripts/unchunk_history_store.py` stores chunked documents whole without the
  overlaps; the migration from the earlier store now joins its chunks the same way.

### 18. The analysis is as of the snapshot date; stale stock data is refused *(decided 2026-10-22)*

- **One reference date: the snapshot.** Every judgement in the report (overdue, order-by dates,
  the 12-month history window) is relative to the date the stock figures are true. Recomputing
  from the run date was rejected: projecting stock forward from an old snapshot assumes nothing
  happened since (no deliveries, no orders placed, no change in sales), so a 6-month-old CSV
  would report stockouts for items that had been reordered and delivered meanwhile.
- **A run accepts a snapshot up to 3 days old** (`require_fresh_snapshot`,
  `max_snapshot_age_days = 3`), and refuses older ones with a request for a current export.
  Stock moves every day (outbound runs at weekends too) and the snapshot is taken nightly at
  23:00. A refused run also signals that the nightly snapshot job has stopped.
- **The report shows the snapshot date first, with its age in plain text** when it is older than
  the report ("Snapshot date: 2026-10-20 (2 days before this report)", then "Report date:
  2026-10-22"). The run id, which carries
  the actual run time, stays in the run record and at the foot of the report for tracing.
- **Runs on fixed past data measure the age against a fixed run date.** `run_date` is set to
  the day after the sample CSV's snapshot (2025-04-05 → 2025-04-06), so the check stays on and
  every date in a demo report stays consistent without regenerating anything; in production
  `run_date` is unset and the age is measured against the actual date of the run. The
  evaluation scripts call the analysis directly with their own fixed snapshot date (2026-03-02)
  and do not pass through the check. Rolling demo dates (snapshot set to today, history dated
  relative to it, reloaded daily) was considered and dropped as more machinery than the demo
  needs.
- Report wording: "expected stockout if ordered today" is not shown; the buyer judges from weeks
  to stockout and lead time. Changes proposed by the analyst are shown with an arrow
  (pipeline value → proposed value) wherever they appear, including the risk level
  (High → Critical).

### 19. Documents beside the evaluation; time limits on model calls *(2026-10-22)*

- **The documents no longer wait for the judge.** Nothing in the documents uses the evaluation's
  result. A first wiring started the evaluation and the report side by side, but LangGraph moves
  in steps and a step ends only when all its nodes have finished, so the purchase request and the
  email still waited for the judge (measured: report done at 5.9 s, purchase request at 189.5 s,
  after the evaluation). The documents are now one node of the main graph, a graph of their own
  inside it (report → purchase request → its check beside the email steps), run beside the
  evaluation; a final node joins them and adds the purchase request's checks to the
  quality-check document. Measured with a 20-second stand-in for the judge: documents done at
  11.0 s, evaluation at 26.5 s. The run still returns when the evaluation is done; sending the
  documents first needs a change to the API and the web page (in "Still to do").
  *Superseded the same day (item 20):* once a blocking judge check could send the analysis back
  for a rewrite, the documents had to wait for the evaluation again; with the judge now taking
  about 9 s, a whole run took 34.4 s.
- **Every model call has a time limit** (`config.py`): analysis, approval and email agents 60 s
  with one retry (`llm_timeout_seconds`, `llm_retries`); judge requests 90 s with one retry
  (`judge_timeout_seconds`, `judge_retries`), after which the checks are recorded as "Not
  checked" and the verdict is "needs review". Without limits a run hung for more than five
  minutes while the judge did not answer; with them the same run ended at 194 s, the judge
  having returned 504 (deadline exceeded) twice and 429 (quota exhausted) once. The offline
  scripts keep two judge retries.
- `gpt-5` models take only their default temperature, so no temperature is passed to them.
- **Judge switched to `claude-haiku-5-5` for now (2026-10-23)**, until judge validation chooses
  one. `gemini-3.7-flash` took 30-109 s per call on 2026-10-22, then returned 504 (deadline),
  503 (unavailable: high demand) and 429 (quota). Claude is from a different provider than the
  analysis and approval agents (OpenAI), as the rule requires; a first call answered in 3.9 s.
  The deploy workflow passes `CLAUDE_API_KEY`, which needs a GitHub secret.
- **Development runs are kept** (`dev/`): development cases, separate from the evaluation set so
  prompts are not tuned on the scenarios used to measure them, and one folder per run with the
  exact prompt text used, since prompts are edited in place while a version is written
  (`scripts/run_dev_case.py`). Until 2026-10-22 test runs were kept only in a session's
  working folder; that day's runs were moved in, without their prompt text.

### 20. Analysis rewrite loop; check notes in the draft; J6 *(2026-10-23)*

- **The analysis is corrected before any document is written.** If it fails a blocking check, the
  analysis agent gets its output and the failures (`prompts/v5/analysis_revision_request.md`;
  for a judge check, the judge's evidence) and corrects only those, up to `analysis_rewrites` (2)
  times in all. The severity decides, not who checks: a blocking code check (C1-C6, C9) or a
  blocking judge check (J1, J6a-c, J7) that fails sends the analysis back; a warning never does;
  a judge check that could not be answered does not either (there is nothing to correct). Code
  checks run first: while one fails and a rewrite is left, the judge is not called, as the
  analysis will change. In the graph: analysis → evaluation → (rewrite → evaluation)* →
  documents → finish. Every attempt, its checks, whether the judge ran and the feedback are kept
  in the run record (`analysis_attempts`). (A first version, the same day, let only code checks
  decide, so that the documents need not wait for the judge; it was changed once the judge,
  Claude Haiku, took about 9 s.) Runs (`kestrel_components`): an order date moved before the
  snapshot date (C5) was corrected on the second attempt each time, the other changes kept; a
  stand-in judge failing J6b once sent the analysis back with the judge's reason and the next
  attempt passed; a whole run took 34.4 s.
- **The rewrite returns the whole analysis, not only the parts to correct (decided 2026-10-23
  after a trial).** Returning only the fields it changes was tried: each rewrite took 1.4-3.5 s
  instead of 4.9-7.3 s, so about 3-4 s were saved per rewrite. But the findings are what the
  questions, changes and actions cite: in `nordic_fasteners` the third attempt rewrote the
  findings alone, the citations then pointed at the wrong findings, and three blocking checks
  failed (C4, C6, J6a) where the first attempt had failed one. Keeping parts consistent would
  have needed a rule for every failure saying which fields to rewrite with it, and would still
  have carried the risk. A few seconds per rewrite were not worth that, so the rewrite returns the
  whole analysis again (`dev/runs/2026-10-23_173030_*`, `173106_*`, `173433_*`).
- **When the rewrites are used up and it still fails, the documents use the attempt with the
  fewest blocking failures**, not the last one, since a later attempt can be worse.
- **The quality-check document is a checklist with reasons, kept for every run** (also for the
  README and portfolio): the run id and snapshot date; an Attempts table (each attempt, whether the
  judge ran, the blocking checks that failed, whether it was sent back, and which attempt the
  documents use); then each check with its severity, result and reason (the judge's evidence for
  judge checks), for the analysis and for the purchase request. It is saved with the run's id in
  its file name, and the same results are in the run record.
- **Checks split and added (2026-10-23).** J1 into J1a (found in the documents), J1b (not
  distorted), J1c (not misattributed); J7 into J7a (stock figures match) and J7b (changed figures
  written with both values); P1 into P1a (nothing added), P1b (figures as given), P1c (meaning
  kept). New: J8 questions marked for the supplier reveal nothing internal (blocking), J9 each
  question and action follows from its findings, J10 no document with a problem for this order
  left unused (code lists the documents behind no judgement; the judge reads those), J11
  findings state facts only (warnings). J2a and J2b stay until J10 has shown it can replace them.
- **The Anthropic judge's tool names each check as a property.** With open property names Haiku
  sometimes put every verdict into one field as a string, so the purchase request checks came back
  unanswered; with the check ids as named, required properties, four runs in a row were answered.
  When a verdict still cannot be read, the start of the judge's answer is kept as its reason.
- Seen once after a rewrite: an order-date change of 0 weeks (2025-04-05 → 2025-04-05) left in
  place of a removed change; it passes every check but adds an empty line to the proposals. To do:
  drop such changes in code or catch them in a check.

- **What is still wrong goes in the purchase request's red draft note, in plain words.** Once every
  check has run, the note lists "Must be corrected before use" (blocking checks not passed) and
  "Please check" (warnings), without check ids: a fixed sentence per code check, and for judge
  checks a one-sentence note the judge writes for the buyer when a check fails. The analysis
  report gets no note: it is built from the analysis after its rewrites, and the full results are
  in the quality-check document. Warnings never cause a rewrite.
- **Judge answers made reliable.** The judge writes its evidence before its verdict (with the
  verdict first, Haiku twice concluded "pass" in its evidence and returned false); the Anthropic
  judge answers through a tool, so the JSON is built by the API (a quote inside a cited sentence
  had broken it); a verdict sent as a JSON string or wrapped once more is read. J7 judges the prose
  only (Haiku had applied it to the change fields).
- **J6a-c, whether proposed changes hold** (blocking, when any change is proposed): direction,
  size, and no item the same reason applies to left out (an order date that cannot move earlier
  than the snapshot date excepted). First run: all three judged correctly, including the overdue
  item as the allowed exception.

### 21. Dates through the lead time: the analysis judges one figure, code works out the rest *(decided 2026-10-24, to be built)*

- **What led here.** The analysis proposed order-date changes itself (weeks earlier or later). For
  a supplier-wide notice of lead times four weeks longer it moved every item four weeks earlier,
  including one already overdue, giving a date in the past (caught by C5); after a rewrite it
  left a change of 0 weeks with the reason "lead times four weeks longer", which the judge passed
  (it checked the four items that moved and read the overdue one as the allowed exception). And
  the risk levels stayed as if nothing had changed, although a longer lead time is what makes an
  item run out before a new order can arrive.
- **The decision.** Every finding about dates comes down to one thing the analysis judges: the
  lead time, for the supplier as a whole or for named items (`lead_time_changes`: items, weeks,
  reason, findings). Code then works out everything that follows from it. The analysis prompt must
  say so plainly: the analysis changes only the lead time, and code calculates and fills in the
  order dates and risk levels (otherwise the model also tries to move the dates).
- **Risk level from weeks to stockout and the lead time** (default 18 weeks = 16 import + 2
  internal):
  - Critical: weeks to stockout < lead time (runs out before a new order can arrive)
  - High: weeks to stockout < lead time + 8
  - Medium: weeks to stockout < lead time + 17
  - Low: otherwise
  With 18 weeks these are the current thresholds (18, 26, 35). Latest order date = latest
  delivery date − lead time. Weeks to stockout, the latest delivery date and the suggested quantity
  are not changed by a lead-time change (the stock runs out on the same day, and the quantity
  rests on the weekly sales, stock ÷ weeks to stockout).
- **The report shows the lead time** (Items and Item Details), with "18 → 22 weeks" highlighted
  when it changes, so a risk level that rises has its reason beside it; the findings state the
  lead-time fact the change rests on. A changed risk level and order date are shown as
  "pipeline value → new value", highlighted.
- **An order date that would fall on or before the snapshot date** is shown as the snapshot date,
  highlighted like any change, with an asterisk; below the table, in red: "* <item> should have
  been ordered by <date>; it is overdue."
- **Checks split by who can judge what.** The judge checks whether each lead-time change holds:
  its direction, its size, and no item the same reason applies to left out (J6). Code checks that
  the changes are well formed (not 0 weeks, items that exist). The order-date arithmetic checks
  (C3 and C5 for order dates) are no longer needed: the dates are calculated by code from the lead
  time, so they cannot be wrong, and a proposed date before the snapshot date or a change of 0
  weeks can no longer come from the model.
- **The ways it could have been solved, and the one chosen** (for the README and portfolio):
  1. Let the model keep moving order dates and add rules: no date before the snapshot date, no
     change of 0 weeks. Each rule patches one symptom; the model still does date arithmetic, the
     judge still has to follow it, and the risk levels still ignore the delay.
  2. Keep the dates as they are and have the model say in an action that an overdue item's gap
     grows. Simple, but the tables and risk levels stay as if nothing changed, and the reader has
     to work out the effect.
  3. Have the model propose the new risk levels as well. More for the model to get right and the
     judge to check, and risk levels are a formula, not a judgement.
  4. **Chosen:** the model judges one thing, the lead time (the supplier's expected one and each
     item's final one); code derives the order dates and risk levels from it with the same formula
     the pipeline already uses. The judgement is narrower and easier to check, the figures are exact
     by construction, a date in the past or a change of nothing cannot come from the model, and the
     report shows the cause (the lead time) next to its effects (risk, order date).
- Each item's final lead time weighs the supplier-wide reason with the item's own and says whether
  they add up or are one; it is written per item (items with the same value may share an entry),
  and every item gets one when the supplier's lead time changes, so no item is passed over.
- **J7b no longer judges the changes' reasons** (decided 2026-10-23). In the Kestrel run the
  judge failed J7b although the prose named neither 22 weeks nor 7,800 (it mentioned only "4
  weeks" and "50%"); the rewrite then appended "(18 → 22 weeks)" and "(5200 → 7800)" to the
  reasons, and each proposal line showed the pair twice, once from code
  ("GasketKit quantity 5,200 → 7,800 (×1.5): ...") and once in the reason. A reason is always
  printed right after the pair code writes, so asking for the pair there only adds a duplicate;
  the summary, findings, questions and actions are still judged. C9 (arrows match the proposed
  changes) still reads the reasons.
  The analysis prompt's pair rule was narrowed to the same parts, saying the reason needs no pair.
- **An order date on the snapshot date counts as overdue** (decided 2026-10-23): the footnote stays
  "it is overdue" whether the date is before the snapshot date or on it, since an order due the
  day of the analysis leaves no time to negotiate with the supplier or get the request approved.
- **The quantity line keeps "(×factor)"** (decided 2026-10-23): it is printed by code from the
  factor the analysis already writes (and C5 already checks), so it costs no model output or check.
- **Fixed: the purchase request input's order timing** now follows the new order date. It had kept
  the timing worked out at 18 weeks ("within 4–8 weeks" for GasketKit, whose order date had moved
  to two weeks after the snapshot date), which sat beside the new date in the agent's input; the
  Kestrel request then called GasketKit's order overdue.
- Still to decide: how the proposals list shows an order-date change (only "old → new", or with the lead time as its
  cause).

### 22. The analysis in two steps: the changes, then the text from the figures after them *(built and tested 2026-10-23; replaced by item 23 on 2026-10-24)*

- **What led here.** After item 21 the analysis judged the lead time and code recalculated the
  figures, but the same answer had already written its text with the figures at 18 weeks. In the
  Kestrel run (`dev/runs/2026-10-23_182404_…`) the model set the lead time to 22 weeks and, in the
  same answer, wrote "the gap is 6 weeks" (18 − 12) in the stock finding, the summary and the
  action that bridges it; at 22 weeks the gap is 10 weeks. Code then gave the purchase request's
  agent the recalculated items (gap 10) beside the finding's "6 weeks", and the Justification
  wrote "a 6-week stockout gap caused by a 4-week extension". The checks could not catch it: J7a
  compared the 6 with the 18-week input and found it right, and P1a found the 6 in its input.
- **Not caused by item 21; shown by it.** Before item 21 nothing was recalculated, so the tables,
  the text and the request all said 6 weeks: consistent, and wrong. Recalculating in code made
  the old figure contradict the new one. The cause is the order: the text that rests on a
  judgement was written before the figures the judgement changes existed.
- **The decision: two calls, in the order the judgements depend on each other.**
  1. Step 1 (`prompts/v5/analysis_changes.md`): the pipeline's figures and the history
     documents in; the findings from the documents and the changes out (`supplier_lead_time`,
     `lead_time_changes`, `quantity_changes`).
  2. Code: each item's figures after the changes (`documents.items_after_changes`: lead time,
     risk level, latest order date and its timing, gap, proposed quantity).
  3. Step 2 (`prompts/v5/analysis_actions.md`): those figures, step 1's findings and changes in
     (not the documents: the findings are its only source of history); the stock findings, the
     questions, the actions and the summary out, numbering its findings on from step 1's.
  The two outputs are joined into one analysis (`quality_checks.merge_steps`), so the documents
  are built as before.
- **Checks per step**, each step with its own rewrite loop (up to `analysis_rewrites` each); step 2
  is written only once step 1 has passed or used up its rewrites:
  - Step 1: C1 (its fields), C2, C3, C4, C5, C9; judge J1a-c, J6a-c, J7a (figures in the reasons,
    against the pipeline's figures), J11.
  - Step 2: C1 (its fields), C2, C4 (also: no finding id already used), C6, C7, C8, C9; judge
    J1a-c, J2a-c, J7a (against the figures after the changes), J8, J9, J10, J11. The judge is
    shown step 1's output as context and told to judge only what step 2 wrote.
  - The quality-check document has a checklist per step and a Step column in the Attempts table;
    the purchase request's red note takes both steps' results.
- **J7b dropped.** It asked for a changed figure to be written as "pipeline value → proposed
  value", because the text was written from the pipeline's figures and the reader could not tell
  which one a figure was. Step 2 writes from the figures after the changes, which are those in the
  tables, so a lead time of 22 weeks or a quantity of 7,800 in the text is simply right, and J7a
  checks it against them. A stray figure is still caught: J7a compares every stock figure with
  the figures after the changes. This also ends the J7b judgement that went wrong (item 21: a
  change's size, "4 weeks", read as a changed figure).
- **Results** (one run each, `dev/runs/2026-10-23_190749_…` Kestrel, `…_190825_…` Nordic):
  - Kestrel: the gap is 10 weeks in F2, the summary and the action; both steps passed at the
    first attempt; ready, one warning (J2b: the GasketKit demand rise gave an action, not a
    question). 29.1 s in all (37.6 s with one call, which also had a rewrite).
  - Nordic: step 1's first attempt set every lead time to 22 weeks for a customs hold that the
    supplier says it has fixed; J6b failed it ("a one-off with a stated fix, added fully to
    every item"), and the rewrite dropped the change. Step 2 passed at once; ready, no warnings.
    33.4 s with that rewrite.
  - Cost per supplier: step 1 about 2.9k input and 0.5k output tokens, step 2 about 3.3k and
    0.35k (one call was 3.9k and 0.85k), and two judge calls instead of one (about 4.2k + 5.2k
    input tokens instead of 5.8k). Each call is shorter, so the time did not grow.
- **Still open** (not changed):
  - The purchase request: Kestrel's Justification still called GasketKit's order overdue (its
    input says "within 2 weeks") and wrote "a 10-week stockout gap caused by a supplier-wide lead
    time increase of four weeks" (6 of the 10 weeks were there at 18 weeks). Its input is now
    consistent, so this is the approval agent's own error; P1a flagged it as a warning.
  - Step 2 recommends reviewing a quantity that step 1 has already raised ("Review the GasketKit
    order size to ensure it meets the increased demand"; Nordic: "evaluate whether the WasherSteel
    order should be adjusted further").

### 23. One call again, the work in the order of the fields, and a newer analysis model *(2026-10-24)*

- **Why the two steps were dropped.** They added a judge call (J1, J7a and J11 judged twice, about
  60% more judge input), lost context between the calls (step 2 asked to review a quantity step 1
  had already raised) and still gave the model a smaller view than one call. The cause of the stale
  "6-week gap" (item 22) was in the one-call prompt itself: it told the model to quote the input's
  figures and "not figures you calculate", and the summary came first in the output, before the
  lead time was decided.
- **The prompt** (`prompts/v5/analysis.md`):
  - the output fields in the order of the work: findings from the documents, the changes,
    `figures_after_changes`, `stock_findings`, questions, actions, summary last (the report is
    assembled in code, so the field order need not follow it; code joins `stock_findings` into
    `findings`, `services/agents.py` `from_model` / `to_model`);
  - a change's consequences stated as such: a changed lead time changes that item's risk level, gap
    and latest order date; the input's 18-week figures no longer hold for it; work them out again
    and use them in everything written after the changes;
  - the new figures under their own names (`new_lead_time_weeks`, `new_risk_level`, `new_gap_weeks`,
    `new_latest_po_date`), apart from the input's fields;
  - the order-date rule names the fields it uses.
- **C10 (blocking, new):** code works out the same figures by the same rules and compares them one by
  one. J7a compares the text with the figures after the changes.
- **What GPT-4o did** (Kestrel, four prompt revisions, one run each): wrong first answers every time
  (7, 4, 4, 8 of 20 values), mostly order dates; right after one or two rewrites, but only because
  C10's message gives the right values. Naming the fields and separating the names did not help. A
  probe showed it understood the rule and failed at writing the date without working
  (`docs/analysis_model_comparison.md`).
- **The model comparison** (`docs/analysis_model_comparison.md`): the same prompt with
  `gpt-6.1-sol`, `gpt-6-luna` and `claude-sonnet-5-5` on both cases: every figure right at first, no
  rewrites, the fixed customs hold not taken as a longer lead time, the defective batch raised.
  Luna costs about 1/25 of GPT-4o per analysis. After a second round (seven-item Kestrel and
  Nordic, two runs each), **`gpt-6.1-sol` was chosen (2026-10-24)**: the three newer models were
  equally accurate once the judge's errors are set aside, and Sol's questions and actions were the
  most specific, with dates and deadlines, which is what the analysis is for; it costs about ten
  times Luna and two thirds of GPT-4o.
- **No more "no documents on this item" lines in Findings** (decided 2026-10-24). Findings are the
  meaningful, important facts taken from the documents searched; that an item has no documents is
  not one. Code had added the line for every item no document names (three of eleven findings in
  Kestrel, each with its own number), whatever the model. Removed from `documents.number_findings`
  and from the analysis prompt. Findings that a document reports nothing wrong (steady demand) stay.
- **Stock findings only where a question or action rests on them** (2026-10-24). Sol wrote one for
  every item, including those with nothing to report ("OilSeal ... remains Low risk with no projected
  gap"), repeating the Items table. The rule now says so plainly and that an item whose figures call
  for nothing gets none. One run each with Sol (`dev/runs/2026-10-24_020443_...`, `_020540_...`):
  Kestrel 7 -> 3 stock findings (BearingCap, ThrustWasher, ValveSeat due today), Nordic 2 -> 1, every
  one cited by an action; both ready at the first answer; questions and actions as specific as
  before. Kestrel's internal question on whether the GasketKit forecast already includes the
  contract's demand, asked in three of the four second-round runs, did not appear (Nordic's did);
  one run cannot tell whether that follows from the change. Dates in the text: either form is fine as
  long as the text is consistent in itself (the user, 2026-10-24).
- **A supplier may be asked about expedited or partial delivery** (decided 2026-10-24). J8 failed a
  Sol question asking what partial-delivery and expedited-shipping options the supplier could offer
  for BearingCap and ThrustWasher, "with quantities and arrival dates", as revealing that the order
  is urgent. When an item will run out before a new order can arrive, asking whether it can come
  sooner is unavoidable, since not asking risks a larger loss; and asking it says that much by
  itself. J8 and the analysis prompt now allow it; stock levels, weeks to stockout, how short stock
  is, budget and the analysis stay internal.
- **Findings in two groups** (decided 2026-10-24): "From the documents", then "From the stock figures
  (after the proposed changes)", each sorted supplier first, then items by risk after any lead-time
  change; numbering runs on across both, so citations stay unique; "(stock position)" after each
  stock finding is dropped, the heading says it. Before, stock findings sat among the document
  findings of the same item.
- **C11 tried and dropped (2026-10-24): figures in prose stay the judge's (J7a).** C11 had code read
  the stock findings' risk level, gap, lead time, weeks to stockout and dates with patterns and
  compare them with the figures after the changes. It caught GPT-4o's "GasketKit gap is 2 weeks"
  (0 after the change), but it reads free text by guessing from wording: it misread "runs out 2
  weeks before a new order" as weeks to stockout and sent the analysis back for nothing, and it
  passes any wording it has no pattern for ("a ten-week gap", "a shortfall of 10 weeks"). Which
  figures to check cannot be told from wording reliably, the same is true of every prose part
  (summary, questions, actions), and a pattern list cannot be shown to be complete. So all figures
  in prose are J7a's, as designed; the judge's reliability on them is measured in judge validation
  with seeded faults (`docs/judge_validation.md`). C10 stays: it compares fixed fields, not prose.
- **Kestrel case: two items whose risk rises with the lead time** (ThrustWasher, High at 18 weeks and
  Critical at 22; SpringClip, Medium and High). In the five-item case every item had the same risk at
  18 and 22 weeks, so a risk worked out at 18 weeks could not show. Runs before 2026-10-24 01:09 have
  five items.
- **Fixed: the judge saw only the figures after the changes** (introduced with C10 for J7a). It then
  read the proposed lead time as the input's and passed GPT-4o's 22-week lead time for Nordic's fixed
  customs hold ("the input lead times are 22 for all items, consistent with the input", J6b). It is
  now shown the pipeline's 18-week figures and, apart, those after the changes; on the rerun J6b
  failed the change as it should.
- **GPT-4o on the seven-item Kestrel and on Nordic** (`dev/runs/2026-10-24_011224_…`, `_011306_…`):
  Kestrel ready after one rewrite (first answer: ThrustWasher's risk and gap at 18 weeks, four order
  dates wrong). Nordic needs review: the first two answers raised every lead time to 22 weeks for the
  fixed customs hold; the third dropped the change but kept "WasherSteel runs out 2 weeks before a new
  order can arrive", a figure at 22 weeks, caught by J7a (and by C11, since dropped) with no rewrite
  left.

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
- **No score; severity and staged comparison (2026-10-23)** (`docs/eval_rubric.md`). The first
  version's 9.5/10 drew the question of what it measured, and weights would draw the same
  question. Each check is blocking or a warning; the production evaluation node marks an
  analysis ready only if every blocking check passes and lists warnings; models are compared in
  stages — blocking checks first (a system that fails here is not chosen), then warnings, then
  cost and response time. The same checks serve fine-tuning: the fine-tuned model is compared
  in the same stages, teacher outputs must pass every blocking check to become training data,
  and passing and failing outputs can form DPO pairs. Which check is blocking is still to be
  decided, together with the v5 revision of the checks.

## MCP servers

`mcp_servers/` (history and replenishment servers, Claude Desktop config example). Docs
to be written after testing in Claude Desktop.

## Still to do

- Judge validation: hand-score about 10 originals, run 8 candidates × 3, choose the
  production judge and the offline panel.
- Refactor `scripts/eval_v2_score.py` onto `services/quality_checks.py`.
- Analysis-model benchmark (GPT-4o against 7 others on quality, cost, latency).
- **Limits on proposed changes (decided 2026-10-23, not yet in the prompt).** A proposed order
  date is never earlier than the snapshot date, and a proposed quantity is never below zero (the
  factor is above 0). Seen in `dev/runs/2026-10-23_013618_arrows-in-the-text-c9-j7-for-changed-figures`: for a supplier-wide
  lead time four weeks longer, the analysis moved every item's order date four weeks earlier,
  including BearingCap, whose order date was already the snapshot date (overdue), giving
  2025-03-08, a date already past. C5 caught it (blocking). To do: state both limits in the
  analysis prompt's rules for `po_date_changes` and `quantity_changes`, and add the quantity
  limit to C5, which checks only the date today.
- **SFT retraining:** the fine-tuned model was trained on raw CSV rows, source risk levels
  with no Critical, and GPT-4o outputs from the old prompt. Regenerate the teacher data with
  the current pipeline and retrain before comparing it with GPT-4o. Keep only teacher outputs
  that pass every blocking check; keep the five holdout scenarios out of the training data;
  check how `scripts/generate_dpo_pairs.py` builds pairs today before using the checks for
  them; compare the result in the staged comparison, fp16 and fp8 serving included.
- Email validator wrongly flags proposed purchase quantities as internal stock figures, and
  a failing email is still sent after three rewrites.
- Email validator: what it is given to judge what is internal. Today it gets general
  categories (stock quantities, weeks to stockout, risk levels, analysis logic). Options: give it
  this run's actual internal data (available stock, weeks to stockout, risk levels, finding
  sentences, internal questions) to compare against, and have code check that no such figure
  appears in the email, leaving the validator the paraphrased leaks ("stock is running low").
  Since v5 it checks the email text before code adds the item lines, so purchase quantities no
  longer reach it.
- Undecided: whether to call the model at all when every item is Low; whether the PR
  document also gets the data-issues section.
- **Later: a second stage for the purchase request** (designed 2026-10-22, not built). The
  request is submitted days after the analysis, after the supplier has answered. Design: the
  analysis report gives each critical question an "Update" and "Source" field, plus "Other new
  facts"; the buyer fills them; a purchase-request run reads them with a current CSV, stores
  each answer as a history document (headers written by code, linked to the first run's id),
  runs the analysis again and writes the request from it. Updates can be wrong or vague; they are
  treated like any history document — kept with their source, quoted without strengthening, and
  checked by the judge against their text. Open: a docx round trip or a web form; whether answers
  count towards the 5-per-scope history limit. To be built after judge validation.
- **v5, before judge validation** — in progress; the design decisions and their reasons are
  in item 16 above. The problems found in v4 that led to it:
  - Critical questions: none when there is nothing to raise. Questions come only from a
    problem in the history or from a Critical item's mitigation. GPT-4o now leaves the list
    empty in some untroubled scenarios (eval_01) and invents neutral questions in others
    ("Can we expect … to maintain steady delivery?", eval_03).
  - Item notes: exactly one of a summary of the documents about the item, or the sentence
    "History reviewed: no notable supplier or item issues.", never both and never as a
    prefix. GPT-4o appended it after a demand surge (eval_16, 716001), contradicting the
    note, and used it as a heading (H2, PartW).
  - Analysis report layout: the prompt lists what the summary must contain but not its
    form, so headings (`#` / `###`, `## Summary` / `**Summary:**`) and the history part
    (separate supplier and item sections, one combined section, prose, or none) vary from
    scenario to scenario. Either fix the layout in the prompt (A), or have the model fill
    structured fields (Critical items, supplier findings, findings per item) and render the
    markdown in code (B, always consistent; changes the output format, C1, the report
    agent's input, the judge's input and the SFT format). The buyer-facing report is
    rewritten by the report agent into fixed sections, so buyers see this less; the judge's
    J5 (Critical items in the summary) depends on where the summary is.
  - Underlying issue: the report is written twice. The analysis agent writes a free-form
    markdown report, and the report agent rewrites it into the buyer-facing one, so
    figures pass through two LLMs while C3 checks only the analysis agent's timeline, and
    the loose markdown also feeds the judge, the purchase-request agent and the email
    agent. Direction for v5: the analysis agent returns structured judgements only
    (Critical-item mitigation, supplier findings, findings per item, questions); code
    renders the figures table from the preprocessing; the report agent writes the prose
    once. Affects the analysis output format, the report agent prompt, the PR and email
    inputs, C1/C3/J5 and the SFT format.
- Purchase request has no prices: there is a Finance sign-off but no unit price or total,
  because the CSV carries none. A query for the last purchase price from SAP Business One
  would supply it.
- **README and portfolio: what moved from the LLMs to code** — a key point of the design: LLMs
  only judge and write for a different reader; whatever code can fill in or verify, code does,
  which makes the output exact, checkable and cheaper. Write it up with examples:
  - Filled in by code, and where:
    | What | Where |
    |---|---|
    | Risk level from weeks to stockout | `services/item_grouping.py`, `risk_level_from_weeks` |
    | Latest order and delivery dates | `services/item_grouping.py`, `build_recommendations_for_item` |
    | Suggested quantity (26 weeks of cover) | `services/item_grouping.py`, `compute_suggested_quantity_at_latest_delivery` |
    | Rows that cannot be analysed, and the data-issues section | `services/item_grouping.py`, `screen_rows`, `data_issue_lines` |
    | The history the model sees (headers, 12-month window, 5 per scope) and its document numbers | `services/vector_store.py`, `search_history`; `services/agents.py`, `retrieve_history`, `history_for_model` |
    | The analysis report: every table and figure, the order of sections and items | `services/documents.py`, `render_analysis_report` |
    | Finding numbers, their sources, and the "no documents" findings | `services/documents.py`, `number_findings` |
    | The purchase request: note, header, items table, Supporting Documents (renumbered), approval block | `services/documents.py`, `render_purchase_request`, `renumber_citations` |
    | What the approval and email agents are given | `services/documents.py`, `approval_input`, `email_input` |
    | The email's item lines and where they go | `services/documents.py`, `email_item_lines`, `render_email` |
    | Bold, italic, highlight and red in the .docx files | `utils/docx_utils.py`, `_add_runs` |
  - Verified by code, and where:
    | What | Where |
    |---|---|
    | C1-C6: format, items, figures, citations, change arithmetic, a stock-based action for each Critical item | `services/quality_checks.py`, `code_checks` |
    | P2, P3: purchase request length and cited documents | `services/quality_checks.py`, `approval_checks` (P3 from `renumber_citations`) |
    | Stock data no older than 3 days | `routers/pipeline.py`, `_require_fresh_snapshot`; `config.py`, `run_date` |
    | Time limits on every model call | `services/agents.py`, `_llm`; `services/quality_checks.py`, `_judge_timeout` and the provider calls; `config.py` |
    | Documents beside the evaluation | `services/agents.py`, `documents_graph`, `documents_node`, `purchasing_graph` |
  - **The supplier email as an example** (worth its own paragraph in the README and portfolio):
    code assembles the whole email — the draft note, greeting, a fixed purpose sentence, the item
    lines (quantities after any proposed change, highlighted), a fixed closing and the signature —
    and the email agent writes only the requests: it turns the analyst's questions marked for the
    supplier into polite questions to a business partner, without naming past problems or items
    outside the email. Its input is only what that needs (`documents.email_input`); its output is
    one JSON field; the validator reads only that field. The way there, and why every fixed part
    was taken away from the model, is in `prompts/CHANGES.md`, v5, steps 1-26. Where:
    `services/documents.py`, `EMAIL_PURPOSE`, `EMAIL_CLOSING`, `email_parts`, `render_email`.
  - **Deciding in code what the agent asks, so it never sees why** (also for the README and
    portfolio): items about to run out (Critical) need the supplier asked for the earliest dispatch
    date and a faster shipping option, but the supplier must not learn that stock is short. In v4
    the email agent received each item's risk level and applied "if Critical, ask…" itself; in v5
    code picks the Critical items and passes only their names (`ask_earliest_dispatch`), so the
    agent knows which items to ask about but not the risk behind it — one less way for internal
    judgement to reach the supplier, and the choice of items is exact rather than left to the
    model. The same idea: `other_items` names the items outside the email to leave out, and the
    snapshot date is not passed at all. Where: `services/documents.py`, `email_input`.
  - **Failures are fixed before the documents, and what cannot be fixed is shown, not hidden**
    (built and tested 2026-10-23, changelog item 20; also for the README and portfolio). When the
    analysis fails a blocking check (for example an order date moved before the snapshot date,
    C5, or a change the judge finds the wrong size, J6b), the analysis agent is sent its output
    and the failures and asked to correct only those, up to a fixed number of times, before any
    document is written, so the purchase request and the email are built only from an analysis
    that passed. If it still fails, the documents
    are written anyway and the red draft note lists the checks not passed ("Not passed: C5 — …"):
    the draft note, already the place for what the buyer must settle before submitting, also
    carries what the pipeline could not settle. Every attempt (output, checks, the feedback sent)
    is kept in the run record. The same pattern as the email's check-and-rewrite loop.
    Severity decides a rewrite, not who checks: blocking code and judge checks both do, warnings
    never do. The cheap code checks run first, so the judge is not called on an analysis that is
    about to change. A whole run with one rewrite took 34.4 s. A faster variant was measured and
    rejected on purpose — worth telling: rewriting only the failing parts saved about 3-4 s per
    rewrite but broke the citations between findings and the judgements that rest on them (three
    blocking failures instead of one in a test), and would have needed a rule per failure for which
    parts to rewrite together; a few seconds were not worth a report that can contradict itself.
    If rewrites run out, the attempt with the fewest blocking failures is used, not the last.
  - **One judgement, the lead time; code works out the dates and the risk** (decided 2026-10-24,
    item 21; for the README, the portfolio and the base documents). The risk formula must be
    stated in all of them: Critical if weeks to stockout < lead time, High if < lead time + 8,
    Medium if < lead time + 17, else Low; lead time 18 weeks by default; latest order date =
    latest delivery date − lead time. The point to make: where the model used to move order dates
    itself (and could move them into the past, or by 0 weeks), it now judges a single figure, the
    lead time, so its judgement is simpler and easier to check; the judge checks only whether
    that figure is justified, and code calculates the order dates and risk levels from it, so the
    order-date arithmetic needs no check at all.
    Tell it as a choice among ways it could have been solved (item 21): patching the model's date
    arithmetic with more rules, only mentioning the effect in an action, or having the model also
    propose risk levels; instead the model judges the one figure that causes the rest, and code
    derives the rest with the formula.
  - **The analysis in two calls, in the order its judgements depend on each other** (item 22; for
    the README, the portfolio and the base documents). The changes first; code recalculates the
    figures; then the text that rests on them. In one call the model wrote "6-week gap" in the
    same answer that made it 10 weeks, and the wrong figure reached the purchase request; every
    check downstream compared it with an input that said the same. The point to make: a check
    can only compare with what it is given, so the fix was the order of the calls, not another
    check, and it also removed a check (J7b) and a rule. Cost: a second, shorter call and a
    second judge call; the time did not grow (29-33 s per supplier).
  - What stayed with LLMs, and why: findings, judgements and the summary (analysis); the
    approvers' Purpose and Justification; the supplier email's wording; and the judge, for what
    needs reading: whether findings are faithful to their documents (J1); whether each
    question, proposed change and action follows from the findings it cites, whether a
    change's size (weeks, factor) fits its basis, and whether items the same reason applies to
    were left out (planned, J6 in item 16; not yet implemented); and the purchase request text
    (P1). Code checks only that a proposal's arithmetic is right, that it starts from the
    pipeline's figure and that its findings exist (C3-C5); whether its logic holds needs the
    judge.
  - The effect: v4's report was written twice by LLMs with figures passing through both; v5
    removed three agents (about 10-14 s per supplier) and every figure is now exact by
    construction (`prompts/CHANGES.md`, v5).
- **README and portfolio: the analysis model was chosen by measurement** (item 23,
  `docs/analysis_model_comparison.md`). The point to make: the same prompt and cases on four models,
  with quality (blocking checks, first-answer figure errors, rewrites, what each analysis raised),
  cost per analysis and time measured together; GPT-4o needed rewrites on every case and missed a
  quality incident, while the newer models were right at first, and the cheapest of them costs
  about 1/25 as much. Also the lesson behind it: several prompt revisions did not fix one model's
  date arithmetic; changing the model did, so the choice of model was tested before more prompt
  work.
- README and portfolio: explain what is hidden from LangSmith and what is still sent, from
  the summary at the top of `docs/observability.md`.
- Return the documents as soon as they are ready, before the evaluation finishes (item 19):
  the API and the web page now wait for the whole run.
- Judge validation must also weigh speed and reliability: on 2026-10-22 `gemini-3.7-flash` took
  30–109 s per call on the same case, then returned 504 and 429 (quota). Include the purchase
  request check (P1) in the validation; it judged the same "60%" for "about 60%" both ways.
- Target: a whole run within one minute. Without the judge one supplier takes about 11 s
  (analysis 5-6 s, documents about 5 s); the judge alone took 30-109 s, so the evaluation
  node's redesign and the judge model decide whether the target is met. Suppliers are processed
  one after another, so a CSV with many suppliers also needs them run side by side (within the
  API rate limits).
- README, for anyone running the sample: runs refuse stock data more than 3 days older than the
  run date, which is fixed for now at 2025-04-06 (`run_date` in `config.py`), so the sample CSV
  (snapshot 2025-04-05) runs and a CSV with another date is refused unless `run_date` is changed
  or `REQUIRE_FRESH_SNAPSHOT=false` is set.
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

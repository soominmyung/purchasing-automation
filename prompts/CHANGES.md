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
| v4 | One history store scoped by document headers; `AvailableStock` | commit 718a062 to d12e168 |
| v5 | Analysis in fields with numbered findings; documents assembled in code; one approval agent | not yet committed (current) |

## Files

| File | Used by |
|---|---|
| `analysis.md` | Analysis agent |
| `approval_request.md` | v5: approval agent (the purchase request's Purpose and Justification) |
| `report_doc.md` | v4 and earlier: report agent |
| `pr_draft.md`, `pr_doc.md` | v4 and earlier: purchase request agents |
| `email_draft.md` | Email draft agent |
| `email_revision_system_suffix.md`, `email_revision_request.md` | Email rewrite after a failed check (`{feedback}` is filled in) |
| `email_validator_system.md`, `email_validator.md` | Email check (`{email}` is filled in) |
| `evaluation.md` | v1 only: the evaluator before the checklist |

The email check and rewrite prompts were written inline in `services/agents.py` until
2026-10-19; their text is the same in every version.

---

## v5 — 2026-10-22 (current)

Design and the reasons for each decision: `docs/changelog_2026-10.md`, items 16 and 19.

**What changed**
- The analysis agent returns fields and no report: a summary; numbered findings, each a fact
  from the history documents or the stock figures, with its source; critical questions marked
  `supplier` or `internal`; proposed order-date and quantity changes (pipeline value, shift or
  factor, proposed value, reason); recommended actions. Every question, change and action
  cites the findings it rests on.
- The analysis report is assembled in code (`services/documents.py`): Items, Findings, Critical
  Questions, Proposals, Recommended Actions, Item Details, Referenced Documents. The report
  agent (`report_doc.md`) is removed.
- The purchase request's two agents (`pr_draft.md`, `pr_doc.md`) are replaced by one approval
  agent (`approval_request.md`) that writes Purpose and Justification; code writes the red
  draft note with the open questions and actions, the items table, Supporting Documents and
  the approval block. Justification covers only what the approvers decide on: why these items
  must be bought now, why at these quantities and order dates, and actions that add cost or
  change how the items are sourced.
- The email agent receives only what its text needs: the supplier, the items (name, quantity,
  requested delivery date), `points_to_clarify` (the analyst's supplier questions, described as
  internal notes, not wording for the supplier), the Critical items for the earliest-dispatch
  question, and `other_items` (items of this supplier outside the email). No snapshot date,
  risk levels, stock figures, findings or internal questions. Code puts the item lines after
  the purpose paragraph; the agent writes no item list. Only text in quotes in the prompt is
  copied into the email (greeting, signature).
- Models: the email draft and its rewrite use `gpt-5.4-mini` (were `gpt-4o-mini` and
  `gpt-4o`); analysis and approval stay on `gpt-4o`.
- Code checks on the analysis: C1 format, C2 items, C3 changes start from the pipeline's
  figures, C4 citations (cited findings exist and apply; a finding's documents apply to what
  it is about), C5 change arithmetic, C6 each Critical item has an action on its stock
  position. Checks on the purchase request text: P1 (judge) adds nothing to its input, P2 (code)
  paragraph length, P3 (code) cites only documents given.

**Why**
- In v4 the report was written twice: the analysis agent wrote a free-form report and the
  report agent rewrote it, so figures passed through two LLMs while only the first was checked,
  and headings and the history part varied from scenario to scenario.
- v4's history labels ("History reviewed: no notable …") were written by the model and used
  inconsistently (eval_16, H2, H5).
- v4's questions and proposals had no recorded basis, so a judgement could not be traced to a
  document, and the email agent, given the whole analysis, could repeat internal findings to
  the supplier ("We noted your recent notice on pricing…").

**Effect measured** (one development case, not from the evaluation set: one supplier, four
items — Critical, High, Medium, Low — and four history documents; 2026-10-22; GPT-4o analysis,
judge `gemini-3.7-flash`):
- All six analysis code checks passed in every run. The output followed the new structure: six
  findings, one of them a stock finding for the Critical item's 9-week gap, which the bridging
  action cites; questions split into supplier and internal; every question and action cited a
  finding.
- The judge passed J1 (no invented, distorted or misattributed history), J1b, J2a, J2b and
  J2c. J5 failed because it asks for a mitigation inside the summary, while the v5 summary
  leaves actions to their own section (the check is to be revised).
- Figures in the tables come from code only; the model's text quotes them.
- Run time per node, against the last v4 run record (2026-10-19, sample CSV, suppliers of one
  or two items, so not the same data): the three agents removed took 13–18 s per supplier
  (report 5.2–6.9 s, PR draft 3.8–7.0 s, PR document 2.8–4.1 s); the approval agent that
  replaces them takes 2.7–3.7 s, so about 10–14 s are saved per supplier. Whole runs did not
  get shorter, because the judge dominates and varies: on the same case the evaluation took
  30.3–109.3 s and the purchase request check 3.5–53.4 s, against 9.3–55.7 s per supplier in
  the v4 record. One judge answer was 6,466 tokens for 2,542 input tokens, mostly thinking.
- With the documents run beside the evaluation (`docs/changelog_2026-10.md`, item 19), they are
  finished 11.0 s after the start (measured with a 20-second stand-in for the judge).

**Tuning the email prompt (2026-10-22).** Each step was one change and one run unless stated;
the same analysis output was used from step 5 on. The outputs of every step are kept in
`dev/runs/2026-10-22_*` (`dev/README.md` says which run was which step).

| Step | Change | Result |
|---|---|---|
| 1 | First v5 email prompt (`gpt-4o-mini`) | The analyst's question copied almost word for word ("Specifically, has the customs documentation process been fully resolved …"); a Low item outside the email named; "As of our snapshot date"; `ITEMS` without braces |
| 2 | No snapshot date in input or purpose; "reveal no internal management information or judgement"; "name only the items in items" | Snapshot date and the Low item gone; the question still named the supplier's past problem ("whether the customs documentation process has effectively resolved the shipment delay issue"), which reads as distrust |
| 3 | "Ask indirectly, as a routine check, whether the steps … where such a problem could arise are expected to go smoothly" | The question was dropped; the fallback for "no questions" was used instead |
| 4 | "Is still asked, without naming the problem" | The problem named again |
| 5 | Same prompt on `gpt-4o` | The problem named, with its date, and the Low item back |
| 6 | Input renamed from `supplier_questions` to `points_to_clarify`, described as the analyst's internal notes; rule split into bullets | The problem still named |
| 7 | Diagnostic: `gpt-4o` asked which rules apply to the point and what it would write | It named the right rule and wrote a compliant question ("whether all documentation processes are expected to proceed smoothly …"): the rule was understood, but not applied when writing the whole email |
| 8 | Same prompt on `gpt-5.4-mini` | Problem not named, Low item left out; but the customs point asked twice, and no thanks |
| 9 | The two bullets merged into one question per point | No repeat; "the related order steps", with the step itself lost |
| 10 | "Name the step … (not the problem itself)" | "the customs documentation step"; the Low item back |
| 11 | `other_items` in the input; "leave out every item in other_items, also where a point mentions it" | Low item gone; "as a routine check" copied into the email |
| 12 | That phrase removed from the rule | The past problem named with its date again |
| 13 | "The question is about this order only … as an ordinary question … no reference to what happened before or when" — five runs | All five clean on the past problem and the items; "for this order only" copied in two, "now in order" in one |
| 14 | "The instructions' wording is not for the email"; manner described by its effect on the reader — five runs | No phrase copied; questions vaguer in places, one "without issues" |
| 15 | Only text in quotes is copied; greeting, `{ITEMS}` and signature quoted — five runs | Greeting and two-line signature right in all five; the model wrote the item list itself instead of `{ITEMS}` in all five |
| 16 | No placeholder: code puts the item lines after the purpose paragraph — five runs | List placed right in all five; "the items below" once, the no-questions fallback added twice, worry-tinged wording ("avoid any delay", "no impact", "complications") in three or four |
| 17 | The agent told the list follows the purpose paragraph and its questions come after | Full run: clean email |

Continued on 2026-10-23 with a new case, `kestrel_components` (`dev/runs/2026-10-23_*`), five runs per
step on the analysis of its first run:

| Step | Change | Result |
|---|---|---|
| 18 | (first full run of the case) | "for the items below" in the requests, though the list is above them |
| 19 | "The list is added directly after the purpose paragraph; everything after it comes after the list" | Requests right in five; the purpose paragraph said "the items listed above" in one, before the list |
| 20 | The order of the parts spelled out (greeting, purpose, list, requests, closing) | Worse: the model wrote the item names as a list itself in two, and a request inside the purpose in one |
| 21 | The agent returns three parts as JSON (purpose, requests, closing); code assembles the email | No stray list, greeting and signature right; "below" in the requests in two; a request inside the purpose in one |
| 22 | The parts numbered in assembly order (the user's wording): "3) comes between 2) and 4): so in 4), the items are already listed above" | No wrong "below"; "the attached requirements" (nothing is attached) in one; a request inside the purpose and "we are reviewing our replenishment needs" in another |
| 23 | Purpose and closing written by code; the agent writes only the requests | Purpose problems gone; the requests repeated the list's quantities and dates in four ("by the dates below: BearingCap 1,300 units by 2025-06-28, …") |
| 24 | "Unless a request adds something the list does not say, refer to the items, quantities and dates as those in the list above" | All five clean: no repeated figures, no wrong "below", no item outside the email, no past problem; the list referred to as "the listed items", "the dates shown". Two asked about the schedule twice in different words |

What step 18-24 showed: each time the model had to place text around a part it does not write, it
guessed (a placeholder, a position, "attached", "below"); the fix that held was to take the fixed
parts away from it altogether. The email agent now writes one thing: the questions for the
supplier, turned from the analyst's notes into polite questions to a business partner.

What was hard:
- **The input's wording outweighed the rules.** The analyst's supplier question is already a
  complete, polite question that names the past problem and every item. Asked to "write a
  polite question that gets this information", the models re-wrapped it. The field name
  `supplier_questions` also said "questions ready for the supplier".
- **Understanding is not applying.** `gpt-4o` explained the rule correctly when asked about the
  one point, yet did not apply it while writing the whole email; `gpt-4o-mini` did the same.
  `gpt-5.4-mini` applied it with the same prompt. A model change, not another prompt layer
  (a separate question-writing agent was considered), was what fixed this.
- **Rules written in email-ready English were copied as wording.** "as a routine check",
  "about this order only", "the related steps" appeared in the emails. The prompt mixed text
  to copy (greeting, signature) with instructions in the same register. Fixed by one rule —
  only quoted text is copied — and by describing manner by its effect on the reader.
- **Placeholders are fragile.** `{ITEMS}` was written without braces, or filled in by the model
  once it was quoted. Code now places the item lines itself.
- **One run per change could not separate a rule's effect from run-to-run variation**: each fix
  seemed to trade one fault for another. From step 13 each change was run five times.

**Tuning the approval prompt (2026-10-22, `gpt-4o`).**

| Step | Change | Result |
|---|---|---|
| 1 | First v5 approval prompt | Wording not in the input ("necessitating immediate action to prevent operational disruptions"; "replacements were promptly provided" for "arrived two weeks later"); five sentences against 2–4. Nothing checked this text |
| 2 | Rules: each reason only as far as its finding states; figures exactly as given; no exaggeration or meaning-changing words. Checks P1–P3 added | Same faults ("promptly", "it is crucial to confirm … to prevent similar issues"), five sentences; P1 and P2 caught them |
| 3 | Facts: no word that judges a figure. Actions: may say why they matter, as far as the findings support. Checks before ordering written as conditions | Seven sentences, "several critical factors", "promptly", no condition; P1 and P2 caught them |
| 4 | Justification narrowed to what the approvers decide; the three overlapping rules merged into one | Four sentences; the defective batch, which bears on no approval decision, left out; no "promptly". P1 caught "60% rise" (input: "about 60%") and "immediately" (the input lacked that the order date had passed) |
| 5 | The order-date timing ("overdue") added to the approval agent's input | "immediately as it is overdue" now supported; "60%" again, which P1 passed this time |

What was hard:
- **Adding rules made the text longer, not more exact.** The cause was scope: "the findings,
  the changes and the actions together" asked for everything in 2–4 sentences, and the model
  chose coverage, dressing facts that bear on no decision as reasons. Narrowing what
  Justification covers fixed length and most added wording.
- **Where advocacy ends.** "It is important to confirm delivery dates" suits a purchase
  request; "promptly" for a stated two weeks does not. The line drawn: no word that judges a
  stated figure; an action's importance may be stated as far as its findings support.
- **A gap in the input, not the prompt.** "Immediately" was flagged because the input did not
  say the order date had passed.
- **The judge is not consistent.** The same "60%" for "about 60%" failed P1 once and passed once.

**Changed figures in the text (2026-10-23).** The text writes an order date or quantity that a
change proposes as "<pipeline value> → <proposed value>", as the tables do (the rule had kept
proposed values out of the text). Code checks that every arrow in the text is one of the
proposed changes (C9, blocking) and highlights it; the judge checks that a changed figure is not
written as only one of the two (J7). First run (`kestrel_components`, `2026-10-23_013618_arrows-in-the-text-c9-j7-for-changed-figures`):
five order-date changes and one quantity change were proposed, but the text mentioned no changed
figure ("requiring a reassessment of order quantities"), so the rule was not exercised; one order
date was moved before the snapshot date, caught by C5.

**Known faults** (not yet changed)
- Approval agent: "about" dropped from a figure; a pre-order check written as "it is crucial to
  confirm …" rather than as a condition.
- Email agent: worry-tinged wording in some runs ("to avoid any delay", "now in order"); the
  no-questions fallback added in some runs although there were questions; "the items below" in
  the first question although code puts the list above it, also after the prompt said so.
- Analysis agent: an order date moved earlier than the snapshot date for an item already overdue
  (limits on changes decided 2026-10-23, not yet in the prompt).
- J5 expects the mitigation in the summary; to be revised with the evaluation criteria.
- A supplier-side quality incident (a defective batch) gave a finding but no question or
  action, and a 60% demand rise gave an action to review the quantity rather than a proposed
  change. Both are allowed by the rules; the analysis-model benchmark is to measure this.
- The judge (`gemini-3.7-flash`) was slow and unreliable on 2026-10-22: 30–109 s per call,
  then 504 (deadline exceeded) and 429 (quota exhausted). To be settled in judge validation.

## v4 — 2026-10-19

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

**Effect measured** (GPT-4o, evaluation scenarios converted to history documents): the
unattributed quality note in eval_08 no longer reaches any item's notes (it reached all
three before); the other-item document added to eval_12 reaches no item's notes; supply-side
documents in H2 and H3 reach only the item they name; no item with history was given the
no-history sentence.

**Known fault — the example teaches a wrong note.** The example's notes end with "History
reviewed: no notable supplier or item issues." although the example supplier had a 3-week
delay. (In v3 the example note named the delay; v4 moved supplier-wide issues to the
summary and put the no-notable sentence in its place.) GPT-4o copied both lessons: it ended
a note with the sentence right after describing a demand surge (eval_16, TitaniumAlloyWheel)
and wrote "No notable supplier issues" for an item of a supplier whose operations had halted
(H5, PanelFiber). Adding a correct example for an item with an incident ("History reviewed:
<what the document says>") changed the eval_16 note to that form. Fixed in v5, where notes
are written by code.

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

# Analysis-agent evaluation rubric

The checklist used to judge the analysis agent's output. It replaces the earlier
evaluation, in which a GPT-4o judge rated each answer against GPT-4o's own answer and so
measured how closely a model imitated the teacher. Here every system, GPT-4o included, is
scored on the same checks, anchored to the input rather than to another model's answer.

## Where these checks run

The same check definitions, in `services/quality_checks.py`, are used in two places.

| | Offline evaluation | Production evaluation node |
|---|---|---|
| Purpose | Compare systems (GPT-4o, the fine-tuned model, …) before choosing one | Check every analysis the pipeline produces before a buyer relies on it |
| Code | `scripts/eval_v2_score.py` | `evaluation_node` in `services/agents.py` |
| Checks and their wording | Identical | Identical |
| When J2a / J2b apply | From scenario labels: only where a problem was built into the scenario | From the inputs: whenever that history is present, and the judge decides whether it describes a problem |
| Judges | Three providers, majority vote | One judge, `evaluator_model` in `config.py` |
| Human review | A sample of about ten scenarios | None |
| Output | Pass rates per check and per system | A pass/fail table in the evaluation report, and `checks` in the API response; the analysis is marked ready only if every check passed |

J2a and J2b differ because production data has no labels: nothing marks a real supplier's
history as containing a problem, so the judge has to read it. Everything else is the same,
so a check that the offline evaluation relies on is also enforced on every live run.

When this document and the code disagree, this document is the intended behaviour and the
code is out of date.

## Scenario set

25 scenarios, none of them used in training.

| Set | Count | Source |
|---|---:|---|
| Generated evaluation set | 20 | `eval_data/eval_scenarios.jsonl`, built by `scripts/generate_eval_scenarios.py` |
| Original SFT holdout | 5 | `eval_data/holdout_scenarios.jsonl`, built by `scripts/build_holdout_eval.py` from the last 5 teacher scenarios |

Generated scenarios by type:

| Type | Count | What it tests |
|---|---:|---|
| `clean` | 5 | Nothing to raise; no question should presume a problem |
| `no_history` | 2 | Histories are the tools' "no history found" strings |
| `ambiguous_attribution` | 3 | Supplier history describes a problem without naming a company |
| `clear_supplier_incident` | 3 | Named, unresolved supplier problem |
| `clear_item_incident` | 3 | Named item problem |
| `high_stock_short_cover` | 2 | Large stock with one to two weeks of cover |
| `missing_values` | 2 | One item has no AvailableStock or WksToOOS |

Supplier and item names do not overlap with the training data.

### Risk mix

`scripts/set_eval_risk_mix.py` sets each item's weeks to stockout so all four risk levels
are covered (the scenarios were first generated with almost every item under the 18-week
lead time). Stock is scaled with the weeks, so the two figures stay consistent; histories
and labels are unchanged.

| Critical | High | Medium | Low | Scenarios with no Critical item | Only Critical | Mixed |
|---:|---:|---:|---:|---:|---:|---:|
| 23 | 13 | 10 | 7 | 7 | 4 | 14 |

Counts are of the 53 items the model sees; the 3 rows with missing figures are excluded.

### History documents

The model receives history as a list of documents, each naming the supplier and items it is
about, as the pipeline does in use. `scripts/convert_eval_history.py` turns each scenario's
two history texts into documents without changing the text: the supplier history becomes a
supplier-wide document, the item history a document about the items it names. Two item
histories are split so a supply-side sentence becomes a supplier × item document (H2: PartW's
lead time; H3: DeviceConnector's erratic supply). One document is added to eval_12, about a
defective batch of an item of the same supplier that is not in the analysis. eval_08 keeps a
supplier-wide quality note that names no item, which must not be attributed to any item.

### Labels

Each scenario carries facts about itself, not the expected answer. The checks below decide
what a good analysis does with them.

| Label | Values |
|---|---|
| `supplier_issue` | `material`, `none` |
| `item_issue` | `material`, `none` |
| `history_present` | `true`, `false` |

A supplier history with no company named counts as `material`: the history tool filters
on supplier metadata, so whatever it returns belongs to that supplier.

## Inputs

Scenarios are passed through the production preprocessing (`services/item_grouping.py`)
before any model sees them, exactly as the pipeline does in use:

- risk level is derived from weeks to stockout (Critical below the 18-week lead time,
  High below 26 weeks, Medium below 35, Low beyond)
- Critical items carry `expected_stockout_weeks` and a fixed note written by code
- items whose AvailableStock or WksToOOS is missing or invalid are excluded and reported in
  the report's opening "Data Issues Requiring Attention" section, so the model never sees them

## Checks

For prompt v5 (2026-10-23). The output has a summary, numbered findings (facts with their
sources) and judgements that cite them (critical questions, proposed changes, actions); the
checks follow that split. Severity: see "Severity, verdicts and comparison" below.

### Code checks — deterministic, every scenario

| ID | Passes when | Severity |
|---|---|---|
| C1 | The output parses and has `summary`, `findings`, `critical_questions`, `po_date_changes`, `quantity_changes` and `recommended_actions` with the right types | Blocking |
| C2 | Every item code it names is in the input | Blocking |
| C3 | Each proposed change starts from the pipeline's figure (the item's recommended order date, or its suggested quantity) | Blocking |
| C4 | Every finding rests on documents that were given and apply to what it is about, or on the stock position; every question, change and action cites at least one finding that exists and concerns its items or the supplier as a whole | Blocking |
| C5 | Each change adds up (the date moved by the stated weeks; the quantity times the factor, rounded up), and no proposed order date is before the snapshot date | Blocking |
| C6 | Each Critical item has an action that cites a stock finding about it | Blocking |
| C7 | The summary names every Critical item | Warning |
| C8 | The summary is within about 100 words | Warning |

### Judge checks — yes or no

| ID | Passes when | Applies to | Severity |
|---|---|---|---|
| J1 | It invents no incident or trend that the history documents do not contain, does not distort what they say (for example, turning "high demand" into "medium demand"), and does not attribute a document to a supplier or item it does not name | All | Blocking |
| J2a | At least one critical question addresses the supplier's problem | `supplier_issue = material` | Warning |
| J2b | At least one critical question addresses an item's problem | `item_issue = material` | Warning |
| J2c | No critical question presumes a supplier or item problem that the inputs do not show. Neutral questions that assume no problem pass | All | Warning |
| J7 | Every figure from the stock data quoted anywhere in the output (available stock, weeks to stockout, the gap, suggested quantity, recommended dates) matches the input for that item; figures from the history documents are J1's | All | Blocking |

### Checks on the purchase request text (production)

| ID | Passes when | By | Severity |
|---|---|---|---|
| P1 | Purpose and Justification add no fact, risk or assessment their input does not give, quote figures exactly with no word that judges them, and change no meaning; saying why an input action matters passes when its findings support it | Judge | Warning |
| P2 | Purpose has 1-2 sentences, Justification 2-4 | Code | Warning |
| P3 | Every cited document number was given | Code | Blocking |

### Checks to be added (decided 2026-10-23, not yet implemented)

| Check | Severity | Note |
|---|---|---|
| A proposed change, and its size, follow from the findings it cites, and items the same reason applies to are not left out | Blocking | J6 in the v5 design |
| A question for the supplier reveals nothing internal (stock levels, urgency, budget, the analysis) | Blocking | |
| Each question and action follows from the findings it cites | Warning | |
| No document that describes a problem for this order is left unused by every finding and judgement (code lists the unused documents; the judge reads only those) | Warning | To replace J2a and J2b |
| Findings state facts only, with no judgement | Warning | |

### Checks dropped for v5

| ID | Check | Why |
|---|---|---|
| J1b | Does not attach the fixed "no history" sentence to an item a document applies to | Code now writes the "no documents" findings; the model never writes that sentence. Claiming that history is absent when it is not is a distortion, under J1 |
| J3 | States that no history is available when there is none | Written by code, as above |
| J5 | The summary names each Critical item with its gap and recommends a mitigation | The v5 summary leaves actions to their own section. Split into C7 (Critical items named in the summary) and C6 (an action on each Critical item's stock position) |

The judge sees the scenario's inputs and the candidate output. It is never told which
model produced the output.

## Judges and aggregation

Offline:

| Provider | Model |
|---|---|
| OpenAI | `gpt-5.5-2026-04-23` |
| Anthropic | `claude-sonnet-5-5` |
| Google | `gemini-3.7-flash` (version `3.7-flash-08-2026`) |

Production: `gemini-3.7-flash`, a different provider from the GPT-4o analysis agent so the
analysis is never judged by a model of its own family.

Pinned versions only, so a rerun gives comparable results. GPT-4o is not a judge because it
is one of the systems under evaluation.

These choices are provisional until the judges have been tested on outputs with known
verdicts; see `docs/judge_validation.md`. Costs per call are in `docs/judge_model_costs.md`.

- Each judge check is decided by majority of the three judges.
- A sample of about ten scenarios is also scored by hand, to check the judges against a
  person with domain knowledge.

## Severity, verdicts and comparison (decided 2026-10-23)

**No score.** The first version rated each analysis 1-10 on four criteria and reported an
average (9.5/10), and readers asked what the number measured. A weighted score would raise the
same question about its weights, and adding checks up hides what failed: an invented incident
and a long summary would differ by the same few points. Results are reported check by check.

**Each check is either blocking or a warning** (assigned 2026-10-23; see the tables under
"Checks"):
- *Blocking*: the analysis must not be used as it is if this fails (for example invented or
  distorted history, a wrong figure, an item not in the input, internal information in a
  question for the supplier).
- *Warning*: usable, but a person should look (for example a summary over length).

**Production evaluation node: checks by severity.** Every analysis is "ready" only if every
blocking check passes; otherwise "needs review". Warnings are listed with the result but do not
change it. A check the judge could not answer counts as not passed.

**Model comparison: staged.**
1. Blocking checks: the share of analyses that pass every blocking check, and the count of
   failures per blocking check. A system that falls short here is not chosen, whatever else.
2. Warning checks, among the systems left: how much a person would have to correct.
3. Cost and response time, among systems of similar quality (target: a whole run within one
   minute).

The set is small (25 scenarios), so a difference of one or two scenarios may be chance: counts
are reported as they are, with how often the judges agreed, and only clear differences are read
as differences.

**The same checks serve fine-tuning.** The fine-tuned model is one of the systems compared, on
the same scenarios and stages; its case rests on stage 3 (cost and latency when self-hosted) at
blocking results close to the teacher's. Teacher outputs used as training data must pass every
blocking check, so the student does not learn the teacher's mistakes; outputs that pass and fail
on the same scenario can form preference pairs for DPO. Quantised serving variants are checked
the same way.

## Checks that were considered and dropped

| ID | Check | Why it was dropped |
|---|---|---|
| J2d | Does not attribute an unattributed supplier problem to this supplier | The history tool only returns this supplier's documents, so attributing it is correct |
| J4a | Points out contradictory figures | Risk level is now derived from weeks to stockout in code, so the figures can no longer contradict each other |
| J4b | Does not invent values for missing figures | Items with missing or invalid figures are now excluded in code before the model sees them |
| — | Flags a question's `reason` field when it says `supplier_history` with no incident | Not worth a check of its own |

## Change log

| Date | Change |
|---|---|
| 2026-10-23 | No score: checks are blocking or warnings; the production verdict follows the blocking checks; models are compared in stages (blocking, warnings, cost and latency); the same checks filter fine-tuning data. Checks revised for prompt v5: C1 and C3 changed; J7 added (stock figures quoted in the text, judged; C3 covers only the changes' starting values); C4-C8 and P1-P3 added; J1b, J3 and J5 dropped (J5 split into C6 and C7); each check given a severity (C4 and P3 blocking, P1 and J2c warnings); five judge checks and a wider C3 planned. |
| 2026-10-19 | History given as documents scoped by their headers (prompt v4). J1 also fails attribution to a supplier or item a document does not name. `CurrentStock` renamed `AvailableStock`. |
| 2026-10-18 | Weeks to stockout reset for 29 items to spread the set across all four risk levels (previously 50 of 53 items were Critical). Holdout scenarios included, so their figures no longer match the teacher dataset. |
| 2026-10-18 | Production evaluation node rewritten to use these checks (it previously scored four 1–10 criteria with GPT-4o judging its own analysis, one of them about email leaks it had no email to judge). Production judge set to `gemini-3.7-flash`. |
| 2026-10-18 | Document created. J1 narrowed to invented or distorted history, with the fixed no-history sentence split out as J1b. J2d dropped and ambiguous supplier histories relabelled `material`. J4a and J4b dropped as the preprocessing now handles them. J5 added for Critical items. Gemini judge set to `gemini-3.7-flash`. |
| 2026-10-16 | Neutral questions in scenarios with nothing to raise pass J2c. Resolved-incident scenarios removed from the generated set as too ambiguous to label. One contradiction scenario removed, leaving 20 generated scenarios plus the 5 original holdout scenarios. |
| 2026-10-15 | Moved from imitation-of-GPT-4o scoring to a fixed checklist scored by judges from three providers. |

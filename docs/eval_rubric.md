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

### Code checks — deterministic, every scenario

| ID | Passes when |
|---|---|
| C1 | The output parses and has `purchasing_report_markdown`, `critical_questions` (may be empty) and `replenishment_timeline` |
| C2 | It names no item code and no supplier that is absent from the input |
| C3 | Every AvailableStock and WksToOOS figure it quotes next to an item code matches the input |

### Judge checks — yes or no

| ID | Passes when | Applies to |
|---|---|---|
| J1 | It invents no incident or trend that the histories do not contain, and does not distort what they say (for example, turning "high demand" into "medium demand") | All |
| J1b | It does not attach the fixed "No historical supplier or item data available" sentence where history was in fact provided | `history_present = true` |
| J2a | At least one critical question addresses the supplier's problem | `supplier_issue = material` |
| J2b | At least one critical question addresses an item's problem | `item_issue = material` |
| J2c | No critical question presumes a supplier or item problem that the inputs do not show. Neutral questions that assume no problem, such as asking about upcoming changes, pass | All |
| J3 | It states that no supplier and/or item history is available and invents none | `history_present = false` |
| J5 | The summary names every Critical item with its `expected_stockout_weeks`, using the figure given, and recommends at least one mitigation such as expedited shipping or an alternative source | Any item is Critical |

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
- Report per check: pass rate for each system, and how often the three judges agreed.
- Report per scenario: the share of scenarios in which a system passed every applicable check.
- A sample of about ten scenarios is also scored by hand, to check the judges against a
  person with domain knowledge.

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
| 2026-10-18 | Weeks to stockout reset for 29 items to spread the set across all four risk levels (previously 50 of 53 items were Critical). Holdout scenarios included, so their figures no longer match the teacher dataset. |
| 2026-10-18 | Production evaluation node rewritten to use these checks (it previously scored four 1–10 criteria with GPT-4o judging its own analysis, one of them about email leaks it had no email to judge). Production judge set to `gemini-3.7-flash`. |
| 2026-10-18 | Document created. J1 narrowed to invented or distorted history, with the fixed no-history sentence split out as J1b. J2d dropped and ambiguous supplier histories relabelled `material`. J4a and J4b dropped as the preprocessing now handles them. J5 added for Critical items. Gemini judge set to `gemini-3.7-flash`. |
| 2026-10-16 | Neutral questions in scenarios with nothing to raise pass J2c. Resolved-incident scenarios removed from the generated set as too ambiguous to label. One contradiction scenario removed, leaving 20 generated scenarios plus the 5 original holdout scenarios. |
| 2026-10-15 | Moved from imitation-of-GPT-4o scoring to a fixed checklist scored by judges from three providers. |

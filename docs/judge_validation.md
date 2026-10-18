# Judge validation

Before a judge model is trusted to score the analysis agent (`docs/eval_rubric.md`), it is
itself tested on outputs whose correct verdict is already known. Without this, a difference
between two analysis models could just as well be a judge getting things wrong.

**Status:** seeded errors built 2026-10-18 (`scripts/build_judge_calibration.py`); hand scoring and the judge run not yet done.

## Calibration set

Two kinds of output, both built from GPT-4o's analyses of the 25 evaluation scenarios
(produced through the production preprocessing, as described in `docs/eval_rubric.md`).

### 1. Seeded errors — the main test

Each seeded output is an original analysis with exactly **one** deliberate fault, aimed at
one check. Because the fault was put there on purpose, the correct verdict for that check is
known to be **fail**. Only the targeted check is scored on a seeded output; the other checks
are left as they were in the original and have no known answer.

| Target | Fault seeded | Applied to | Count |
|---|---|---|---:|
| J1 — invented history | One sentence describing an incident that is in neither history, added to the report or an item's notes | Scenarios with history | 5 |
| J1 — distorted history | One fact from the history changed in the output (for example "high demand" to "medium demand", a 3-week delay to "a minor delay") | Scenarios with history | 3 |
| J1b | An item's notes replaced with the fixed sentence "No historical supplier or item data available…" although history applies to it | Scenarios with history | 4 |
| J2a | Every critical question about the supplier's problem removed | `supplier_issue = material` | 4 |
| J2b | Every critical question about the item's problem removed | `item_issue = material` | 4 |
| J2c | A question presuming a problem the data does not show (for example "Have the repeated delivery delays been resolved?") added | `clean`, `no_history` | 4 |
| J3 | Every statement that no history is available removed | `no_history` | 2 |
| J5 — omitted | Every mention of the Critical item removed from the report summary | Critical items present | 4 |
| J5 — wrong figure | The expected stockout figure in the summary changed to a different number | Critical items present | 3 |
| | | **Total** | **33** |

How the faults are made:
- Every fault is an exact text edit written out in `scripts/build_judge_calibration.py`, so
  each one can be read and reviewed; the build stops if the original text is not found.
- Invented incidents, distortions and presumptive questions were written by hand against the
  scenario's histories, so each is a clear fault and the only change.
- A fault is placed only on an output that passes the targeted check before the edit. GPT-4o
  attaches the fixed no-history sentence to items with history in about ten of its 25
  outputs, so J1b faults go only to outputs that do not already do this.

Stored in `eval_data/judge_calibration.jsonl`: an id, the scenario, the targeted check, the
expected verdict, a one-line description of the change, and the full modified output.

### 2. Hand-scored originals — the false-alarm test

About ten unmodified analyses, chosen across scenario types, are scored on every applicable
check by a person with purchasing domain knowledge. These give the correct verdict where the
analysis is sound, so a judge that fails a good analysis can be caught.

Judge agreement is deliberately **not** used as ground truth here: that would test the judges
against themselves.

## Candidates

| Provider | Model |
|---|---|
| OpenAI | `gpt-5.5`, `gpt-5.4-mini`, `gpt-5-nano`, `gpt-6-luna` |
| Anthropic | `claude-sonnet-5-5`, `claude-haiku-5-5` |
| Google | `gemini-3.8-flash`, `gemini-3.1-flash-lite` |

Each candidate judges every calibration output **three times**, with the same prompt the
production evaluation node uses (`services/quality_checks.py`).

## Metrics

| Metric | Definition | Why it matters |
|---|---|---|
| Detection rate | Share of seeded faults the judge marked as fail | A missed fault lets a bad analysis through as "ready". The most important number. |
| False alarm rate | Share of hand-scored passing checks the judge marked as fail | High values send sound analyses to human review for nothing |
| Flip rate | Share of (output, check) pairs whose verdict was not the same in all three runs | A judge that changes its mind cannot be relied on |
| No-answer rate | Share of checks the judge failed to answer (error or unparseable) | Counted as failures in production, so it inflates human review |
| Cost per call | Mean of measured tokens × list price | — |
| Latency | Mean seconds per call | — |

Detection rate is reported per check as well as overall, since a judge may be reliable on
format-like checks and weak on subtle ones such as distorted history.

## Selection rule

- **Production judge:** among candidates with detection rate ≥ 90 %, false alarm rate ≤ 10 %
  and flip rate ≤ 5 %, the cheapest, with latency as the tie-break.
- **Offline judge panel:** the most accurate candidate from each of the three providers.

If no candidate meets the production thresholds, the thresholds are not lowered quietly: the
result is recorded here and the judge prompt is revisited first.

## Results

*To be filled in after the run.*

### Overall

| Model | Detection rate | False alarm rate | Flip rate | No-answer rate | Cost per call | Latency |
|---|---:|---:|---:|---:|---:|---:|
| | | | | | | |

### Detection rate by check

| Model | J1 invented | J1 distorted | J1b | J2a | J2b | J2c | J3 | J5 omitted | J5 wrong figure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | | | | | | | | | |

### Decision

*Which judges were chosen, and why.*

## Change log

| Date | Change |
|---|---|
| 2026-10-18 | Design written. |
| 2026-10-18 | 33 seeded errors built from GPT-4o outputs regenerated after the risk mix was spread across all four levels. J1b faults replace the item's notes, mirroring how GPT-4o makes this error. |

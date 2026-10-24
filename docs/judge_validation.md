# Judge validation

Before a judge model is trusted to score the analysis agent (`docs/eval_rubric.md`), it is
itself tested on outputs whose correct verdict is already known. Without this, a difference
between two analysis models could just as well be a judge getting things wrong.

**Status:** seeded errors rebuilt 2026-10-19 (`scripts/build_judge_calibration.py`); hand-scoring sheet ready (`eval_data/hand_scoring_sheet.md`), scores not yet in; judge run not yet done.

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
| J1 — misattributed | A document's content written into the notes of an item the document does not name (eval_08: the unattributed quality note; eval_12: the other item's defective batch) | eval_08, eval_12 | 2 |
| J1b | An item's notes replaced with the fixed sentence "No historical supplier or item data available…" although history applies to it | Scenarios with history | 4 |
| J2a | Every critical question about the supplier's problem removed | `supplier_issue = material` | 4 |
| J2b | Every critical question about the item's problem removed | `item_issue = material` | 4 |
| J2c | A question presuming a problem the data does not show (for example "Have the repeated delivery delays been resolved?") added | `clean`, `no_history` | 4 |
| J3 | Every statement that no history is available removed | `no_history` | 2 |
| J5 — omitted | Every mention of the Critical item removed from the report summary | Critical items present | 4 |
| J5 — wrong figure | The expected stockout figure in the summary changed to a different number | Critical items present | 3 |
| | | **Total** | **35** |

How the faults are made:
- Every fault is an exact text edit written out in `scripts/build_judge_calibration.py`, so
  each one can be read and reviewed; the build stops if the original text is not found.
- Invented incidents, distortions and presumptive questions were written by hand against the
  scenario's histories, so each is a clear fault and the only change.
- A fault is placed only on an output that passes the targeted check before the edit. GPT-4o
  attaches the fixed no-history sentence to items with history in about ten of its 25
  outputs, so J1b faults go only to outputs that do not already do this.

**To add when the set is rebuilt for prompt v5** (decided 2026-10-24): faults for J7a in which
a lead time changes and the text keeps a figure from before the change, since only the judge
checks figures in prose. From real runs: the summary and an action keeping the 18-week gap after
the lead time went to 22 weeks ("6-week gap" for 10, `dev/runs/2026-10-23_182404_…`); a stock
finding keeping a 22-week gap after the change was withdrawn ("WasherSteel runs out 2 weeks before
a new order can arrive", `dev/runs/2026-10-24_011306_…`); a risk level or order date at 18 weeks in
an action. Seed them in each prose part (stock finding, summary, question, action), and in wordings
a pattern would miss ("a ten-week gap", "a shortfall of 10 weeks").

Stored in `eval_data/judge_calibration.jsonl`: an id, the scenario, the targeted check, the
expected verdict, a one-line description of the change, and the full modified output.

### 2. Hand-scored originals — the false-alarm test

Ten unmodified analyses (eval_01, 03, 06, 08, 11, 14, 16, 18, 19 and H3: every scenario type,
and suppliers with no, some and only Critical items) are scored on every applicable check, 43
in all, by a person with purchasing domain knowledge. The sheet shows each check worded exactly
as the judges receive it, and the applicable checks are chosen by the same code the judges use
(`scripts/hand_scoring.py sheet`); `scripts/hand_scoring.py collect` writes the verdicts to
`eval_data/hand_scores.jsonl`. These give the correct verdict where the
analysis is sound, so a judge that fails a good analysis can be caught.

Judge agreement is deliberately **not** used as ground truth here: that would test the judges
against themselves.

## Candidates

**Revised 2026-10-24 for prompt v5**, with the analysis model now `gpt-6.1-sol` (OpenAI).

| Model | Price (input / output, $ per 1M) | Role |
|---|---|---|
| `claude-haiku-5-5` | 0.10 / 0.50 | production candidate (the current judge) |
| `gemini-3.1-flash-lite` | 0.25 / 1.50 | production candidate; no thinking tokens; its availability is measured too (no-answer rate) |
| `gpt-6-luna` | 0.10 / 0.50 | for reference only: of the analysis model's family, so not a production candidate |

Left out after a first look (2026-10-24): `claude-sonnet-5-5` (2 / 10) and Gemini Flash
(`gemini-3.8-flash`, 0.75 / 3.75 plus about 800 thinking tokens per call) on cost: as the judge
they would add about 125% and 60% to a supplier's total cost, against about 6% for Haiku (judge
cost per 1,000 supplier analyses, estimated: Haiku $2.5, Luna $3, Flash $24, Sonnet $50, against
about $40 for the whole pipeline with Sol); Flash also on availability. Sonnet was first kept as an
accuracy reference but also refuses the judge's forced tool call (it thinks by default), and with
the tool left to its choice it broke the verdict format. `gemini-3.1-flash-lite` answered a test
judgement of fourteen checks in 2.6 s and caught the seeded fault.

**How they were chosen:**
1. **Not the analysis model, and not its family, for the production judge.** LLM judges favour
   their own outputs (self-enhancement or self-preference bias): Zheng et al., "Judging
   LLM-as-a-Judge with MT-Bench and Chatbot Arena", NeurIPS 2023 Datasets and Benchmarks
   (arXiv:2306.05685); Panickssery, Bowman and Feng, "LLM Evaluators Recognize and Favor Their Own
   Generations", NeurIPS 2024 (arXiv:2404.13076), who find that the more a model recognises its own
   outputs, the more it favours them. Wataoka, Takahashi and Ri, "Self-Preference Bias in
   LLM-as-a-Judge" (arXiv:2410.21819, preprint), trace it to judges rating text that is familiar to
   them (lower perplexity) higher whoever wrote it, which is why a model of the same family, trained
   and tuned alike, is also left out. The checks most exposed here are those that need a judgement
   (J2, J9, J11); those that compare the text with its inputs (J1, J6, J7a) less so. It matters most
   where the judge compares analysis models: an OpenAI judge scoring Sol against Sonnet could favour
   Sol. The analysis comparisons so far used `claude-haiku-5-5`, of Sonnet's family, which could
   have favoured Sonnet (`docs/analysis_model_comparison.md`, Limits).
2. **A cost ceiling for production.** The judge runs on every analysis, again on every rewrite, and
   on the purchase request; at `claude-haiku-5-5` prices that is about $0.0025 per supplier, at
   Sonnet's about $0.05, more than the analysis itself. Production candidates are priced at the
   mini tier (0.75 / 4.50) or below; Sonnet is the reference.
3. **Available when needed.** Gemini refused four of six analysis calls for demand on 2026-10-24
   (`docs/analysis_model_comparison.md`); it stays in so that its no-answer rate is measured rather
   than assumed.
4. Other providers (Mistral, DeepSeek, xAI, Cohere, Amazon, Llama hosts) and an open model served
   on the project's own GPU (`docs/inference_serving_vllm.md`) were considered; none has an account
   in this project yet, so they wait.

**Offline judge panel** (comparing analysis models): one judge per provider, so that no family's
preference decides a comparison.

Each candidate judges every calibration output **three times**, with the same prompt the
production evaluation node uses (`services/quality_checks.py`), in two ways: all of an output's
checks in one call, as now, and one call per group of checks (J1-J2, J6-J7a, J8-J11, P1) made at
the same time, which could cut the judge's time for about 1.7 times its cost (+3% of a
supplier's total). Run plan, settings and every verdict: `scripts/run_judge_validation.py`,
`eval_results/judge_validation_v5_settings.json`, `eval_results/judge_validation_v5_runs.jsonl`.

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

| Model | J1 invented | J1 distorted | J1 misattributed | J1b | J2a | J2b | J2c | J3 | J5 omitted | J5 wrong figure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | | | | | | | | | | |

### Decision

*Which judges were chosen, and why.*

## Change log

| Date | Change |
|---|---|
| 2026-10-18 | Design written. |
| 2026-10-19 | Rebuilt on GPT-4o outputs made with prompt v4 (history documents, AvailableStock). Two misattribution faults added for the extended J1, making 35. Hand-scoring sheet rebuilt. |
| 2026-10-18 | 33 seeded errors built from GPT-4o outputs regenerated after the risk mix was spread across all four levels. J1b faults replace the item's notes, mirroring how GPT-4o makes this error. |

# Evaluation record: what was measured, and how much

**2026-10-25.** Every model, setting and prompt in the pipeline was chosen on measurements, not on
defaults: for each agent, the same inputs were run through every candidate, the outputs were scored
by code checks and an LLM judge (itself validated first), and every output was read by hand. This
page counts that work, from the files kept in `eval_results/`, `eval_data/` and `dev/runs/`. The
reasoning behind each choice is in the documents named under each section.

## In total

| | Count |
|---|---:|
| Outputs written by the agents for evaluation (analyses, purchase requests, supplier emails) | **about 1,360** |
| LLM judge calls | **about 5,480** |
| Check verdicts from those calls (one per check per output per run) | **about 35,500** |
| Outputs read by hand, issue by issue | **about 1,050** (plus 89 verdicts scored by the user) |
| Recorded API cost of the evaluation runs below | **about $9.40** (development runs of the analysis not included) |

Counted from the record files on 2026-10-25. "Judge call": one call that scores one output on all
its checks. "Check verdict": one pass/fail on one check.

## 1. The analysis agent

| | Count | Where |
|---|---:|---|
| Development runs of the whole pipeline (Kestrel, Nordic, Hanbit cases; analysis model rounds 1-3; prompt changes) | 87 recorded runs | `dev/runs/2026-10-2*` |
| Analyses of the 25 evaluation scenarios, rebuilt per prompt version | 149 (5 sets of 25, one stopped at 24; plus GPT-4o's 25) | `eval_results/v5_base_outputs*.jsonl`, `eval_v2_gpt4o_outputs.jsonl` |
| Models compared | `gpt-4o`, `gpt-6-luna`, `gpt-6.1-sol`, `claude-sonnet-5-5`, Gemini Flash; effort set per model | `docs/analysis_model_comparison.md` |

Chosen: `gpt-6.1-sol`, effort medium (`docs/analysis_model_selection.md`). **What decided it:** once
the newer models were equally accurate, Sol wrote the actions the buyer can act on, with deadlines as
dates ("obtain a delivery commitment for no later than 6 September 2025"), where Luna wrote "confirm
the revised lead time". **What it taught:** several prompt revisions did not fix GPT-4o's date
arithmetic; changing the model did. Prompt decisions measured
on the 25 analyses: action timing judged against the latest order date (overdue items with a check
put before their order 9 → 0), quantity changes on UK demand only, estimated sizes that say what they
rest on (changelog items 23-25).

## 2. The judge (the evaluation agent)

| | Count | Where |
|---|---:|---|
| Calibration samples (46 seeded faults + 5 real ones) | 51 | `eval_data/judge_calibration_v5.jsonl` |
| Round 1: Haiku, Gemini Flash-Lite, Luna, three runs | 855 judge calls, 9,705 verdicts | `eval_results/judge_validation_v5_round1_runs.jsonl` |
| Round 2: Haiku with thinking, medium and high effort | 570 judge calls, 6,470 verdicts | `..._round2_runs.jsonl` |
| False-alarm test on six real outputs scored by hand | 89 hand verdicts; 72 judge calls, 1,068 verdicts | `eval_data/hand_scores_v5.jsonl`, `eval_results/judge_false_alarm_v5_runs.jsonl` |
| P1 rewording, two wordings | 84 judge calls, 252 verdicts | `eval_results/judge_p1_rewording*_runs.jsonl` |
| Recorded cost | $2.96 | |

Chosen: `claude-haiku-5-5`, thinking at high effort, scoring-sheet format (`docs/judge_selection.md`).
**What decided it:** letting the same Haiku think raised its detection from 83.7% to 93.9% and cut its
flips from 17.6% to 3.9%; round 1 had been unfair to it, its answer forced through a tool call that
leaves no room to think. **What it taught:** how a judge is made to answer matters as much as the
model; warnings favour recall over precision.

## 3. The purchase request writer (approval agent)

| | Count | Where |
|---|---:|---|
| Purchase requests written (6 rounds; up to 9 model and effort settings) | 600 | `eval_results/approval_model_comparison*.jsonl` |
| Judge calls on them (P1a-c, three runs each) | 1,800 (5,400 verdicts) | `eval_results/approval_model_judged*.jsonl` |
| Read by hand, issue by issue | about 525 | `eval_results/approval_model_issues_*.json`, `docs/approval_model_comparison.md` |
| Recorded cost | $2.03 writing, $1.64 judging | |

Chosen: `gpt-6.1-sol`, effort low (`docs/approval_model_selection.md`). **What decided it:** Sol carried
the analysis's facts, figures and conditions over exactly, where Luna gave a bridging date as the
order's delivery and `gpt-4o` added "a proven track record, supporting the decision to proceed".
**What it taught:** the scores and the texts disagreed (`gpt-5.4-mini` kept the length rule best and
wrote the worst text), and the recurring faults came from the prompt and the analysis agent upstream,
fixed there.

## 4. The supplier email writer and its validator

| | Count | Where |
|---|---:|---|
| Emails written (3 rounds x 7 settings x 25), each through the production validator loop | 525 emails, 533 drafts with rewrites, 533 validator checks | `eval_results/email_model*_comparison.jsonl` |
| Judge calls on them (E1, E2, E4-E7, three runs each; four scorings as the checks were corrected) | 2,100 (12,600 verdicts) | `eval_results/email_model*_judged.jsonl` |
| Read by hand, issue by issue | 525 | `docs/email_model_comparison.md` |
| Recorded cost | $0.67 writing and validating, $2.10 judging | |

Chosen: `gpt-6.1-sol`, effort low (`docs/email_model_comparison.md`). **What decided it:** Sol rewrote
the analyst's notes on a supplier's past problems as ordinary questions about this order ("how
terminal integrity will be verified"), where Luna left the trace ("the cause of the loose-terminal
failure"). What the reading found besides: `gpt-5.4-mini`, the email model until now, kept copying the
analyst's notes about a supplier's past problems into the email ("the loose-terminal issue in the
January PowerModuleZ batch") whatever the prompt said; one `gpt-6-luna` setting wrote raw JSON into
the email until the parser was fixed; the production validator passed all 525 emails, catching none
of these.

## What the numbers stand for

- **Same inputs for every candidate.** Each comparison used one fixed set of inputs (the 25 evaluation
  scenarios, or their saved analyses), so differences come from the model, the effort or the prompt.
- **Effort and temperature set and recorded per model**, after checking what each API accepts.
- **The judge was validated before it was used**: seeded faults for detection, hand-scored outputs for
  false alarms, repeated runs for flip.
- **Every output was also read.** The reading changed the outcome more than once: it ruled out the
  model the metrics favoured for the purchase request, traced recurring faults to the prompts and to
  the analysis agent's actions, and showed where the judge's checks themselves were wrong (an email
  check that failed questions the prompt required, corrected and scored again).
- **Every step is kept**: the prompt text of each run, every output, every verdict, the settings.

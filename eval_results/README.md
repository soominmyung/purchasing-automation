# Evaluation results

Raw results behind the model choices in `docs/`. Every input is synthetic (`eval_data/`). The
`.jsonl` files hold every output and verdict; the `.md` files show the same outputs side by side for
reading; each `*_settings.json` records the models, check wording and start time of its run.

Numbers in file names (`_v1`, `_v2`, `_v3`, `_v3b`) are rounds within prompt version 5, not prompt
versions.

## Analyses

| Files | What they hold |
|---|---|
| `v5_base_outputs*.jsonl` | `gpt-6.1-sol` analyses of the 25 evaluation scenarios, the inputs of the purchase request and email comparisons; rebuilt after each analysis prompt change (the suffix names the change) |

## Judge (`docs/judge_selection.md`)

| Files | What they hold |
|---|---|
| `judge_validation_v5_round1_*`, `judge_validation_v5_round2_*` | Detection and flip on the 51 seeded samples (`eval_data/judge_calibration_v5.jsonl`): round 1 without thinking, round 2 with thinking |
| `judge_false_alarm_v5_*` | False alarms against the 89 verdicts scored by hand (`eval_data/hand_scores_v5.jsonl`) |
| `judge_p1_rewording*` | The two wordings tried for P1a |

## Purchase request (`docs/approval_model_selection.md`)

`approval_model_comparison*` are the texts written, `approval_model_judged*` the same texts with the
judge's verdicts and the issues found by reading, `approval_model_issues_*.json` those issues on
their own.

| Suffix | Run |
|---|---|
| (none) | The first prompt: four models, then every model and effort |
| `_revised_v1`, `_revised` | The revised approval prompt: first wording, kept wording |
| `_timing_v1`, `_timing` | After the analysis prompt's timing rule: first wording, kept wording (the final comparison) |
| `_estimate` | After the analysis prompt's rule on estimated sizes |

## Supplier email (`docs/email_model_selection.md`)

| Files | Run |
|---|---|
| `email_model_v1_*`, `email_model_v2_*` | Rounds 1 and 2 |
| `email_model_comparison.*` | Round 3, the final emails |
| `email_model_v3_*`, `email_model_v3b_*`, `email_model_judged*` | Round 3 scored three times as the check wording was corrected; the last is final |
| `email_validator_comparison*` | The validators compared on round 3's emails and planted faults |

## Self-hosting (`docs/serving_gemma4_vllm.md`)

| Files | What they hold |
|---|---|
| `sft_screen_<model>*.jsonl` | Open models and SFT adapters on the 25 evaluation cases with the production checks. Tags: `fac-rule` the current prompt; `tie-in` and `in-dist` other scenario sets; `serve-bf16`, `serve-fp8` the serving comparison; no tag, the first screening round |
| `serving_bench_gemma4-12b-{bf16,fp8}.json` | vLLM throughput and latency, 1–16 concurrent requests |

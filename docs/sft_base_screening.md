# SFT base-model screening: five candidates, untuned

Run on 2026-10-25, before any training, to choose the base model for the SFT retraining
(`docs/sft_plan.md`, section 5). Every candidate was run as the analysis agent, untuned, on the
25 evaluation scenarios (`eval_data/eval_scenarios.jsonl` 20, `holdout_scenarios.jsonl` 5). The
prompt was the production prompt (v5), and the production code path was used
(`services/agents.run_analysis_agent`, model `local:<name>`). The first answer was scored with the
production checks (code checks C1–C10 and the judge, `claude-haiku-5-5` at high effort). There were
no rewrites. The reference is `gpt-6.1-sol`'s outputs on the same prompt
(`eval_results/v5_base_outputs_estimate.jsonl`), scored the same way.

## Results

| Model | Size | Parsed | **Ready** (every blocking check passed) | Blocking failures by check | Time per analysis (median) |
|---|---|---|---|---|---|
| `gpt-6.1-sol`, medium (reference) | — | 25/25 | **25/25** | none | 13.4 s |
| **`google/gemma-4-12B-it`** | 12.0B | 25/25 | **15/25** | J1b 4, C10 4, J8 2, J1c 1, C6 1, J6b 1 | 11.4 s |
| `Qwen/Qwen3.5-9B` | 9.7B | 25/25 | 4/25 | C10 16, J7a 14, J6b 9, J1b 5, C5 4, J1a 3, C4 3, J8 2, C2 1, J1c 1, J6a 1 | 11.6 s |
| `google/gemma-4-E4B-it` | 4.5B effective | 25/25 | 2/25 | C5 14, C4 13, J8 8, C10 7, J6b 7, J7a 6, C6 3, J1b 3, C2 3, J1c 2, J6a 2 | 9.1 s |
| `mistralai/Ministral-3-14B-Instruct-2512` | 13.9B | 25/25 | 2/25 | C10 18, J7a 10, J6b 9, J8 6, J1b 5, C5 3, J1c 2, C4 2, J1a 1, C6 1, J6a 1 | 17.5 s |
| `mistralai/Ministral-3-8B-Instruct-2512` | 8.9B | 25/25 | 0/25 | C10 23, J7a 15, J8 15, J6b 12, C5 6, J1b 5, C4 5, J1a 5, J6a 3, J1c 2, J6c 1 | 14.9 s |

Every model wrote valid JSON in every case, so the differences are in content, not format.

## The figures after a change (C10), split into its two kinds

C10 compares the model's `figures_after_changes` with code's, value by value. It failed in two
different ways:

| Model | Figures written for items whose lead time was not changed | A figure wrong |
|---|---|---|
| `gpt-6.1-sol` | 0 | 0 |
| Gemma 4 12B | 3 | **1** |
| Gemma 4 E4B | 4 | 3 |
| Qwen3.5-9B | 5 | 11 |
| Ministral 3 14B | 7 | 11 |
| Ministral 3 8B | 11 | 13 |

**Figures for unchanged items.** This is a format slip. The prompt's output template shows
`figures_after_changes` filled in (line 83), and its rule "[] when there is no lead-time change"
comes 40 lines later (line 123). For example, in eval_02 Ministral 3 8B returned
`lead_time_changes: []` and still gave both items "figures after changes" at 18 weeks. Gemma 4 12B,
Qwen and Ministral 3 14B returned `[]` there. Training on the teacher's outputs, which always follow
the rule, is likely to fix this.

**A figure wrong.** This is arithmetic done without reasoning. The rules are in the prompt's STOCK
POSITION section. The usual errors:

- an order date left before the snapshot date instead of set to it. Ministral 3 8B, eval_08:
  `new_latest_po_date 2026-02-15, by the rules 2026-03-02`;
- the risk level not re-banded at the new lead time. The same case: `new_risk_level High, by the
  rules Critical`.

`gpt-6.1-sol` reasons before answering. The candidates were run with thinking off, as the training
targets are final answers. GPT-4o, also run without reasoning, failed the same way on the same
prompt: 4–8 of 20 values wrong, mostly order dates (changelog item 23). The problem is a model
without reasoning writing multi-step date arithmetic straight out, not open models as such.
Training may not fix it fully. So the learning curve will report the two kinds separately
(`docs/sft_plan.md`, section 3).

## What the leader still gets wrong (Gemma 4 12B, read)

- **J1b, documents slightly strengthened (4).**
  - eval_10: the document says "occasional delays" and a "temporary reduction in product quality";
    the summary drops "occasional" and presents both as ongoing.
  - H3: an undated past halt ("faced a halt") is written as "currently experiencing a significant
    production halt".
- **J8, internal figures in a supplier question (2).** eval_06: "to bridge the 13-week gap".
- **C10 (4).** Three are figures for unchanged items, and one is a wrong value. H2, PartW at
  20 weeks: High where the rules give Critical, and the order date a week late.

These are the kinds of fault the teacher's verified outputs show the right way every time.

## Why the others fall behind

- **Qwen3.5-9B:** real arithmetic errors in 11 cases. eval_08 has four wrong values in one answer:
  an order date, a risk level, a gap of 31 weeks where the rules give 0, and a second order date.
- **Gemma 4 E4B:** fails the task's structure, not only the arithmetic.
  - C5 14: it proposes a supplier-wide lead time but gives no item its final lead time.
  - C4 13: findings without a basis, and wrong citations.
- **Ministral 3 8B and 14B:** C10 in 23 and 18 cases, both kinds. Internal figures in supplier
  questions (J8) in 15 and 6. The 14B is also the slowest (17.5 s), at the edge of the card's memory.

## Settings, and where they differed

| | |
|---|---|
| Engine | vLLM `v0.31.0` (`vllm/vllm-openai`), Docker Desktop on Windows, RTX 5090 32 GB (`scripts/serve_local_model.ps1`) |
| Precision | bf16 for every model (the Ministral `-BF16` repositories) |
| Decoding | temperature 0; thinking off where the chat template has it (Qwen3.5, Gemma 4) |
| Calls | one at a time, so the time per analysis is not affected by other requests |
| Max model length, max output tokens | 16,384 / 6,000 for Gemma 4 12B, Gemma 4 E4B and Qwen3.5-9B; 16,384 / 4,000 for Ministral 3 8B; **8,192 / 3,000 for Ministral 3 14B**: its bf16 weights (about 26 GiB) left no room for a longer KV cache on the 32 GB card, of which Windows holds about 1.6 GB |
| Ministral | served with `--tokenizer-mode mistral --config-format mistral --load-format mistral`; no chat-template arguments (vLLM's Mistral tokenizer rejects them; the first 8B attempt failed all 25 requests on this and was discarded) |

Outputs are 1–2k tokens, so the lower output limits did not cut any answer.

## Files

- Results, one line per case with input, output and every check: `eval_results/sft_screen_<model>.jsonl`.
- Scripts: `scripts/screen_base_models.py` (`--model`, `--sol`, `--report`),
  `scripts/serve_local_model.ps1`, `scripts/screen_all_candidates.ps1`.
- Code: `services/agents._llm` takes `local:<name>`; the address is in `config.py`
  (`local_llm_url`, `local_llm_api_key`, `local_llm_max_tokens`).

## Round 2: the figures-after-changes rule made exact (2026-10-25)

Round 1 showed several models copying the filled-in output template for items whose lead time
was not changed. The rule was rewritten to match C10 exactly ("one entry for each item in
lead_time_changes, and for no other item. [] when lead_time_changes is []";
`prompts/CHANGES.md`), and all five models and Sol were run again. Results:
`eval_results/sft_screen_<model>_fac-rule.jsonl`.

| Model | Ready, round 1 → 2 | Figures for unchanged items, 1 → 2 | Of Sol's 11 lead-time changes, also changed | Wrong figures / figures checked (round 2) | Time (median) |
|---|---|---|---|---|---|
| `gpt-6.1-sol` | 25 → 24 | 0 → 0 | 11 | 0/88 (0%) | 14.1 s |
| **Gemma 4 12B** | 15 → **14** | 3 → 1 | **2** | 2/12 (17%) | 11.6 s |
| Qwen3.5-9B | 4 → 8 | 5 → **0** | 11 | 29/92 (32%) | 11.5 s |
| **Gemma 4 E4B** | 2 → 3 | 4 → 4 | **10** | **8/104 (7.7%)** | 8.6 s |
| Ministral 3 14B | 2 → 0 | 7 → 8 | 11 | 32/88 (36%) | 20.6 s |
| Ministral 3 8B | 0 → 0 | 11 → 12 | 11 | 36/112 (32%) | 14.6 s |

Sol's one failure in round 2 is a judge reading. In eval_01 its summary says "only one week between
the standard lead time and stockout" (19 weeks of stock, 18-week lead time), which is right, and
J7a took it against a gap of 0 weeks. Differences of one case are within run-to-run variation (Sol
25 → 24, Gemma 4 12B 15 → 14).

**The rule change worked where a model followed rules.** Qwen went from 5 such cases to 0, and from
4 to 8 ready; Gemma 4 12B from 3 to 1. Ministral 3 8B and 14B and Gemma 4 E4B did not change, so for
them it is instruction-following, not wording. Wrong figures were unchanged for every model, as
expected: that is arithmetic, not the prompt.

### Readiness alone misleads: Gemma 4 12B rarely changes a lead time

Sol changed the lead time in 11 cases, on delivery-delay and lead-time notices in the history
(eval_08–13, H1–H5). Gemma 4 12B changed it in 2 of them and missed eval_08–12 and H1, H3, H4, H5.

A case with no lead-time change has no figures after a change to get wrong (C10) and no change size
to judge (J6b). A missing change is not checked: the user decided on 2026-10-25 that there is no
judge check for it. So part of Gemma 4 12B's 14 ready cases are changes it did not make. In
production that is a serious miss: an order placed by an 18-week date for an item whose supplier
has announced 22 weeks arrives late.

**Why Gemma 4 12B made so few changes: it saw the delays but would not estimate a size.** The
history documents in these cases name a delay but give no size: "recent logistical complications
have led to delays in shipping schedules" (eval_08), "significant delivery delays due to a recent
strike … not been resolved" (eval_11), "a significant production halt" (H3). Sol applied a
provisional allowance (20 or 22 weeks), with the reason saying it is an analyst's estimate to be
confirmed, as the prompt allows. Gemma 4 12B:

- recorded the delay as a finding in 8 of the 9 cases;
- in 5 of them, asked the buyer to confirm the timing before ordering, for example eval_11: "Confirm
  current lead times and delivery schedules for this order before placing, given the unresolved
  warehouse strike";
- did not put a number on it.

This is the behaviour the user decided against on 2026-10-25: confirm the delay, and allow for it
in the meantime. So it is a policy the prompt states and the model did not follow, not a failure to
notice. The teacher's outputs apply the allowance every time, so it is the kind of behaviour
training should change; the change-detection measure will show whether it did.

Gemma 4 E4B is the opposite. It changed the lead time in 10 of Sol's 11 cases and had the lowest
error rate on the figures it worked out (8 of 104, 7.7%). It fails on the output's structure
instead:

- C5 14: a supplier-wide lead time with no final lead time per item;
- C4 11: findings without a basis, or wrong citations.

**Wrong figures on the same items.** The "figures checked" column differs between models because
each model changed the lead time of different items (four figures per changed item), so its rates
are not on the same items. On the items Sol changed (11 cases, 22 items), counting only those the
model changed too:

| Model | Of Sol's 22 items, also changed | Of those, an item with a wrong figure |
|---|---|---|
| Gemma 4 E4B | 21 | **3 (14%)** |
| Gemma 4 12B | 3 | 1 (33%; too few to judge) |
| Qwen3.5-9B | 22 | 21 (95%) |
| Ministral 3 14B | 22 | 22 (100%) |
| Ministral 3 8B | 22 | 22 (100%) |

**Measure to add.** The SFT results report change detection beside readiness: Sol's lead-time
changes, and the scenarios' planted labels (Sol is the reference, not the truth). This is a
reported measure, not a new blocking check.

## Performance by model, in short

| Model | What it does well | What it gets wrong | Example |
|---|---|---|---|
| `gpt-6.1-sol` (reference) | every check, every figure right; applies a provisional lead time with its basis and asks to confirm it | — (one judge reading in round 2) | eval_11: 22 weeks for an unresolved strike, "an analyst estimate … to be confirmed" |
| **Gemma 4 12B** | the most often ready (14–15/25); faithful findings; the output's format | puts no number on an unsized delay, only asks to confirm it (9 of Sol's 11 changes); slight strengthening of documents (J1b 4–5); internal figures in a supplier question (J8 2–3) | H3: an undated past halt written as "currently experiencing a significant production halt"; eval_06: "to bridge the 13-week gap" in a supplier question |
| **Gemma 4 E4B** | applies the lead-time changes (10 of 11); the most accurate figures after a change (3 of 21 items wrong) | the output's structure: no final lead time per item (C5 14); citations (C4 11–13) | eval_01: a supplier-wide lead time with no item given its final lead time |
| Qwen3.5-9B | applies every change (11 of 11); follows the revised rule (0 figures for unchanged items) | date arithmetic: 21 of 22 changed items with a wrong figure | eval_08: four wrong values in one answer (an order date, Critical written as High, a 31-week gap where the rules give 0, a second order date) |
| Ministral 3 14B | applies the changes (11 of 11) | arithmetic (22 of 22 items wrong); figures for unchanged items (8); internal figures to the supplier (J8 6–10); slowest (17–21 s) | eval_09: Critical written as Medium, High as Low |
| Ministral 3 8B | applies the changes (11 of 11) | arithmetic (22 of 22 items wrong); figures for unchanged items in almost every case (11–12); J8 14–15 | eval_02: no lead-time change, yet 18-week "figures after changes" for both items |

## Decision

**Gemma 4 12B (`google/gemma-4-12B-it`), decided by the user on 2026-10-25.**

- **Its main gap turned out to be fixable.** Reading the outputs showed that Gemma 4 12B found the
  delays: it recorded them as findings in 8 of the 9 cases, and in 5 asked for the lead time to be
  confirmed before ordering. What it did not do was put the provisional allowance into the lead
  time. That is a policy stated in the prompt and shown in every teacher output, the kind of
  behaviour SFT is likely to change, not a failure to notice.
- **Everything else it already does best.** It was the most often ready (14–15/25), with faithful
  findings, the output's format kept, and one arithmetic slip.
- **Gemma 4 E4B was set aside despite the most accurate figures.** It is built for phones and
  laptops, with about 4.5B effective parameters. Its failures are in holding the prompt's many
  rules at once: no final lead time per item in 14 cases, wrong citations in 11–13. A model that
  small has less room to improve with training, and the rules will only grow.
- **Qwen3.5-9B and Ministral 3** apply the changes but get nearly every changed item's figures
  wrong (21–22 of 22). That is arithmetic without reasoning, which training on final answers is
  unlikely to fix.
- **It fits the deployment.** At 12B it is about 12 GB at fp8, served on one L4.

The SFT results will report, beside readiness, whether training made it apply the allowances: its
change detection against Sol's changes and the scenarios' labels.

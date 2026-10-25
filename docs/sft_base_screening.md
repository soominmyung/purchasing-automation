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

## Decision

**Open, for the user.** The screening points to **Gemma 4 12B**:

- 15 of 25 ready untuned, against 4 or fewer for every other candidate;
- one arithmetic error;
- the rest are faults that the teacher's verified outputs show the right way every time.

It is 12B, about 12 GB at fp8, so it fits an L4 for serving.

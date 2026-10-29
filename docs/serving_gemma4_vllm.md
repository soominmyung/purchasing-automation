# Self-hosted serving with vLLM: Gemma 4 12B, bf16 against fp8

A self-contained record of the serving measurement made on 2026-10-26: what was served, on what
data, how quality and speed were measured, and what came out. It is the serving half of the
self-hosting question; the model half is summarised in section 8.

## In short

Gemma 4 12B (untuned) was served by vLLM on one RTX 5090, in bf16 and in fp8, and measured on
the pipeline's real analysis request.

- **Memory:** fp8 used 13.3 GiB against 23.9 GiB, 44% less, which fits an L4.
- **Throughput:**
  - fp8 gave 1.6–1.8× the throughput of bf16, with lower latency;
  - continuous batching raised throughput 8.6× (bf16) and 10× (fp8) from 1 to 16 concurrent
    requests.
- **Quality:**
  - measured with the production checks on the 25 evaluation cases, fp8 was ready in 13 cases
    against 14 and 15 in two bf16 runs;
  - the two bf16 runs themselves differ in 3 cases, so the fp8 cost is about the size of
    run-to-run variation;
  - fp8's extra failures are one omission ("to be confirmed" left off an estimated size), not
    wrong judgements.

## 1. Why this was measured

The analysis agent runs on `gpt-6.1-sol` through an API. Serving a model in-house would keep the
analysis input (every item's figures and every history document) away from model companies, fix
the cost, and remove the dependence on an outside API. This
measurement answers the serving half:

- how much memory and speed self-hosting takes;
- what fp8 quantization costs in quality.

The question of whether an open model is good enough for the role is answered separately.

## 2. What was served

| | |
|---|---|
| Model | `google/gemma-4-12B-it`, untuned: the best of five open-weight models screened untuned on the same 25 cases |
| Engine | vLLM `v0.31.0` (Docker image `vllm/vllm-openai:v0.31.0`), OpenAI-compatible API |
| GPU | NVIDIA RTX 5090, 32 GB (Windows holds about 1.6 GB) |
| Variants | **bf16** (`--dtype bfloat16`); **fp8** (the same, plus `--quantization fp8`: weights quantized to fp8 when loaded) |
| Server flags | `--max-model-len 16384 --gpu-memory-utilization 0.92 --seed 0` (`scripts/serve_local_model.ps1`) |
| Decoding | temperature 0, thinking off (`enable_thinking: false`), as the pipeline calls it |
| How the pipeline calls it | the production analysis code path, with the analysis model set to `local:gemma-4-12b-it` (`services/agents._llm`, `config.local_llm_url`) |

## 3. The data

**The 25 evaluation cases** that every model in this project was compared on. None was used in training.

- **Two sets:** 20 generated cases (`eval_data/eval_scenarios.jsonl`) and 5 held-out cases
  (`eval_data/holdout_scenarios.jsonl`).
- **Types:**
  - clean (5), no history (2), ambiguous attribution (3);
  - a clear supplier incident (3), a clear item incident (3);
  - high stock with short cover (2), missing values (2);
  - and the 5 holdout cases.
- **53 items:**
  - Critical 23, High 13, Medium 10, Low 7;
  - 11 cases where `gpt-6.1-sol` changed a lead time on delays in the history.
- **Each case as the model receives it:** the production system prompt (`prompts/v5/analysis.md`) and the pipeline's user JSON:
  - snapshot date and supplier;
  - each item's figures as code works them out (risk level, gap, latest order and delivery
    dates, suggested quantity);
  - the numbered history documents.
- **Size:** about 4,400 prompt tokens per request.

## 4. How quality was measured

Each case went through the production analysis code path once, with no rewrite, and its first
answer was scored by the production checks (`services/quality_checks.evaluate_analysis`):

- **Code checks:**
  - C1 format;
  - C2 item codes;
  - C3 quantity starting points;
  - C4 citations (every finding rests on a document that applies, every question, change and
    action cites a finding);
  - C5 well-formed changes;
  - C6 an action for every Critical item;
  - C9 arrows in the text;
  - C10 the figures after a lead-time change, recalculated by code and compared value by value.
- **Judge checks**, by `claude-haiku-5-5` at high effort, a different model family from the
  writer:
  - J1a–c: history statements found in the documents, not distorted, not misattributed;
  - J3a–c: each change follows from its findings, its size fits them (or is marked as an
    estimate to be confirmed), and no item the reason applies to is left out;
  - J4: figures quoted in the text match the figures after the changes;
  - J5: questions marked for the supplier reveal nothing internal.
- **The judge was validated before use** (`docs/judge_selection.md`): it caught 93.9% of 51 seeded
  faults, with 8.0% false alarms on 89 hand-scored verdicts.
- **Verdict:** ready only if every blocking check passed.

fp8 and the second bf16 run were run 6 cases at a time; the first bf16 run is the screening
run of the untuned model, one case at a time, with the same model, prompt and server flags. Comparing
the two bf16 runs measures run-to-run variation: decoding is not exactly repeatable even at
temperature 0, least of all under batching.

## 5. How speed and memory were measured

- **Memory:** vLLM's own start-up report of the memory taken by weights and runtime.
- **Throughput** (`scripts/bench_vllm_v5.py`):
  - **requests:** the same production requests, one warm-up request, then 1, 2, 4, 8 and 16
    concurrent requests;
  - **load:** 8 requests at levels 1–4, 16 at level 8, 32 at level 16;
  - **settings:** answers up to 1,500 tokens, temperature 0;
  - **reported:** output tokens per second across all requests at each level, and p50 and p95
    latency per request.

## 6. Results

### Memory

| | bf16 | fp8 |
|---|---|---|
| Weights and runtime | 23.9 GiB | **13.3 GiB** |

At bf16 the model does not fit a 24 GB L4; at fp8 it does, with room for the KV cache.

### Throughput and latency

| Concurrent requests | bf16 tok/s | fp8 tok/s | fp8 ÷ bf16 | bf16 p50 / p95 | fp8 p50 / p95 |
|---|---|---|---|---|---|
| 1 | 52.8 | 82.8 | 1.57× | 9.1 / 12.8 s | 5.8 / 7.6 s |
| 2 | 94.3 | 141.6 | 1.50× | 9.3 / 13.0 s | 5.9 / 7.7 s |
| 4 | 146.0 | 227.1 | 1.56× | 9.0 / 12.9 s | 6.1 / 7.9 s |
| 8 | 315.4 | 514.9 | 1.63× | 12.0 / 19.5 s | 7.9 / 12.0 s |
| 16 | 453.5 | 834.5 | 1.84× | 16.1 / 22.2 s | 8.6 / 14.2 s |

- **Continuous batching:** from 1 to 16 concurrent requests, throughput rose 8.6× at bf16 and 10×
  at fp8, while latency per request rose 1.8× (bf16) and 1.5× (fp8).
- **fp8:** faster at every level, and its advantage grows with load, where memory bandwidth and
  KV-cache room matter more.

### Quality: production checks, 25 cases

| | Ready | Lead-time changes applied (of Sol's 11) | Blocking failures by check |
|---|---|---|---|
| bf16, run 1 (one case at a time) | 14 | 1 | J1b 5, J5 3, J1c 2, C10 2, J3b 2, C5 1, J1a 1, C6 1 |
| bf16, run 2 (6 at a time) | 15 | 1 | J1b 5, C10 3, J5 2, C5 1, J1c 1, C6 1, J3b 1 |
| fp8 (6 at a time) | 13 | 3 | J1b 6, J3b 5, J1a 2, C10 2, J5 1, J4 1, J1c 1 |
| *`gpt-6.1-sol` (reference, API)* | *24* | *11* | *J4 1 (a judge reading)* |

**Case by case.** 9 cases were ready in all three runs, and 4 failed the same way in all three
(eval_06, 09, 18, 19).

- **The two bf16 runs disagree on 3 cases:** eval_02 (J1b in run 1, ready in run 2), eval_14 (J3b,
  then ready), H1 (ready, then J1b).
- **fp8 against bf16:**
  - ready in fp8 only: H3, and fewer failures in eval_10 and H2;
  - failed in fp8 only: eval_04, 11, 15, 16, and eval_01 with different checks.
- **What fp8's extra failures are.** All five of its J3b failures are one omission. It proposed a
  size the documents do not give, and did not mark it as its estimate to be confirmed:
  - 22 weeks for an unresolved strike (eval_11, the value Sol chose);
  - 25 weeks (eval_13);
  - 19 weeks (H2);
  - a 50% quantity increase for an unsized surge (eval_15, eval_16).

  In those cases bf16 made no change, so it had nothing to fail. The rest (J1a, J1b, J4) moves
  between runs at bf16 as well.

**Reading:**

- fp8's quality cost on this workload is about the size of the run-to-run variation (13 against
  14–15). It is not shown to be zero.
- The difference is in a wording rule, not in judgement.
- Both precisions are well below `gpt-6.1-sol` (24). That gap is the model's, not the serving's.

## 7. Limits

- **The GPU:** an RTX 5090, not the L4 a deployment would use. Throughput on an L4 would be lower,
  as it has less memory bandwidth, and only fp8 fits.
- **Repeats:** one fp8 run and two bf16 runs. Differences of one or two cases are within
  variation.
- **The model:** untuned. QLoRA fine-tuned adapters were tried and were worse than the untuned
  model (ready in 3–10 of 25 against 14), so none is used (`docs/sft_gemma4.md`).
- **Speed is the engine's, not the pipeline's.** In the pipeline the judge (an API) is called
  after the analysis.

## 8. Conclusion

- **The serving side was measured as feasible.** In fp8 the model takes 13.3 GiB, within one 24 GB
  card, and on the RTX 5090 it answered the production request in about 6 s alone, or at 835 tokens
  per second across 16 requests.
- **The model side is not.** The same checks put the best open model at 13–15 of 25 against Sol's
  24, and fine-tuning did not close the gap.
- **So production stays on `gpt-6.1-sol` for now** (decided 2026-10-26), and serving is
  the measured option if an open model closes the gap.

## 9. Files and how to repeat it

| What | Where |
|---|---|
| Serve one model | `scripts/serve_local_model.ps1 -Model google/gemma-4-12B-it -Name gemma-4-12b-it [-Extra "--quantization","fp8"]` |
| The whole comparison (bf16 then fp8: quality, then throughput) | `scripts/serve_quant_compare.ps1` |
| Quality runs | `scripts/screen_base_models.py --model local:gemma-4-12b-it --tag serve-bf16 --workers 6`; tables from `scripts/report_sft_results.py` |
| Throughput | `scripts/bench_vllm_v5.py --model gemma-4-12b-it --label gemma4-12b-bf16` |
| Results | `eval_results/serving_bench_gemma4-12b-{bf16,fp8}.json`; `eval_results/sft_screen_gemma-4-12b-it_{fac-rule,serve-bf16,serve-fp8}.jsonl` (bf16 run 1 is the screening run, tag `fac-rule`) |

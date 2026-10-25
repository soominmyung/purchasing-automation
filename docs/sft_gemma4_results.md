# Gemma 4 12B SFT (v5 analysis agent): setup and results

The record of the SFT retraining done on 2026-10-25. It covers the plan (`docs/sft_plan.md`), the
base model choice (`docs/sft_base_screening.md`), the training data (`docs/sft_training_data.md`),
the training settings, the learning curve and the evaluation. **Evaluated on 2026-10-25 (all six adapters): at this size SFT did not
improve on the untuned model.** Tables
come from `python scripts/report_sft_results.py`.

## 1. The question

Can a self-hosted open-weight model take `gpt-6.1-sol`'s place as the analysis agent, measured by
the production checks on the cases every model was compared on? The analysis call receives the
most company data (every item's figures and every history document), so serving it in-house
would keep that data away from model companies (`docs/sft_plan.md`, section 6.3).

## 2. Base model

**Gemma 4 12B (`google/gemma-4-12B-it`)**, chosen after screening five open-weight models untuned
on the 25 evaluation cases with the production prompt and checks (`docs/sft_base_screening.md`).

| Untuned, 25 cases | Ready | Lead-time changes applied (of Sol's 11) | Changed items with a wrong figure |
|---|---|---|---|
| `gpt-6.1-sol` (reference) | 24 | 11 | 0/22 |
| **Gemma 4 12B** | **14** | **1** | 1/1 |
| Gemma 4 E4B | 3 | 1 | 2/2 |
| Qwen3.5-9B | 8 | 11 | 21/22 |
| Ministral 3 14B / 8B | 0 / 0 | 11 / 11 | 22/22 / 22/22 |

Why Gemma 4 12B:

- It was the most often ready.
- Its main gap was a policy, not a blind spot. It found the delays in the history but put no
  provisional allowance into the lead time, and the teacher's outputs show that allowance every
  time.
- Gemma 4 E4B (about 4.5B effective) broke the output's structure, and the Qwen and Ministral models
  got nearly every changed figure wrong.

## 3. Training data

Full record: `docs/sft_training_data.md`.

- **Scenarios** (`scripts/generate_sft_scenarios.py`): 117 synthetic suppliers, 13 types × 9, with
  2–6 items and 0–11 history documents each (598 documents in all).
  - Names and documents come from `gpt-6.1-sol` (low effort); the stock figures are set in code by
    type.
  - Risk mix: Low 80, Medium 74, High 67, Critical 56.
  - No supplier or product name is shared with the evaluation cases.
- **Teacher** (`scripts/build_sft_teacher_data.py`): `gpt-6.1-sol` at medium effort, as the
  production analysis agent (prompt v5), scored by code checks C1–C10 and the judge
  (`claude-haiku-5-5`, high effort, one call).
  - **98 of 116 passed** and are the examples. One scenario had no usable row and was dropped; the
    18 failures are kept for DPO.
- **Split** (fixed in `scripts/train_sft_v5.py`): 13 validation runs (a tenth per type, at least 4
  runs in the type) and 85 training runs, in a fixed type-interleaved order. The learning curve
  takes the first 25, the first 50, then all 85, so each size contains the smaller ones.
- **Kept out:** the 25 evaluation cases and the 7 tie-in generalisation scenarios.

**What one example is:**

- the production system prompt;
- the pipeline's user JSON (items with code-worked figures, numbered history documents);
- the teacher's raw answer.

These are rendered as Gemma 4's generation prompt (thinking off, which ends in an empty thought
channel) plus the answer and `<turn|>`. The loss is on the answer only.

| | Tokens |
|---|---|
| Typical example | prompt about 4,800, answer about 2,000 |
| Longest example | 8,372 |
| Mean | about 6,300 |

The training text was built so that it matches how vLLM prompts the model in serving. Rendering
the whole conversation with the chat template drops the empty thought channel, which would have
left training and serving mismatched.

## 4. Training settings

| Setting | Value |
|---|---|
| Method | QLoRA: base loaded in 4-bit (Unsloth), LoRA on the language model's attention and MLP layers (vision layers not trained) |
| LoRA | rank 16, alpha 32, dropout 0 (Unsloth's fast path) |
| Optimiser | AdamW 8-bit, learning rate 2e-4, cosine schedule, warm-up 10% of steps, weight decay 0.01 |
| Batch | 1 per device × gradient accumulation 8 = 8 |
| Epochs | 3; validation loss each epoch; the checkpoint with the lowest validation loss kept |
| Max length | 12,288 tokens (no example cut) |
| Precision | bf16 |
| Seeds | 1 and 2 for each size |
| Hardware | RTX 5090 (32 GB), the user's PC; Docker `unsloth/unsloth:core-v0.1.903-beta` (torch 2.11 + CUDA 12.8, Unsloth 2026.10.1, transformers 5.17, TRL 1.13) |
| Run | `scripts/train_learning_curve.ps1` (six runs in sequence); adapters and logs in `C:\sft\runs\<name>` (`train_log.json` holds every run's settings, data ids and loss history) |

Found while setting up:

- the Hugging Face cache needed an explicit `HF_HUB_CACHE` in the container;
- transformers 5 removed `warmup_ratio` (now `warmup_steps`);
- the repository on Google Drive is not mounted into Docker; the script and data are copied to
  `C:\sft\work` instead.

## 5. Learning curve: validation loss

Six runs from 21:43 to 23:20 (1 h 37 min). The best checkpoint was the last epoch's in every run.

| Training runs | Seed 1 | Seed 2 | Mean | Time |
|---|---|---|---|---|
| 25 | 0.460 | 0.444 | **0.452** | 7 min |
| 50 | 0.411 | 0.381 | **0.396** | 13 min |
| 85 | 0.325 | 0.331 | **0.328** | 20 min |

Training loss for the 85-run, seed-1 run: 0.83 → 0.29 over 33 steps.

The loss still falls at 85 runs. The 50 → 85 gain (0.07) is larger than the spread between seeds
(at most 0.03). Validation loss measures closeness to the teacher's answers, not passing the
checks, so whether more data is needed is decided on the evaluation (section 6).

## 6. Evaluation

**Set-up:**

- **Serving:** one vLLM server (v0.31.0, bf16) on the RTX 5090 with the base model and the six
  adapters (`--enable-lora`, `local:sft-n<size>_s<seed>`).
- **Code path and checks:** the same production code path and checks as the screening, on the
  25 evaluation cases and the 7 tie-in scenarios.
- **Concurrency:** 6 cases at a time, so the times are not comparable with the screening's
  one-at-a-time times.

**First adapter: 85 runs, seed 1.**

| 25 cases | Ready | Lead-time changes applied (of Sol's 11) | Changed items with a wrong figure | Blocking failures by check |
|---|---|---|---|---|
| `gpt-6.1-sol` | 24/25 | 11/11 | 0/22 | J7a 1 |
| Gemma 4 12B, untuned | 14/25 | 1/11 | 1/1 | J1b 5, J8 3, J1c 2, C10 2, J6b 2, C5 1, J1a 1, C6 1 |
| **SFT, 85 runs, seed 1** | **3/25** | 3/11 | 5/5 | **J8 17**, J7a 5, C4 4, C10 3, J1c 2, J1b 2, C2 1, J1a 1 |

On the tie-in generalisation set (a type never trained on) the same adapter was ready in 5 of 7,
both failures J8, the same pattern; Sol was ready in 7 of 7.

### Why readiness fell: an action's phrase moved into the supplier questions

The J8 failures are one pattern, in 17 of 25 cases: a question addressed to the supplier ends
with the internal gap. For example, eval_02: "What expedited shipping, earlier partial delivery or
alternative source can you offer for CircuitBoardA **to bridge the 8-week gap**, and what quantities
and delivery dates can you commit to?"

- **The teacher never writes this.** In the 98 training examples, 0 of 156 supplier questions
  contain "bridge". The phrase belongs to the internal actions: 60 of 300 actions read like "Ask
  AcmeParts for expedited shipping of WidgetX … to bridge the 14-week gap", as in the prompt's own
  example. The teacher's supplier questions read "What expedited shipping or early partial delivery
  can you offer for X …".
- **The fine-tuned model merged the two.** 21 of its 33 supplier questions carry the action's
  "to bridge the N-week gap". With 85 examples it learned the shared shape of the two fields but
  not which one may name the gap.
- **The evaluation set exposes it more than the training set would.** 43% of the evaluation items
  are Critical, against 18% in training. Critical items are where expedite questions are asked.

Lead-time changes rose from 1 to 3 of Sol's 11, but all five changed items had a wrong figure,
so the allowance policy and its arithmetic were not learned at this size.

**Status:**

- the other five adapters (seed 2, and 25 and 50 runs) are being evaluated;
- whether the pattern holds across seeds and sizes is to be read from them.

Options under consideration, for the user to decide:

- more data, weighted towards the evaluation's risk mix;
- training examples where a Critical item's supplier question and action both appear;
- DPO on pairs that differ in this phrase.

## 7. Results by size and seed

**25 evaluation cases.** "Ready if J8 set aside" removes the one pattern of section 6 to show
whether anything else improved.

| Model | Ready | Ready if J8 set aside | Lead-time changes applied (of Sol's 11) | Changed items with a wrong figure | Supplier questions with "bridge … gap" | Blocking failures by check |
|---|---|---|---|---|---|---|
| `gpt-6.1-sol` | 24 | 24 | 11 | 0/22 | — | J7a 1 |
| Gemma 4 12B, untuned | **14** | **16** | 1 | 1/1 | — | J1b 5, J8 3, J1c 2, C10 2, J6b 2, C5 1, J1a 1, C6 1 |
| SFT 25 runs, seed 1 | 5 | 14 | 1 | 2/2 | 12/21 | J8 11, C4 4, J7a 3, J1b 2, J1c 2, C10 2, C1 2, J6b 1, J1a 1 |
| SFT 25, seed 2 | 6 | 16 | 2 | 3/3 | 16/29 | J8 14, C4 5, J7a 3, C10 2, J1c 1, J1b 1, C2 1 |
| SFT 50, seed 1 | 10 | 13 | 2 | 3/3 | 12/23 | J8 11, C4 9, J7a 3, J1b 2, C10 2, J1c 1, J6b 1, C2 1 |
| SFT 50, seed 2 | 8 | 16 | 1 | 1/1 | 14/28 | J8 13, C4 5, C2 2, J1b 1, J7a 1, C10 1, C1 1 |
| SFT 85, seed 1 | 3 | 12 | 3 | 5/5 | 21/33 | J8 17, J7a 5, C4 4, C10 3, J1c 2, J1b 2, C2 1, J1a 1 |
| SFT 85, seed 2 | 6 | 12 | 6 | 9/12 | 16/31 | J8 14, C4 6, J7a 6, C10 5, J1b 1, J1c 1 |

**Tie-in generalisation set (7):**

| | Ready |
|---|---|
| Sol | 7 |
| untuned | 4 |
| SFT 25 | 0 / 3 (seed 1 / seed 2) |
| SFT 50 | 2 / 4 |
| SFT 85 | 5 / 2 |

The SFT failures are J8, invalid JSON (C1, 2–3 in the smaller runs) and citations (C4).

**What it shows:**

1. **SFT at this size made the model worse, at every size and seed.** Readiness fell from 14 to
   3–10. The J8 pattern appears in every adapter, already at 25 runs: 12–21 of each adapter's
   supplier questions carry the action's "to bridge the N-week gap", against 0 of 156 in the
   teacher's data. Wrong citations (C4) rose too, from 0 untuned to 4–9.
2. **Setting J8 aside, nothing improved either.** 12–16 ready against 16 untuned.
3. **Applying the lead-time allowance rose with data but not the arithmetic.** Changes applied
   went from 1 to 6 of 11 at 85 runs (seed 2), but 9 of the 12 changed items had a wrong figure.
   The policy the teacher shows is being learned; computing the figures after it is not, as
   expected of answers trained without reasoning (`docs/sft_base_screening.md`).
4. **Validation loss and task results disagree.** Loss fell steadily (0.45 → 0.33), while
   readiness did not rise. The model came closer to the teacher's wording, phrases included,
   without keeping the rules behind it, such as which field may name the gap.

**Reading of the cause (to be tested, not settled):**

- With 25–85 examples of long, structured answers, the model picks up frequent phrases across
  fields. "To bridge the N-week gap" is in 60 of 300 teacher actions, and supplier questions about
  expediting have the same shape.
- The evaluation set is weighted to Critical items (43%, against 18% in training), where this
  happens.
- Greedy decoding (temperature 0) makes a learned phrase come out every time.

**Options, for the user to decide:**

- **More data, matched to the evaluation's mix** (more Critical items). The validation loss had not
  levelled off.
- **Targeted examples**, where a Critical item's supplier question and its action appear together
  and differ only in what they may say.
- **DPO** on pairs from the SFT model's own outputs that differ in this phrase (on-policy pairs;
  `docs/sft_plan.md`, section 3.5).
- **Reasoning traces**, so that the arithmetic after a change is worked out rather than recalled
  (section 3 of the screening record).
- **Accept the result for now:** at this scale, the untuned model with the production prompt is
  better than the fine-tuned one, and the production analysis stays on `gpt-6.1-sol`.

## 7a. Why quality fell: the cause, tested (2026-10-26)

Three tests separate the causes: the expedite questions on Critical cases, an in-distribution
evaluation, and where the phrases come from.

**1. SFT learned the teacher's policy, and kept the base model's wording.** On the 18 evaluation
cases that have a Critical item:

| | Supplier questions | Of them, asking for expedited or partial delivery | Of those, naming the gap ("bridge … gap") |
|---|---|---|---|
| `gpt-6.1-sol` | 32 | 18 | **0** |
| Gemma 4 12B, untuned | 11 | 2 | 2 |
| SFT 85, seed 1 | 26 | **21** | 20 |
| SFT 85, seed 2 | 24 | **19** | 15 |

- **The teacher asks the supplier about expediting for Critical items.** It does so 18 times, and
  never names the gap.
- **The untuned model rarely asks** (2 times), and both times it names the gap.
- **SFT learned to ask as often as the teacher** (19–21 times), which is what the teacher's data
  shows. It did not learn the teacher's wording: the words come from the actions. "Alternative
  source" and "bridge … gap" are in 21 and 60 of the teacher's 300 actions, and in 0 of its 156
  supplier questions. They appear in 17 and 21 of the fine-tuned model's 33 supplier questions.

So the fall in readiness is mostly the model doing the right thing (asking the supplier for
expedited delivery of a Critical item) in the wrong words. A habit the untuned model showed only
twice is now exercised about twenty times. On the evaluation cases without a Critical item, the
fine-tuned model had no J8 failure (0 of 7).

**2. On data like its training data, SFT changed little.** The 13 validation scenarios were held
out from training, but are of the training distribution (same generator, Korean food and household
products, dated documents). The teacher passed all 13; they were chosen from passing runs.

| 13 in-distribution cases | Ready | Blocking failures by check |
|---|---|---|
| Gemma 4 12B, untuned | 3 | J1b 6, J8 5, C10 4, J7a 2, and one each of J1a, C6, J6a, J6b, J6c, C2, C4 |
| SFT 85, seed 1 | 2 | J8 5, J7a 4, C10 3, C5 2, J6b 2, and one each of J6a, J1b, C2, C4, C1 |
| SFT 85, seed 2 | 4 | J8 4, C5 3, C10 3, J1b 2, J7a 2, J6b 2, and one each of J6a, J6c, C2, C4, C1 |

- **One gain:** SFT cut distortion of the documents (J1b 6 → 1–2).
- **What it did not change:**
  - the internal figures in supplier questions (J8 5 → 4–5);
  - the arithmetic after a change (C10 4 → 3).
- **Overall:** readiness stayed about where it was (3 → 2–4). With 85 examples the model moved toward
  the teacher in some rules and not in others. It did not get the many rules right at once that
  "ready" requires.

**3. On the evaluation set the shift made it worse.** The evaluation cases differ from the
training data:

- industrial parts (MetalBolt, CircuitBoardA), not food and household goods;
- short, undated, generic history documents;
- 43% Critical items, against 18%.

The Critical share is where test 1's pattern happens, so the evaluation set turned it into the
main failure (17 of 18 Critical cases).

**The cause, in one paragraph.**

- SFT on 85 verified examples moved Gemma 4 12B toward the teacher's behaviour: it asked suppliers
  about expediting, and it distorted documents less.
- It did not reach the teacher's precision in the details that the checks enforce: which field may
  name the gap, and the arithmetic after a change.
- Where a learned behaviour met an untouched habit, readiness fell, and most on an evaluation set
  weighted to the cases where they meet.
- It is not overfitting: validation loss fell to the last epoch. Nor is it a broken setup: the
  in-distribution results match the untuned model.
- It is a capability and data-size gap: a 12B model, 85 examples, and long structured answers with
  many rules.

## 8. Decision: stop here, production stays on `gpt-6.1-sol`

**Decided by the user on 2026-10-26.** No DPO and no further SFT rounds. The analysis agent
stays on `gpt-6.1-sol`, and this failed attempt is kept as the record of why.

**The reasons, as the user put them:** the gap between the open model and the teacher is large,
and training made the results worse, not better.

**What the record shows:**

- **Untuned, the gap is wide.** Gemma 4 12B was ready in 14 of 25 cases against Sol's 24, and
  applied 1 of Sol's 11 lead-time allowances.
- **SFT did not close it at 25–85 examples.** Every adapter fell below the untuned model (3–10
  ready). The changes it learned to apply came with wrong figures (9 of 12 at 85 runs).
- **Closing it would need more than one more round:** more data in the evaluation's risk mix,
  targeted examples or DPO for the field mix-up, and reasoning traces for the arithmetic. Each is
  a project of its own.
- **At this point the API is the better choice for the role.** The analysis is a few calls a day,
  at $0.021–0.033 each (`docs/analysis_model_selection.md`). Its quality is the point of the
  pipeline.

**Where self-hosting stands:**

- the serving side is measured: vLLM, fp8 against bf16, `docs/sft_plan.md` section 6, and
  section 9 below;
- the model side is not yet good enough, by the same checks that chose every production model.

The finding worth keeping: **validation loss fell steadily while readiness fell.** Judged by loss
alone, the fine-tune would have looked like a success. The production checks caught the regression
and traced it to one phrase crossing fields.

## 9. Serving: Gemma 4 12B, bf16 against fp8 (2026-10-26)

Measured on the RTX 5090 (32 GB) with vLLM v0.31.0, untuned Gemma 4 12B
(`scripts/serve_quant_compare.ps1`):

- **Quality:** the 25 evaluation cases with the production checks, 6 cases at a time.
- **Throughput:** the production request itself (`scripts/bench_vllm_v5.py`). That is the v5 system
  prompt and the pipeline's JSON for the evaluation scenarios, about 4,400 prompt tokens, answers
  up to 1,500 tokens, temperature 0, at 1–16 concurrent requests.

These are not L4 figures. At bf16 the model does not fit an L4 at all (weights about 24 GB);
fp8 is what an L4 would serve.

**Memory:**

| | bf16 | fp8 |
|---|---|---|
| Weights and runtime | 23.9 GiB | **13.3 GiB** |

**Throughput (output tokens per second, all requests; latency per request):**

| Concurrent requests | bf16 tok/s | fp8 tok/s | bf16 p50 / p95 | fp8 p50 / p95 |
|---|---|---|---|---|
| 1 | 52.8 | **82.8** | 9.1 / 12.8 s | 5.8 / 7.6 s |
| 2 | 94.3 | 141.6 | 9.3 / 13.0 s | 5.9 / 7.7 s |
| 4 | 146.0 | 227.1 | 9.0 / 12.9 s | 6.1 / 7.9 s |
| 8 | 315.4 | 514.9 | 12.0 / 19.5 s | 7.9 / 12.0 s |
| 16 | 453.5 | **834.5** | 16.1 / 22.2 s | 8.6 / 14.2 s |

- **Continuous batching:** from 1 to 16 concurrent requests, throughput rose 8.6× at bf16
  (52.8 → 453.5 tok/s) and 10× at fp8 (82.8 → 834.5), while latency per request rose far less.
- **fp8 against bf16:** 1.6× the throughput at one request and 1.8× at sixteen, with lower
  latency, on 56% of the memory.

**Quality, production checks, 25 cases:**

| | Ready | Lead-time changes applied (of Sol's 11) | Blocking failures by check |
|---|---|---|---|
| bf16, this run | 15 | 1 | J1b 5, C10 3, J8 2, C5 1, J1c 1, C6 1, J6b 1 |
| bf16, round-2 screening run | 14 | 1 | J1b 5, J8 3, J1c 2, C10 2, J6b 2, C5 1, J1a 1, C6 1 |
| fp8 | 13 | 3 | J1b 6, J6b 5, J1a 2, C10 2, J8 1, J7a 1, J1c 1 |

**Run-to-run variation is part of the comparison.** The two bf16 runs, same model and settings,
disagree on 3 of the 25 cases (batched decoding is not exactly repeatable). bf16 and fp8 disagree
on 6, and fp8 is ready in 13 against 14–15.

So the quality cost of fp8 is small, about the size of the run-to-run variation, and not shown to
be zero. Its extra J6b failures come with the extra lead-time changes it made (3 against 1).
These are measured with the production checks, not a preference score.

**Case by case** (the 16 cases where any run failed; the other 9 were ready in all three):

| Case | bf16, run 1 | bf16, run 2 | fp8 |
|---|---|---|---|
| eval_01 | J8 | J8 | J1b, J7a |
| eval_02 | J1b | ready | ready |
| eval_04 | ready | ready | J1a |
| eval_06 | J8 | J8 | J8 |
| eval_09 | J1b, J1c | J1b | J1b |
| eval_10 | C10, J1b | C10, J1b | J1b |
| eval_11 | ready | ready | C10, J6b |
| eval_13 | C5 | C5, C10 | C10, J6b |
| eval_14 | J6b | ready | ready |
| eval_15 | ready | ready | J1b, J6b |
| eval_16 | ready | ready | J6b |
| eval_18 | J1a, J1b | J1b | J1a, J1b |
| eval_19 | J1c | J1c | J1c |
| H1 | ready | J1b | ready |
| H2 | C6, C10, J6b, J8 | C6, C10, J6b | J1b, J6b |
| H3 | J1b | J1b | ready |

**Every one of fp8's five J6b failures is the same omission.** It proposed a change of a size the
documents do not give, and did not say it was its estimate to be confirmed:

- lead time 22 weeks for an unresolved strike (eval_11, the same 22 weeks as Sol);
- 25 weeks (eval_13);
- 19 weeks (H2);
- a 50% quantity increase for an unsized surge (eval_15, eval_16).

bf16 mostly made no change in these cases, so it had nothing to fail. In other words, fp8 here
made the allowance the prompt asks for more often, but left out the "to be confirmed" wording; it
did not get its judgement wrong. The other differences (J1a, J1b, J7a) move between runs at bf16
as well.

## 10. Files

| What | Where |
|---|---|
| Scenarios and teacher data | `training_data/sft_v5/` |
| Scripts | `scripts/generate_sft_scenarios.py`, `scripts/build_sft_teacher_data.py`, `scripts/train_sft_v5.py`, `scripts/train_learning_curve.ps1`, `scripts/screen_base_models.py` (`--workers`, `--scenarios`), `scripts/serve_local_model.ps1`, `scripts/report_sft_results.py` |
| Evaluation outputs | `eval_results/sft_screen_sft-n<size>_s<seed>_{fac-rule,tie-in}.jsonl` |
| Adapters and training logs | `C:\sft\runs\gemma4-12b_n<size>_s<seed>\` (not in the repository: about 280 MB each) |

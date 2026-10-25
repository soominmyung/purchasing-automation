# SFT retraining plan (v5 analysis agent)

Plan agreed with the user on 2026-10-25, before any data is generated. It records what will be
trained, on what data, how the result will be measured, and the training settings, with the reason
for each choice. Decisions still open are marked **Open**. Results will be written up separately
when the work is done.

## 0. The earlier SFT (March 2026) and why it is redone

| | The March 2026 SFT |
|---|---|
| Model | Llama-3-8B (base), QLoRA 4-bit with Unsloth, on a Vertex AI T4 (`scripts/train_sft.py`) |
| Data | 30 synthetic scenarios (25 for training, 5 held out); teacher GPT-4o |
| Input | raw CSV rows; risk level taken from the source (no Critical); history as one free-text line each for the supplier and the items |
| Output | the old analysis JSON (`purchasing_report_markdown`, …) |
| Evaluation | the 5 held-out scenarios, against GPT-4o |

The v5 pipeline differs on both sides:

- **Input.** Code works out every figure, and the history comes as numbered documents.
- **Output.** The analysis is one call that returns structured fields in the order of the work:
  findings from the documents, the proposed changes, the figures after the changes
  (`figures_after_changes`), stock findings, questions, actions and the summary. Code checks the
  figures after the changes value by value (C10), and renders the report from the fields
  (changelog item 23).

The March model has never seen this input or this output, so its data cannot be reused. Its 25
training inputs could be converted, but 25 is far too few.

The March evaluation was also open to a fair criticism, one the user has heard: *it only measured how
well the small model imitated GPT-4o.* This plan is built to answer that (§3.3).

## 1. What is trained

**Decided:** the analysis agent only, in one LoRA adapter. The approval and email agents
are left out.

- The analysis is where the pipeline's judgement happens. The approval and email agents write short
  text, cheaply, with `gpt-6.1-sol` at low effort.
- One pipeline run gives one training example: the analysis prompt and input (the figures and the
  numbered history documents), and the analysis JSON.
- Rewrite calls (the analysis sent back with check failures) are not trained; see §2.2.

## 2. Training data

### 2.1 Inputs: new scenarios

New supplier scenarios are generated: a stock CSV plus history documents, in the format the pipeline
reads.

**How many: decided by a learning curve, not fixed in advance.** The user's first reaction to 300
suppliers was that it is too many, and the plan agreed with that. Cost is not the reason (300 runs
would cost about $15–20). The reason is that 300 is a guess. The plan instead:

1. Generate about **120 supplier scenarios**, about 12 for each of about 10 types.
2. Keep those whose teacher output passes (§2.2). We expect about 100, one training example
   each.
3. Train on **25, 50 and 100 suppliers** (nested subsets, each type in the same proportion). Measure
   each model on the evaluation set (§3).
4. Stop where the curve flattens. If it is still rising at 100, generate more.

The learning curve is a planned figure for the README and the portfolio. It turns "how much data?"
from a guess into a measured answer, for example: "from 50 to 100 suppliers, first-answer blocking
failures fell only from X to Y". **Each point is trained twice, with two seeds (decided by the user, 2026-10-25).** With 25–100
examples, the seed (the order the examples are shown in, the LoRA initialisation, dropout) can move
the result as much as the data does. With one run per point, a flat or rising curve could be luck.
Each point is drawn as the mean of the two runs, with their spread as an error bar. A gain between
two sizes counts only if it is larger than the spread between seeds. The seeds and every run's
results are recorded. Variety matters more than
count, so the types are balanced at every size.

**Types** (aligned with the evaluation set's categories, plus the v5 cases found in development):

- **History content:** clean history; no history; a supplier delay notice or lead-time change; a
  quality incident, about the supplier or about an item; a demand surge, in the UK, in Korea or in a
  market not stated; ambiguous attribution.
- **Stock position:** Critical items; every item Low; an item that goes from Low to Medium at a
  longer lead time; high stock but short cover.
- **Data:** missing or invalid values.

**Kept apart from training:**

- the 25 cases used for every model comparison (`eval_data/eval_scenarios.jsonl`, 20, and
  `eval_data/holdout_scenarios.jsonl`, 5);
- the generalisation set (§2.3).

Supplier and item names will not overlap with either. A similarity check after generation removes
near-copies.

### 2.2 Targets: verified teacher outputs

- **Teacher:** `gpt-6.1-sol` at medium effort, run through the real pipeline: the same code, prompts
  (v5) and figures as production.
- **Kept:**
  - only runs whose **first answer passes every blocking check**: code checks C1–C10, and the judge
    (`claude-haiku-5-5`, high effort, three runs, majority vote);
  - each kept run gives its input and output exactly as the pipeline sent and received them.
- **Not kept: outputs that passed only after a rewrite.** A rewrite answers a different input: the
  first output plus the failures. Pairing the corrected output with the original input would credit
  the model with an answer it reached only with the checks' help, so only first answers that passed
  are used.
- **Failed outputs are stored, not discarded.** They can be the rejected side of DPO pairs later
  (§3.5).
- **Recorded:** the teacher's raw first-answer pass rate before filtering. The student is compared
  with it (§3.3).
- **Cost and time (estimate):**
  - Sol analysis costs $0.021–0.033 per run (measured, `docs/analysis_model_selection.md`), about
    $0.05 with the judge;
  - 120 runs come to about $6;
  - a run takes about 35 s, so the whole set takes well under an hour in parallel.
- **Human in the loop:** the user reads about 20 teacher outputs before training, as in every model
  comparison so far.

### 2.3 Generalisation set: a type never trained on

**Purpose.** To show that the student learned the task's general rules, not the teacher's patterns. It
is tested on a kind of history it has never seen in training.

**Proposed by the user (2026-10-25): a tie-in sale.** A supplier, when asked for a product, pushes a
newly launched product with it (named, for example, "AquaPlus Mini"). It tends to give supply to other
buyers when the bundle is refused. This is a good choice for these reasons:

- **The right severity.** It matters for the order but is not a Critical-type emergency. Critical
  cases, stock-outs and quality incidents stay in training, where the pipeline most needs them right.
- **Truly new.** It appears in no training type, no evaluation case, and no rule or example in the
  v5 prompt. The only way to handle it well is through the prompt's general rules:
  - findings faithful to the document, quoted without strengthening;
  - questions and actions that follow from the findings they cite;
  - actions timed against the latest order date;
  - no quantity change without a basis in UK demand.
- **Checkable.** The same checks and the judge apply unchanged.

**Design.**

- About **6–8 scenarios**, written once and kept out of training, with varied set-ups:
  - the tie-in on a Critical item or on a Medium item;
  - stated in a supplier email, or reported by a colleague;
  - with and without a supply threat.
- **Measured** with the same checks as §3, and read side by side. Sol is run on the same scenarios as
  the reference.
- **No expected behaviour is written into the prompt or the checks for this type.** What counts as a
  good analysis is judged by the existing rules and by reading. Setting a rule for it would decide a
  purchasing question the user has not decided.

**Decided (the user, 2026-10-25):** no second held-out type; the tie-in sale is the only one.

## 3. Measurement

### 3.1 Set-up

- **Same inputs, checks and table** as the analysis model comparison
  (`docs/analysis_model_selection.md`), so that the results sit in one table with the earlier ones.
- **Inputs:**
  - the 25 evaluation cases;
  - the generalisation set (§2.3).
- **How the model is swapped in:** only the analysis model is replaced in the pipeline. vLLM serves
  an OpenAI-compatible API, so the pipeline code barely changes.
- **Compared:**

  | Model | Role in the comparison |
  |---|---|
  | the chosen base model, untuned, same prompt | the floor; the gain from training is SFT minus this |
  | the SFT model, at each learning-curve size | the subject |
  | `gpt-6.1-sol`, medium | the production model and teacher; the ceiling |

- **GPT-4o is not included** (decided by the user, 2026-10-25). It was already measured on the v5
  prompt when the analysis model was chosen: 0 of 4 first answers passed
  (`docs/analysis_model_selection.md`). The question here is different: *can a self-hosted small model
  take Sol's place in this role?*
- **The March adapter is not included either.** It cannot read v5 input, so comparing it would only
  show the format change.

### 3.2 Measures

**Quality**

- first-answer blocking failures;
- rewrites needed;
- figure errors;
- JSON that fails to parse.

**What each analysis raised**

- the questions and actions, judged on how specific they are, the criterion that chose Sol;
- whether the issue planted in each scenario was raised (§3.3, point 3).

**Running it**

- time per analysis;
- cost per analysis: Sol's API price against L4 GPU time;
- throughput.

**Reading:** outputs side by side, with issues marked. The decision leads with the performance found
by reading, with real sentences; cost and speed come after.

**Serving:** fp16 against fp8, on the same cases, for quality and throughput. This continues
`docs/inference_serving_vllm.md`.

### 3.3 Not "how well does it imitate the teacher"

The measures are chosen so that the answer does not rest on resemblance to the teacher:

1. **No similarity metric.** No BLEU, no ROUGE, no agreement-with-teacher score.
2. **The analysis checks and the judge measure the task, not the teacher.**
   - **The rubric** (`docs/eval_rubric.md`) defines what a correct analysis is. Code checks C1–C10
     test figures, items, citations and change arithmetic against the input. The judge checks
     whether:
     - findings are faithful to their documents;
     - questions, changes and actions follow from the findings they cite;
     - a change's size fits its basis.
   - **The judge was itself validated** before use (`docs/judge_selection.md`): it caught 93.9% of 51
     seeded faults, and raised false alarms on 8.0% of 89 verdicts the user scored by hand. Its
     verdicts are measured, not taken on trust.
   - **The judge is of a different family from the teacher** (Claude against GPT). It has no
     preference for the teacher's style.
3. **Planted issues as ground truth.** The evaluation scenarios carry labels set when they were
   written (`supplier_issue`, `item_issue`, `data_issue`). Whether an analysis raised the planted
   incident is decided by the scenario's design, not by what the teacher wrote.
4. **Verified targets, compared with the teacher's raw rate.** The student learns only outputs that
   passed the checks (§2.2). If its first-answer pass rate is above the teacher's unfiltered rate, it
   has learned the standard, not copied the teacher, mistakes included.
5. **Generalisation to an unseen type** (§2.3). Following the rules on a kind of history never trained
   on is not imitation.
6. **The purpose is a deployment requirement, not "as good as GPT".** Served from the company's own GCP
   project (§6), the analysis, the call that receives the most data (every item's figures and every
   history document), goes to no model company; its cost is fixed by the GPU's hours, and it no longer
   depends on an outside model's availability or changes. The other agents still use APIs (§6.3). So
   the question is whether the small model is good enough for this role and where it falls short. Its
   shortfalls are reported with real examples.

### 3.4 What gets reported

- **The learning curve,** for the README and the portfolio:
  - first-answer blocking failures against training suppliers (25 / 50 / 100);
  - planted-issue detection against training suppliers;
  - the untuned base model and Sol drawn as reference lines.
- **One table:** every model and size × every case.
- **The generalisation set,** read side by side with Sol.
- **Serving:** fp16 against fp8.

### 3.5 Later, optional: DPO

The stored pairs (§2.2) give a DPO step: the same input with a passing output and a failing one. That
step learns from the checks, not from the teacher. `scripts/generate_dpo_pairs.py` must be checked
before it is reused.

## 4. Training settings

| Setting | March 2026 | Planned | Why |
|---|---|---|---|
| Base model | Llama-3-8B, base | **a current instruct model** (decided); which one is **Open**, §5 | chat format, following a system prompt and JSON output come with instruct tuning |
| Prompt format | its own `### Instruction` template | **the pipeline's exact system and user messages, in the model's chat template** | training and serving see the same input, so the model drops into the pipeline unchanged |
| Loss | on the whole text, prompt included, with packing | **on the answer only** (completion-only) | otherwise much of the training goes into reproducing the long prompt |
| Sequence length | 4,096 | **6,144–8,192**, set from the longest example | the analysis call took 3.9–4.8k input tokens in recent runs (prompt and input), plus about 1–2k output tokens; no example may be cut |
| LoRA | r 16, α 32, lr 2e-4, 5 epochs | r 16, α 32, lr 2e-4, **2–3 epochs** | 5 epochs on about 100 examples risks overfitting |
| Checkpoint choice | none | **10% of the generated data held out as validation**; the checkpoint is chosen by validation loss | the 25 evaluation cases are used only for the final measurement, never to choose a checkpoint |
| GPU, training | Vertex AI T4 (16 GB, no bf16) | **the user's PC, RTX 5090 (32 GB, bf16, fp8), under WSL2 or Docker**; the final configuration run once more as a **Vertex AI custom training job** with the same container | the 5090 has more memory and several times the speed of an L4, at no hourly cost, which the learning curve's many runs need; the Vertex run shows the same job reproduced in the cloud (decided by the user, 2026-10-25) |
| GPU, serving and measurement | — | **an L4 (24 GB) on GCP**, the production serving VM (§6) | measured on the GPU it is deployed on, not on a development PC; continues `docs/inference_serving_vllm.md` |
| Output format at serving | not enforced | **measured with and without vLLM guided JSON (the output schema)** | separates what decoding guarantees (the format) from what training must supply (the content) |
| Thinking mode | — | **off**, if the model has one | the teacher's targets are final answers; Sol's reasoning is not available to train on |
| Tracking | W&B | W&B, plus the settings and data version recorded in the results document | |

## 5. Base-model candidates

**Criteria:**

- QLoRA training and fp16 or fp8 serving on one L4 (24 GB);
- a context of at least 8k;
- a licence that allows company use (Apache 2.0);
- supported by Unsloth and vLLM.

Checked on 2026-10-25. Most sources are secondary, so each licence and release date will be confirmed
on the official model card before a model is chosen.

| Candidate | Released | Size | Licence | Notes |
|---|---|---|---|---|
| **Gemma 4 12B** | 2026-06 | 12B dense | Apache 2.0 | the newest small dense model found |
| **Qwen3.5-9B** | 2026-03 | about 9–9.7B dense | Apache 2.0 (to confirm) | widely fine-tuned with Unsloth; has a thinking mode, to be turned off |
| **Gemma 4 E4B** | 2026-04 | about 4.5B effective (8B with embeddings) | Apache 2.0 | the smallest; tests how small the model can be |
| **Ministral 3 8B Instruct** | 2025-12 | 8B | Apache 2.0 | official vLLM recipe |
| **Ministral 3 14B Instruct** | 2025-12 | 14B | Apache 2.0 | about 14 GB in fp8, so it fits an L4; the upper end for one L4 |

**Looked at and excluded:**

| Model | Why excluded |
|---|---|
| Mistral Small 4 (2026-03) | a mixture of experts with 119B parameters (6.5B active per token). All weights must be in memory, about 60 GB even at 4-bit, so it fits no single L4 or T4 |
| Mistral Large 4 (2026-10, about 1T) | far too large. There is no "Ministral 4" |
| Qwen3.6-27B, Qwen3.8-27B | 27B dense: about 27 GB in fp8, too large to serve on an L4. Qwen 3.6 and 3.8 have no smaller sizes; the small Qwen models are 3.5 |
| Gemma 4 26B MoE, Gemma 4 31B | too large for an L4 |
| Llama | no new small model after Llama 3.1 8B (2024) |

**On a T4 instead of an L4.** The question was asked about Ministral 3 14B; the same holds for Gemma 4
12B.

- **Training:** possible but tight. The 4-bit weights take about 8–9 GB of the T4's 16 GB, which
  leaves little for activations at a 6–8k sequence even with gradient checkpointing. A T4 has no bf16,
  so training is in fp16, which is less stable. It is also several times slower than an L4.
- **Serving:** fp16 (about 28 GB) does not fit. A T4 has no fp8 support, so the model could be served
  only at 4-bit (AWQ or GPTQ, about 9 GB), with a small KV cache.
- **So** a T4 would rule out the fp16-against-fp8 comparison. The plan keeps the L4.


**How the model is chosen (proposed, Open):**

1. Run every candidate untuned, with the same prompt, on part of the evaluation set.
2. Read the outputs side by side.
3. Train one or two of them.

No candidate is dropped on impression. The screening runs on the RTX 5090.

## 6. Serving

**Decided (the user, 2026-10-25):** the fine-tuned analysis model is served on **an L4 VM in the
company's GCP project**, with vLLM. The pipeline itself stays on the on-prem SAP server. The company
server has no GPU. This is the set-up actually built and measured (§6.4), and it is the same recipe as
`docs/inference_serving_vllm.md`.

The SAP server is not used for serving because it has no GPU.

### 6.1 Where it runs

| | |
|---|---|
| Pipeline | the on-prem SAP Business One server, as today: FastAPI, the history sync, the documents |
| Model | a GCP Compute Engine VM, `g2-standard-8`, with one NVIDIA L4 (24 GB), in the company's own GCP project, in a region near the UK office where L4s are offered (to be checked) |
| Engine | the official `vllm/vllm-openai` Docker container: the chosen base model with the LoRA adapter (`--enable-lora`), at fp8 if the measurement shows no loss of quality (§3.2), otherwise fp16 |

### 6.2 How the pipeline reaches it

- **The network.** The VM has no open ports except vLLM's. The GCP firewall allows only the company's
  fixed IP address, and vLLM requires an API key (`--api-key`). The call goes over HTTPS, with a TLS
  proxy in front of vLLM on the VM.
- **The call.** vLLM serves an OpenAI-compatible API. To the pipeline, the analysis model is an OpenAI
  model at another address (`https://<vm address>/v1`, model name `purchasing-analysis`). The only code
  change is in `services/agents._llm`: a model name such as `local:purchasing-analysis` creates
  `ChatOpenAI` with that `base_url` and key, both set in `config.py`. The prompt, checks, judge and
  documents do not change.
- **The output format.** When the pipeline asks for the analysis's JSON schema, vLLM's guided decoding
  can enforce it. Whether to switch it on is decided by the measurement (§4).
- **The other agents.** The approval agent, the email agent, the validator and the judge stay on their
  APIs. Only the analysis is trained (§1).

### 6.3 Where the data goes

What each step of a run sends, and to whom, with the analysis served from GCP:

| Step | What it sends | Where it goes |
|---|---|---|
| History ingest (when a document is added) | the document's text, for its embedding | OpenAI embeddings API (Chroma embeds at ingest; retrieval does not use it) |
| **Analysis, both calls** | every item's figures and the supplier's history documents in full | **the company's GCP VM** (with `gpt-6.1-sol` today: OpenAI) |
| Judge (evaluation) | the same input as the analysis, plus the analysis | Anthropic (`claude-haiku-5-5`) |
| Purchase request (approval agent) | the requested items' figures, and the findings, changes and actions about them | OpenAI (`gpt-6.1-sol`) |
| Supplier email (draft agent) | item names, quantities, delivery dates, the questions for the supplier | OpenAI (`gpt-6.1-sol`) |
| Email validator | the email's text and the names of the items left out | Anthropic (`claude-haiku-5-5`) |
| Tracing | structure, timings, token counts; prompt and response text hidden where real data runs | LangSmith (`docs/observability.md`) |

So serving the analysis from GCP takes the analysis call off OpenAI, but not the whole run: the judge
still receives the full input, and the purchase request and email agents receive parts of it. Keeping
all of the company's data off model companies' APIs would need those agents self-hosted too, or the
judge left out in production. Neither is planned.

### 6.4 Measurement

The serving measurement (§3.2) runs on this VM type with the production flags:

- the same vLLM version;
- the same `--max-model-len` and `--gpu-memory-utilization`;
- fp16 against fp8;
- the adapter served the same way.

The figures measured are therefore the production figures.

### 6.5 Running it

- **Cost.** The buyers run the pipeline a few times a day. The VM can be started on schedule for
  working hours and stopped outside them, or left running; the measurement gives the cost per analysis
  either way.
- **Health.** The pipeline checks vLLM's `/health` before an analysis.
- **If the model is down.** What the pipeline should do then is **Open**, for the user to decide: stop
  with an error, or fall back to `gpt-6.1-sol`, which sends that run's data to the API.
- **Adapter versions.** Each trained adapter is a folder in Cloud Storage, kept with its version and its
  evaluation results. An update points vLLM at the new one and restarts the container; a rollback
  points it back at the previous one.

## 7. Order of work

1. Confirm the candidates' model cards. Run the untuned screening; the user chooses the base model(s).
2. Write the scenario generator and the tie-in generalisation scenarios. The user reviews a sample of
   the generated inputs.
3. Run the teacher through the pipeline, filter, and record the raw pass rate. The user reads about 20
   outputs.
4. Train at 25, 50 and 100 suppliers on the RTX 5090, and choose checkpoints by validation loss. Run the
   final configuration once as a Vertex AI custom training job with the same container.
5. Launch the GCP L4 VM with the three access layers of §6.2 set up by the user (firewall rule for the
   company IP, vLLM API key, HTTPS through Caddy). Serve at fp16 and fp8, with the production flags (§6.4); measure on the 25 cases and the generalisation set, and read side
   by side.
6. Write up the results, with the learning curve. Set up the production serving VM (§6), then update
   the README and both portfolios.

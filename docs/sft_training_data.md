# SFT training data (v5): what it is and how it was made

Training data for fine-tuning Gemma 4 12B as the analysis agent (`docs/sft_plan.md`). This page
records what the data is, how it was made and checked, and what was kept out of it. Counts
marked *(to fill)* are completed when the teacher run finishes.

## In one line

117 synthetic supplier scenarios in the pipeline's own input format. `gpt-6.1-sol` analysed each
one through the production pipeline, and only answers that passed every blocking check (code checks
C1–C10 and the validated judge) were kept. Each training example is the exact prompt the pipeline
sends and the exact answer it received.

## 1. The scenarios (the inputs)

**Why synthetic.** The company's stock and supplier records stayed with the company, and synthetic
scenarios let the mix of situations be set on purpose, each with a known planted issue.

**What one scenario is.** One supplier with 2–6 items (stock figures) and 0–11 history documents.
Each document has a header saying what it is about, as purchasing documents do in practice.

| | Count |
|---|---|
| Scenarios | 117: 13 types × 9 |
| Items | 2 per supplier: 39 · 3: 31 · 4: 23 · 5: 10 · 6: 14 |
| History documents | 598 in all: 226 about the supplier as a whole, 303 about some of its items, 69 about items whoever supplies them; 9 scenarios have none |
| Risk levels (at 18 weeks) | Low 80 · Medium 74 · High 67 · Critical 56 · not computable 3 (in the first fill; the full set has the same mix by design) |

**The 13 types.**

- **History:**
  - clean history; no history;
  - a delay with no size ("significant dispatch delays … unresolved");
  - a lead-time notice with a size (+2 to +6 weeks);
  - an item quality incident; a supplier-wide quality problem;
  - a demand change in the UK; a demand surge in Korea; a demand surge in an unstated market;
  - ambiguous attribution ("there have been reports of …").
- **Stock:** at least one Critical item with a minor or resolved issue; every item Low (half with a
  lead-time notice that moves one item to Medium).
- **Data:** missing or invalid stock values.

**How they were made** (`scripts/generate_sft_scenarios.py`):

1. **Names and history documents.** `gpt-6.1-sol` (low effort) wrote the supplier's name, its product
   names and its history documents, from a fixed type instruction. Each scenario was given a
   product line and a first letter, so the names vary.
2. **Stock figures, set in code by type.** Weeks to stockout come from fixed bands (Critical
   2–17, High 18–25, Medium 26–34, Low 35–70), and available stock is consistent with them. The
   first attempt let the generator choose the figures, and 250 of 274 items came out Critical, so
   that attempt was discarded (kept outside the repository).
3. **Risk levels from the pipeline's own rule** (`risk_level_from_weeks`); item codes assigned in
   code.
4. **Checks.** Supplier names are unique. No supplier or product name is shared with the 25
   evaluation cases. History documents may not restate stock, cover or risk: 0 of 363 did in the
   first attempt.

## 2. The answers (the targets)

`scripts/build_sft_teacher_data.py`:

- **Teacher:** `gpt-6.1-sol` at medium effort, the production analysis agent, with the production
  prompt (v5, with the 2026-10-25 `figures_after_changes` rule) and the production code path.
- **Scored by the production checks:** code checks C1–C10 and the judge (`claude-haiku-5-5`, high
  effort, one call as in production; the plan's three-run majority was not used, to save time).
- **Kept for training:** first answers that passed every blocking check.
  - Not kept: answers that passed only after a rewrite. A rewrite answers a different input (the
    first output plus its failures).
  - Failed answers are kept apart for DPO pairs.
- **Result: 98 of 116 runs passed** (84%) and are the training examples.
  - One scenario (`sft_missing_values_05`) had no usable row (every value invalid) and was dropped.
  - The 18 failures, kept for DPO, by check: J1b 5, J6b 5, J6a 3, J1a 3, J8 2, C4 1, J1c 1, J7a 1.
    In four lead-time-notice scenarios, the generated documents gave the supplier's own production
    lead time ("from eight to eleven weeks"), and the teacher read it as the whole import lead time
    and cut 18 weeks to 13–14. The judge caught all four (J6a/J6b); the scenarios were ambiguous.
  - Passing by type: lead_time_notice 5/9 and demand_korea 5/9 are the lowest; every other type
    passed 7–9 of 9.
  - On the 7 tie-in generalisation scenarios, the teacher passed 7 of 7.

## 3. One training example

Scenario `sft_delay_unsized_05`, Kkotmaru Foods: three items; a strike at the supplier's packing
facility, then "its dispatch backlog remained unresolved", with no number of weeks.

| Part | Content | Size |
|---|---|---|
| System | the production analysis prompt (`prompts/v5/analysis.md`) | 14,393 characters |
| User | the pipeline's JSON: snapshot date, the supplier, each item's figures as code worked them out (risk level, gap, latest order and delivery dates, suggested quantity, notes), and the numbered history documents | 2,779 characters |
| Assistant | the teacher's JSON answer: findings citing their documents; a supplier-wide lead time of 22 weeks with its reason ("continuing delays from the unresolved dispatch backlog … an estimate to be confirmed"); the figures after the change; questions; actions; the summary | 7,448 characters |

The teacher used 4,346 input tokens and 1,943 output tokens on this call.

In training, the three messages go through Gemma 4's chat template with thinking off, and the loss
is on the assistant's answer only (`scripts/train_sft_v5.py`). This example is 4,801 prompt tokens
and about 2,159 answer tokens; the longest example in the first runs was 7,062 tokens, so the
12,288 limit cuts none.

As the model sees it (contents shortened):

```
<bos><|turn>system
You are a senior procurement analyst ... (the whole prompt)<turn|>
<|turn>user
{"snapshot_date": "2026-03-02", "supplier": "Kkotmaru Foods", "items": [...], "history": [...]}<turn|>
<|turn>model
<|channel>thought
<channel|>                      <- up to here: the prompt, no loss
{ "findings": [...], "supplier_lead_time": {...}, ..., "summary": "..." }<turn|>   <- the loss
```

**Training format = serving format.** With thinking off, Gemma 4's template puts an empty thought
channel before the answer when it prompts for one (and vLLM serves it that way). Rendering the
whole conversation leaves it out. So the training text is built as the generation prompt plus the
answer, which makes the model start its answer exactly where it will start it in serving. This
was found by rendering an example before training.

## 4. Splits, and what is kept out

| Set | Contents | Used for |
|---|---|---|
| Training | the passing runs, in a fixed, type-balanced order | SFT; the learning curve takes the first 25, the first 50, then all |
| Validation | a tenth of the passing runs, type-balanced | validation loss, to choose the checkpoint |
| **Evaluation** | the 25 cases every model was compared on (`eval_data/`) | the final measurement only; never in training or checkpoint choice |
| **Generalisation** | 7 tie-in-sale scenarios (`generalisation_tie_in.jsonl`): a supplier pushing a newly launched product with the order | a type no training scenario has, to test the rules rather than the patterns |

## 5. Questions this answers

- **"Isn't it only imitating the teacher?"**
  - The targets are verified answers, not raw teacher output.
  - The measure is the task's own checks and the scenarios' planted issues, not similarity to the
    teacher.
  - The generalisation set is a type the model never saw (`docs/sft_plan.md`, section 3.3).
- **"Why so few examples?"** The size is set by the learning curve (25 / 50 / all, two seeds each):
  add data only while the curve still rises.
- **"How do you know the data is right?"**
  - Every target passed the same blocking checks as production, with a judge validated against
    seeded faults (93.9% caught) and hand scores (8.0% false alarms).
  - Generated scenarios were read in a sample (three types so far). Reading a sample of teacher
    answers is *(to fill)*.
- **"Leakage?"** The evaluation and generalisation sets share no supplier or product name with the
  training data. They are generated and stored apart, and are never used to choose a checkpoint.
- **"Why not real data?"** Company data stayed with the company. Synthetic scenarios also give every
  case a known planted issue to measure against.

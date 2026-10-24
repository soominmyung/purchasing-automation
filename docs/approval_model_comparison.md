# Approval agent model comparison

**2026-10-24.** Which model should write the purchase request's Purpose and Justification
(`AGENT_MODELS["approval"]` in `services/agents.py`). GPT-4o had held the role since the start of the
project, never compared.

## What the agent does, and what it needs

It writes two paragraphs for the approvers from facts already settled by the analysis and the
pipeline (items with their figures after any change, findings with their documents, changes,
actions; `documents.approval_input`). It needs no deep reasoning; it needs to keep to the rules
(two to four sentences, citations only to documents given), to add nothing the input does not give,
and to carry figures and wording over unchanged.

## Method

- The approval input saved with each of the 25 v5 base outputs (`eval_results/v5_base_outputs.jsonl`),
  so the analysis is not run again and every model gets exactly what the pipeline gave GPT-4o
  (`scripts/compare_approval_models.py`). Same prompt (`prompts/v5/approval_request.md`). One run each.
- No style examples: the local store's `request_examples` collection is empty (`docs/Examples.zip`
  has two examples, never ingested here), so every model wrote without the "Reference (tone only)"
  the pipeline adds when examples exist. The same holds for the base outputs and all development runs.
- Candidates: `gpt-4o` (current), `gpt-6-luna` (cheapest), `gpt-5.4-mini` (already the email agent,
  where it kept to the rules better than the others), `gpt-6.1-sol` (quality reference).
  `claude-haiku-5-5` and `gemini-3.1-flash-lite` were left out as judge candidates: the judge checks
  this text (P1a-c), and would be judging its own writing.
- Measured now: P2 (sentences), P3 (citations), phrases that add an assessment, time, cost. The
  judge's checks P1a-c wait for the judge's validation (`docs/judge_validation.md`).

## Results

| Model | P2 (1-2 / 2-4 sentences) | Justification sentences | P3 citations | Adds an assessment of the supplier | Words | Time (mean, max) | Cost per request |
|---|---|---|---|---|---|---|---|
| `gpt-4o` | 13/25 | 2-6, mean 4.4 | 25/25 | **8** | 105 | 2.8 s, 4.3 s | $0.0047 |
| `gpt-6-luna` | 23/25 | 2-5, mean 3.7 | 25/25 | 0 | 126 | 10.9 s, 20.8 s | $0.0006 |
| `gpt-5.4-mini` | 24/25 | 2-5, mean 3.6 | 25/25 | 0 | 117 | 1.3 s, 1.7 s | $0.0018 |
| `gpt-6.1-sol` | 25/25 | 2-4, mean 3.7 | 25/25 | 0 | 130 | 11.6 s, 17.1 s | $0.0064 |

"Adds an assessment": phrases such as "proven track record", "supporting the decision to proceed with
this supplier", "supports the reliability of this procurement", which no finding gives (the findings
say only that the supplier delivered consistently).

The texts side by side, with each scenario's input: `eval_results/approval_model_comparison.md`.
First reading (eval_01): `gpt-5.4-mini` writes the system's own terms into the approvers' text
("the dates shown in the pipeline", "the request rests on the pipeline figures already provided");
`gpt-6-luna` carries field values over awkwardly ("an expected stockout of 0"); `gpt-6.1-sol` carries
the facts over cleanly; `gpt-4o` adds "supporting the reliability of this procurement".

The sentence count is a matter of the prompt or of a fixed frame for the output, and weighs little
in the choice (the user, 2026-10-24).

## Decision

Pending: the outputs are read side by side first, then P1a-c once the judge is validated.

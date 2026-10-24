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

## Round 2: set efforts, and the judge's checks (2026-10-24)

The first run left each model's reasoning effort at the API's default and did not record it. What the
models accept, checked against the API on 2026-10-24:

| Model | Efforts | Default | Temperature |
|---|---|---|---|
| `gpt-4o` | none (not a reasoning model) | — | 0 |
| `gpt-6-luna` | none, low, medium, high, xhigh | medium | 0 only at effort none; otherwise only 1 |
| `gpt-5.4-mini` | none, low, medium, high, xhigh | none | 0 only at effort none; otherwise only 1 |
| `gpt-6.1-sol` | low, medium, high, xhigh | medium | only 1 |

Effort none is no reasoning: no reasoning tokens. `services/agents.py` (`_llm`) now passes
temperature 0 with effort none; before, these models ran at temperature 1 even at none. The approval
agent needs little reasoning, so each model was also run at its least efforts: none (temperature 0)
and low, low for Sol. Same 25 inputs and prompt (`scripts/compare_approval_models.py`).

P1a-c by the chosen judge (`anthropic:claude-haiku-5-5?effort=high&format=schema`,
`docs/judge_selection.md`), three runs, majority verdict (`scripts/judge_approval_models.py`;
`eval_results/approval_model_judged.jsonl`, $0.63). "Reading": outputs with an issue found by reading
every text, before the judge's verdicts (`eval_results/approval_model_issues_effort.json`; for the
first run, `eval_results/approval_model_comparison.md`). Texts, verdicts and notes side by side:
`eval_results/approval_model_judged.md`.

| Setting | P1 fail (judge) | Reading | P2 | Reasoning tokens | Time | Cost per request |
|---|---:|---:|---:|---:|---:|---:|
| `gpt-4o`, temperature 0 | 19 (P1a 16, P1b 11) | 22 | 13 | — | 2.8 s | $0.0047 |
| `gpt-6-luna`, medium (default) | 3 | 5 | 23 | — | 10.9 s | $0.0006 |
| `gpt-6-luna`, none, temperature 0 | 5 | 4 | 23 | 0 | 3.2 s | $0.0002 |
| `gpt-6-luna`, low | 5 | 9 | 24 | 128 | 3.9 s | $0.0003 |
| `gpt-5.4-mini`, none (default), temperature 1 | 11 | 15 | 24 | — | 1.3 s | $0.0018 |
| `gpt-5.4-mini`, none, temperature 0 | 6 | 18 | 24 | 0 | 2.0 s | $0.0018 |
| `gpt-5.4-mini`, low | 15 | 19 | 25 | 80 | 2.8 s | $0.0023 |
| `gpt-6.1-sol`, medium (default) | **1** | **2** | 25 | — | 11.6 s | $0.0064 |
| `gpt-6.1-sol`, low | 2 | 7 | 25 | 19 | 5.5 s | $0.0049 |

Counts are of 25 outputs. What the reading found, by setting:
- `gpt-5.4-mini` at every effort: this tool's terms ("the pipeline"), false statements ("the revised
  order dates and quantities" where nothing was revised), citations to documents that support nothing
  in the sentence or do not exist ([3], [4] with two documents), empty placeholders ("[ ]"), main
  actions left out. More effort made it worse.
- `gpt-6-luna`: none was as good as medium; low was worse (placeholders such as "[no document]",
  "expected stockout in 0 weeks" for a gap of 0). One output at none broke off as invalid JSON.
- `gpt-6.1-sol` low halved the time but moved checks to "before ordering" where the actions order now
  and check after (four outputs), which would hold back an overdue order.

The judge and the reading rank the settings alike. The judge does not see citations, placeholders or
missing actions, which is where most of `gpt-5.4-mini`'s faults are.

## Round 3: the prompt revised (2026-10-24)

From round 2, the recurring faults that came from the prompt: the prompt itself said "the pipeline's
figures" and "The pipeline writes the rest of the document", which the models repeated; the input's
field names were written into the text ("expected_stockout_weeks of 0"); and findings from the stock
figures, which come with an empty `documents` list, got empty or made-up brackets. Revised in
`prompts/v5/approval_request.md`, as the user set it out:
- The writer is given a role: a purchasing officer putting the request to the approvers for their
  decision, objectively and neutrally, not arguing for the purchase or judging the supplier, the
  items or the figures (the user: "buyer" could read as someone persuading for the purchase).
- "the pipeline" taken out of the prompt.
- The input's figures in plain words, never by their field names, and no mention of the tool that
  prepared them or of the input itself. Limited to this tool's terms: banning words such as "system"
  outright would break on items or documents that use them.
- A finding whose documents are empty, and the items' own figures, take no brackets; where needed the
  sentence says they come from the current stock position. The input is unchanged.

Run on the three settings still in the running, the same 25 inputs
(`eval_results/approval_model_comparison_revised_v1.jsonl`; judged:
`eval_results/approval_model_judged_revised_v1.jsonl`, $0.20; reading:
`eval_results/approval_model_issues_revised_v1.json`; side by side:
`eval_results/approval_model_judged_revised_v1.md`):

| Setting | P1 fail (judge), before → after | Reading, before → after | Time | Cost per request |
|---|---:|---:|---:|---:|
| `gpt-6-luna`, none, temperature 0 | 5 → 3 | 4 → 8 | 3.3 s | $0.0002 |
| `gpt-6.1-sol`, medium | 1 → 2 | 2 → **1** | 11.1 s | $0.0075 |
| `gpt-6.1-sol`, low | 2 → 3 | 7 → 4 | 4.8 s | $0.0055 |

- "the pipeline", field names and empty brackets are gone from all three, but for one placeholder
  ("[No document reference: stock figures.]", Luna) and two raw values ("a gap of 0.0 weeks", Luna).
  Figures now read "The current stock position shows ...".
- `gpt-6-luna` at none began to comment on its input instead ("with no changes recorded"), and again
  one output broke off as invalid JSON (H1).
- A new fault the judge caught: saying a quantity is "as indicated by" or "supported by" the current
  stock position (P1a, four outputs), which the input does not say; the side effect of telling the
  model to name the stock position. The judge also caught Sol medium narrowing an action's scope
  ("for NanoCircuitX" where the action covers both products), missed by the reading.
- `gpt-6.1-sol` low still moves checks to "before ordering" (four outputs).

## Round 4: the stock-position clause taken out (2026-10-24)

The round 3 phrasing "as indicated by the current stock position" came from the clause asking the
model to name the stock position: seven outputs tied a quantity to it ("the current stock position
supports the suggested quantity of 780 units", "based on the current stock position"; the judge failed
four, the three with "based on" or "reflect" passed), against one in round 2. The user: the source
need not be named, since the rule already says a finding with empty documents takes no brackets.
The clause was taken out; the rest of the round 3 revision kept. Round 3's files were kept with
`_revised_v1` in their names. Same three settings and 25 inputs
(`eval_results/approval_model_comparison_revised.jsonl`, `eval_results/approval_model_judged_revised.jsonl`,
$0.20, `eval_results/approval_model_issues_revised.json`, side by side:
`eval_results/approval_model_judged_revised.md`):

| Setting | P1 fail (judge): round 2 → 3 → 4 | Reading: round 2 → 3 → 4 | Time | Cost per request |
|---|---:|---:|---:|---:|
| `gpt-6-luna`, none, temperature 0 | 5 → 3 → 1 | 4 → 8 → 7 | 3.1 s | $0.0002 |
| `gpt-6.1-sol`, medium | 1 → 2 → 1 | 2 → 1 → 3 | 10.8 s | $0.0070 |
| `gpt-6.1-sol`, low | 2 → 3 → 0 | 7 → 4 → 2 | 4.6 s | $0.0052 |

- No output ties a quantity to the stock position; no "pipeline", no empty or made-up brackets.
- `gpt-6-luna`: raw values ("a gap of 0.0 weeks", twice), stray or wrong citations after stock figures
  (two), a comment on its input ("no changes are recorded"), meaning changed (two); no invalid JSON this
  time.
- `gpt-6.1-sol`, both efforts: the one recurring fault is a check moved to "before placing the order"
  where the actions place the order now (medium three, low two; eval_08 and H1 in both). Across rounds
  2-4 it appears in both efforts, so it is not an effect of low effort alone. The judge passes it.
- The judge's one Sol-medium fail (eval_05, P1c: "requested" for the input's suggested quantity) is a
  stricter reading than the user's.

## Round 5: the analysis says when each check comes (2026-10-24)

Why Sol moved checks to "before ordering": the analysis agent's actions often gave no timing ("Obtain
confirmed lead times and shipment milestones for this order, then replace the provisional planning
allowance ... and reassess ordering deadlines"), while the approval prompt asks for "an action that
checks something before the order is placed" to be written as a condition. Sol had to decide which
checks come first, and put supplier-wide checks before an order other actions placed now. The same
gap is in the analysis report the buyer reads. The user: fix it in the analysis prompt, and leave the
approval prompt as it is; a new approval rule would leave the agent without footing when an action
gives no timing.

`prompts/v5/analysis.md`, recommended_actions, in two wordings:
- v1: a check action says whether it comes before the order, alongside it or after it, with no date
  needed. The Kestrel and Nordic dev runs put supplier-wide checks "before placing each order" next to
  "place the BearingCap ... orders immediately": the same judgement, now in the analysis; and dates
  dropped out of Kestrel's actions ("it needs no date" read as no dates).
- v2 (kept), as the user set it: judge the timing realistically against the item's latest order date
  (new_latest_po_date or recommended_latest_po_date): whether there is time to do it before the order,
  or it has to run alongside or after so that the order is not late; an action whose timing is already
  clear is left as it is, and no date or deadline the action needs is dropped. No fixed rule.
  The example's GadgetY action got "before placing it".

Dev runs (Sol medium, both ready at the first attempt, 16 of 16 blocking checks):
`dev/runs/2026-10-24_205451_action-timing-judged-against-the-latest-order-date` (Kestrel: overdue
orders "immediately. In parallel, ..."; ValveSeat's commitment "alongside order placement rather than
delaying release"; the remaining orders' commitments "before placement", with their dates) and
`dev/runs/2026-10-24_205625_...` (Nordic: the customs check "alongside the immediate HexBoltM12 order,
without delaying it. Complete these checks before placing the remaining orders").

The 25 analyses rebuilt with Sol medium (`scripts/build_v5_base_outputs.py --timing`; v1:
`eval_results/v5_base_outputs_timing_v1.jsonl`, v2: `eval_results/v5_base_outputs_timing.jsonl`;
`eval_results/v5_base_outputs.jsonl` is left as the judge-validation base). Counted by wording
(`scripts/count_action_timing.py`):

| Analyses | Actions | "before" the order | "alongside" | Dates in actions | Overdue item and "before", without "now" |
|---|---:|---:|---:|---:|---:|
| Base (before the rule) | 63 | 13 | 4 | 38 | 3 |
| v1 | 58 | 27 | 32 | 38 | 9 |
| v2 | 64 | 15 | 47 | 72 | 0 |

Purchase requests written from the v2 analyses, the revised approval prompt unchanged
(`eval_results/approval_model_comparison_timing.jsonl`; judged:
`eval_results/approval_model_judged_timing.jsonl`, $0.21; side by side:
`eval_results/approval_model_judged_timing.md`). The run stopped once when the OpenAI credit ran out
and resumed after the user added credit.

| Setting | P1 fail (judge) | Reading | Time | Cost per request |
|---|---:|---:|---:|---:|
| `gpt-6-luna`, none, temperature 0 | 2 | 7 | 3.6 s | $0.0002 |
| `gpt-6-luna`, low | 3 | 6 | 4.1 s | $0.0003 |
| `gpt-6-luna`, medium (default) | 4 | 4 | 10.3 s | $0.0006 |
| `gpt-6.1-sol`, medium | 0 | 0 | 12.8 s | $0.0076 |
| `gpt-6.1-sol`, low | 1 (P1c, a stricter reading) | 0 | 5.3 s | $0.0056 |

- `gpt-6-luna` none, read after the user asked why it had been left unread (Claude had dropped it on
  its own reading of rounds 3-4, not on the user's decision): an item not requested named (eval_01
  "PlasticCap"); comments on its input ("No changes or additional actions are specified", "no quantity
  changes specified"); figures misstated (eval_14 "delivery at the 18-week lead time is expected after
  15 weeks", for a 15-week gap; H2 "for delivery by 2026-03-16", the bridging delivery given as the
  order's); a finding garbled (H2 "its history of rarely used substitutes"); raw values ("a gap of
  0.0 weeks", twice). No timing fault and no invalid JSON.
- `gpt-6-luna` low and medium, added at the user's question (2026-10-25: Luna with reasoning had not
  been run since round 2), read the same way. Low: the quality finding attributed to the supplier
  (eval_08), a citation left after stock figures (eval_17, H1), "a gap of 0.0 weeks" (eval_18), a
  finding inverted (H2 "ComponentZ has historically been substituted when ..."), frequencies weakened
  (H3 "can experience" for "often experiences"). Medium: the same attribution (eval_08), the bridging
  date given as the order's delivery (eval_20 "place the ... order on 2026-03-02 for delivery by
  2026-05-25"), the input mentioned ("A finding records ..."), the same weakened frequencies (H3).
  Reasoning helps Luna a little (7 → 6 → 4) but it keeps misplacing figures and wording that Sol
  carries over cleanly.
- Every "before ordering" left in the Sol texts carries over an analysis action that says so, for an
  item with time before its latest order date (eval_09 MicroProcessorX, eval_13, eval_16, H1 GadgetY,
  H4); none holds back an overdue order.
- A change seen in the v2 analyses, for the user to weigh: provisional quantity uplifts for surges the
  documents do not size (eval_15 650 → 813, eval_16 1,629 → 2,444, H4 520 → 624; reasons say
  "provisional planning allowance ... not a documented demand forecast"), against none in the base
  and two in v1. Allowed by J6b when marked as an estimate; whether the timing rule caused them is not
  shown.

## Round 6: the chosen setting on the analyses of changelog item 25 (2026-10-25)

After the decision, the analysis prompt changed (changelog item 25: quantity changes on UK demand
only; an estimated size says what it is based on; lead-time changes in 11 scenarios instead of 7, no
quantity changes). `gpt-6.1-sol` at low wrote the 25 requests again from those analyses
(`scripts/compare_approval_models.py --estimate`; `eval_results/approval_model_comparison_estimate.jsonl`,
judged: `eval_results/approval_model_judged_estimate.jsonl`, $0.07). 6.0 s and $0.0057 per request.

- Judge: P1c failed on 2 of 25, both stricter readings than a person's ("requested quantity" for the
  input's suggested quantity, eval_19; "in parallel with and without delaying" for "do not delay",
  eval_20).
- Reading: no fault of the writer's. The new lead-time changes were carried over with their status
  ("a provisional analyst estimate increases both lead times from 18 to 20 weeks, pending validation
  alongside order placement"), and the unknown market as the analysis left it ("although the market
  concerned is not identified"; "retains 650 units ... with immediate verification of the surge's
  market, persistence and UK forecast coverage").
- Carried over from the analysis: "even if ordered on the snapshot date" (eval_06, eval_20) and
  "three weeks of lead-time headroom" (eval_13: 27 weeks to stockout against a 24-week lead time).
  Neither is a fault (the user, 2026-10-25): "snapshot date" is a fine term for the reader, and the
  headroom is correct and came from the analysis.

## Decision

**`gpt-6.1-sol` at effort low, decided by the user on 2026-10-25** (`AGENT_MODELS["approval"]`,
`AGENT_EFFORT["approval"]` in `services/agents.py`), on round 5, the only round with both prompts as
they now are: no issue in 25 outputs at low as at medium, 5.3 s against 12.8 s, $0.0056 against
$0.0076 per request. The user: compare on the latest prompts only, and 25 outputs are enough; no second
run. The earlier rounds, where low moved checks before the order more often than medium, ran on
prompts since changed.

# Choosing the purchase request writer

**2026-10-25.** The case for the approval agent's model, effort and prompt, in one place. The full
record of every run is in `docs/approval_model_comparison.md` (rounds 1–5); this is the summary to
cite.

## 1. What the agent does

The approval agent writes the two paragraphs of the purchase request that the approvers (Purchasing
Manager, Operations Director, Finance Officer) read: Purpose (1–2 sentences) and Justification (2–4).
Its input is settled before it runs: the requested items with their figures after any lead-time
change, the analysis's findings with their document numbers, the changes and the actions. It needs
little reasoning. What it needs is to carry facts, figures and wording over exactly, add nothing, and
cite only the documents given. The rest of the document (item table, document list, approval block)
is filled in by code.

## 2. How it was tested

- The same 25 inputs for every model: the approval input saved with each v5 base output, so every
  model got exactly what the pipeline gives (`scripts/compare_approval_models.py`).
- Candidates: `gpt-4o` (in the role since the start, never compared), `gpt-6-luna` (cheapest),
  `gpt-5.4-mini` (the email agent), `gpt-6.1-sol` (the analysis model). Not Claude or Gemini: Haiku is
  the judge that checks this text, and Gemini answered with 429/503 errors (Gemini 4 Argon: 429 on 5
  of 5 calls, and $4/$20 per million tokens from 2026-12-01).
- Each reasoning model at set efforts, checked against the API first: `gpt-6-luna` and `gpt-5.4-mini`
  take none to xhigh and accept temperature 0 only at none; `gpt-6.1-sol` takes low to xhigh, at
  temperature 1 only. The first run had left every model at its default and not recorded it.
- Three readings of every text:
  1. Code: P2 (sentence counts) and P3 (citations to documents that exist).
  2. The judge chosen in `docs/judge_selection.md` (Haiku, thinking at high effort): P1a–c (nothing
     added, no judging words, no overstatement), three runs, majority verdict.
  3. Reading every text by hand against its input, before the judge's verdicts were seen, with each
     issue written down (`eval_results/approval_model_issues_*.json`).

## 3. What the outputs showed

### The models (round 2, the first prompt; 25 outputs each)

| Setting | P1 fail (judge) | Outputs with an issue (reading) | Time | Cost per request |
|---|---:|---:|---:|---:|
| `gpt-4o` | 19 | 22 | 2.8 s | $0.0047 |
| `gpt-5.4-mini`, none / low | 6 / 15 | 18 / 19 | 2.0 / 2.8 s | $0.0018 / $0.0023 |
| `gpt-6-luna`, none / low / medium | 5 / 5 / 3 | 4 / 9 / 5 | 3.2 / 3.9 / 10.9 s | $0.0002–0.0006 |
| `gpt-6.1-sol`, low / medium | 2 / 1 | 7 / 2 | 5.5 / 11.6 s | $0.0049 / $0.0064 |

What each wrote, in the approvers' text:
- **`gpt-4o`** added its own assessment of the supplier ("a proven track record, supporting the
  decision to proceed with this supplier") in 8 of 25 and got figures wrong ("stockout in 8 weeks"
  for 12). Its P2 fails (13 of 25) also showed it ignoring the length rule.
- **`gpt-5.4-mini`** kept the length rule best but wrote this tool's terms ("the dates shown in the
  pipeline"), false statements ("the revised order dates and quantities" where nothing was revised),
  citations to documents that do not exist ([3], [4] with two given) and empty placeholders ("[ ]").
  More effort made it worse. The metrics alone pointed to it; reading the texts ruled it out.
- **`gpt-6-luna`** carried field values over ("an expected stockout of 0") and broke off as invalid
  JSON in 2 of its 100 outputs at effort none (rounds 2–5).
- **`gpt-6.1-sol`** carried facts and conditions over cleanly.

### The prompt (rounds 3–4)

Reading the outputs showed that some recurring faults came from the prompt and the input, not the
models:
- The prompt itself said "the pipeline's figures" and "The pipeline writes the rest of the document";
  the models repeated it.
- Findings from the stock figures arrive with `"documents": []`; the models wrote the empty list out
  as "[ ]" or filled it with "[no document]" or a wrong number.

Revised as the user set it out: the writer is a **purchasing officer putting the request to the
approvers for their decision, objectively and neutrally, not arguing for the purchase** ("buyer"
could read as someone persuading); figures in plain words, never field names, no mention of the tool
or the input; findings with empty documents take no brackets. A first wording also asked the model to
name "the current stock position" as the source; seven outputs then tied quantities to it ("the
current stock position supports the suggested quantity of 780"), so that clause was taken out.

With the prompt revised, "the pipeline", field names and empty brackets were gone from the Sol texts.

### Actions moved before the order (rounds 4–5)

One fault remained in Sol at both efforts: a check moved to "before placing the order" where the
actions place the order now, e.g. "The request is conditional on confirming actual lead times before
order placement" for an overdue item. The cause was upstream: the analysis agent's actions often gave
no timing ("Obtain confirmed lead times ... for this order, then ... reassess ordering deadlines"), so
the writer had to decide which checks come first. The same ambiguity was in the analysis report the
buyer reads.

Fixed in the analysis prompt, not the approval prompt (the user: a new approval rule would leave the
writer without footing whenever an action gave no timing). The first wording ("say before, alongside
or after the order; no date needed") made the analysis put supplier-wide checks "before placing each
order" next to "place ... immediately", and dropped dates. The kept wording, as the user set it, asks
the model to **judge the timing realistically against the item's latest order date**, leave actions
whose timing is already clear as they are, and keep every date. Across the 25 analyses:

| Analyses | "Alongside" the order | Dates in actions | Overdue item with a check "before" it |
|---|---:|---:|---:|
| Before the rule | 4 | 38 | 3 |
| First wording | 32 | 38 | 9 |
| Kept wording | 47 | 72 | 0 |

For example, H2 went from "Before placing the order, obtain current prices and quotation validity" to
"Immediately obtain current prices, quote validity ... alongside today's order placement, and resolve
any required price approval without delaying release".

### The final comparison (round 5, both prompts revised; 25 outputs each)

| Setting | P1 fail (judge) | Outputs with an issue (reading) | Time | Cost per request |
|---|---:|---:|---:|---:|
| `gpt-6-luna`, none | 2 | 7 | 3.6 s | $0.0002 |
| `gpt-6-luna`, low | 3 | 6 | 4.1 s | $0.0003 |
| `gpt-6-luna`, medium | 4 | 4 | 10.3 s | $0.0006 |
| `gpt-6.1-sol`, low | 1 (a stricter reading) | 0 | 5.3 s | $0.0056 |
| `gpt-6.1-sol`, medium | 0 | 0 | 12.8 s | $0.0076 |

Every "before ordering" left in the Sol texts carries over an analysis action that says so, for an
item with time before its latest order date; none holds an overdue order back. Luna, at every effort,
still misplaced figures and wording: the bridging delivery date given as the order's ("place the order
on 2026-03-02 for delivery by 2026-05-25"), a finding inverted ("ComponentZ has historically been
substituted" for "ComponentZ historically substitutes"), frequencies weakened ("can experience" for
"often experiences"), a quality finding attributed to the supplier. More reasoning helped Luna a
little (7 → 6 → 4); Sol carried the same inputs over cleanly even at low effort (18 reasoning tokens).

## 4. What each round showed

- **Round 1: the metrics and the texts disagreed.** `gpt-5.4-mini` kept the sentence-length rule best
  (24 of 25) and `gpt-4o` worst (13); read, `gpt-5.4-mini` put "the dates shown in the pipeline",
  "the revised order dates" (nothing was revised) and "[ ]" into the approvers' text, and `gpt-4o`
  added "a proven track record, supporting the decision to proceed with this supplier". Sol carried
  the facts over cleanly.
- **Round 2 (efforts set): reasoning is not what this agent needs.** Luna wrote as well with no
  reasoning as at medium; `gpt-5.4-mini` got worse with more effort; Sol at low halved the time but
  moved checks "before ordering" where the actions order now. The judge's P1a-c ranked the settings
  as the reading did, but do not see citations, placeholders or missing actions.
- **Round 3: recurring faults came from the prompt, not the models.** The prompt's own "the
  pipeline's figures" was repeated in the text; findings from the stock figures arrive with an empty
  document list, written out as "[ ]". The writer was made a purchasing officer putting the request
  to the approvers neutrally (the user's wording); a clause asking it to name "the current stock
  position" then made seven outputs tie quantities to it ("the current stock position supports the
  suggested quantity of 780 units"), so it was taken out (round 4).
- **Round 5: the last fault came from the analysis agent.** Sol moved checks before the order because
  the analysis's actions gave no timing ("Obtain confirmed lead times ... for this order, then ...
  reassess ordering deadlines"), and the approval prompt asked it to decide which checks come first.
  Fixed in the analysis prompt (changelog item 24), the fault was gone in both Sol settings. Luna,
  read at every effort under the same conditions, still misplaced figures and wording.
- **What this taught:** read the outputs, not only the scores, and trace a fault to where it arises
  (the prompt, the input, the agent upstream) before blaming the model.

## 5. Decision

**`gpt-6.1-sol` at effort low** (decided by the user, 2026-10-25).

**Why Sol: it carries the facts, figures and conditions over exactly, and adds nothing.** The
approvers decide on this text, so what matters is that it says what the analysis says, no more and no
less. On the same inputs, the others did not:
- `gpt-6-luna`, at every effort: the bridging delivery date given as the order's ("place the order on
  2026-03-02 for delivery by 2026-05-25"), a finding inverted ("ComponentZ has historically been
  substituted" for "ComponentZ historically substitutes"), frequencies weakened ("can experience" for
  "often experiences");
- `gpt-5.4-mini`: this tool's terms, false statements and empty placeholders in the approvers' text;
- `gpt-4o`: its own assessment of the supplier, and figures wrong.

**Why low effort:** on the latest prompts (round 5) low and medium both had no issue in 25 outputs; low takes less than half the time (5.3 s against
12.8 s) and about three quarters of the cost ($0.0056 against $0.0076). The user compared on the
latest prompts only: in rounds 2–4 low had more outputs with an issue (13 against 6, 9 of them a check
moved before the order), but on prompts since changed, and the cause of that fault was removed in the
analysis prompt.

## 6. A person in the loop

Every comparison was read by hand as well as scored, and the reading changed the outcome more than
once: it ruled out the model the metrics favoured (`gpt-5.4-mini`), traced recurring faults to the
prompt and to the analysis agent's actions rather than to the models, and set the prompt changes
(the purchasing officer's neutral role, the timing judged against the latest order date). When Luna
had been left out of reading on an earlier impression, the user asked why; it was then read at every
effort under the same conditions before being ruled out.

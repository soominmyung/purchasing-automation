# Supplier email model comparison

**2026-10-25.** Which model should write the supplier email's requests (`AGENT_MODELS["email_draft"]`
and `["email_rewrite"]` in `services/agents.py`), and what the prompt had to say. `gpt-5.4-mini` had
held the role since the v5 prompt tuning of 2026-10-22 (`prompts/CHANGES.md`), chosen there on one or
two development cases, never on the same inputs as other models.

## What the agent does

It writes only the requests paragraph of an email to the supplier; code adds the greeting, the
purpose, the item list, the closing and the signature. Its input: the supplier, the items (name,
quantity, requested delivery date), `points_to_clarify` (the analysis agent's questions marked for
the supplier), `ask_earliest_dispatch` (the Critical items) and `other_items` (items of this supplier
outside the email). It must ask what the points need without revealing anything internal and
without naming a problem the supplier had before.

## Method

- The email input of each of the 25 analyses in `eval_results/v5_base_outputs_estimate.jsonl`, the
  same for every setting (`scripts/compare_email_models.py`).
- Seven settings: `gpt-5.4-mini` none and low, `gpt-6-luna` none, low and medium, `gpt-6.1-sol` low
  and medium (effort and temperature as the API accepts, `docs/approval_model_comparison.md` round 2).
  No Claude model (the judge, Haiku, checks this text); no Gemini (unavailable in testing).
- Each email through the production loop: draft, the validator (`gpt-4o-mini` after a keyword check),
  a rewrite by the same model after a fail, at most three drafts.
- Checks: E3 in code (no item of `other_items` named); E1, E2, E4-E7 by the chosen judge, three runs,
  majority. Every final email also read by hand, before the judge's verdicts were seen.
- The code passes the email agent's effort since this comparison: before, `_llm` was called without it
  for the email draft, rewrite and validator, so a setting would have been ignored.

## Round 1: the prompt as it was

| Setting | Judge: any fail | Reading |
|---|---:|---|
| `gpt-5.4-mini` none / low | 11 / 13 | analyst's questions dropped (eval_08 "how the logistical complications affect confirmed delivery dates"); vague additions ("If any detail is needed to proceed smoothly…", "so we can plan accordingly") |
| `gpt-6-luna` none / low / medium | 10 / 10 / 5 | raw JSON in the email (`{"requests":"Could you please…"}`) 7 / 1 / 3 times; "the cause of the loose-terminal issue" |
| `gpt-6.1-sol` low / medium | 12 / 11 | none (two borderline: an ongoing strike named) |

The judge and the reading disagreed because of the checks, not the emails: E4 failed questions the
prompt requires to be left out (items of `other_items`), E5 counted the required earliest-dispatch
question beside an expedited-delivery point as a repeat, E7 failed neutral wording taken over from a
point. The raw JSON: Luna wrote line breaks unescaped inside the JSON string; `documents.email_parts`
now reads it with `json.loads(strict=False)`, which recovers 10 of the 11.

The validator passed all 175 emails, its one fail a false alarm (it objected to an item's name).

## Round 2: the user's corrections, and a fault of the rewrite

The user's decisions: E1 no longer looks for urgency; E4 fails a question about an item outside the
email; the prompt's "faster shipping option" question is dropped (for these imports air freight is
the only answer) and the earliest-dispatch question is added only when no point already asks for an
earlier delivery; E7 checks no purchase order promised, no item list repeated, and the tone of an
email to a business partner, not "own words" (the user: copying a point that already reads well is
fine).

Claude's rewrite of the point rule put "Where a point already reads well as a question to the
supplier, use it as it is" before the rule on past problems, as a separate case. The analyst's points
are questions, so the models copied them, past problems included:
- `gpt-5.4-mini`, eval_12: "whether **the loose-terminal failure in the January PowerModuleZ batch**
  was isolated" (a past fault, its month, and a product not in the order);
- "**the reported** shipment delays", "**the reported** temporary reduction in quality", "Have
  operations resumed and has the backlog cleared?" in other settings.

The user's correction: the email's style applies to every point, question or problem alike.

## Round 3: the prompt as it now is

`prompts/v5/email_draft.md`: every point written up in the style of the email, keeping its wording
where it suits; "Whatever a point says, the question says nothing about a problem the supplier had
before or when"; wording from INPUT rewritten where it does not suit the email; no "also in different
words" in the repeat rule; no style examples (removed from the purchase request and the email: the
ones in `docs/Examples.zip` say "our stock review" and ask in lists); no "pipeline". E4 then rewritten
so that an indirect question about a past problem covers its point (it had failed what E2 requires).

| Setting | E2 | E4 | E5 | E6 | Judge: any fail | Reading | Time | Cost per email |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gpt-5.4-mini` none | 7 | 0 | 2 | 0 | 8 | 7 | 2.4 s | $0.0010 |
| `gpt-5.4-mini` low | 3 | 1 | 6 | 0 | 8 | 6 | 2.5 s | $0.0013 |
| `gpt-6-luna` none | 0 | 0 | 1 | 0 | 1 | 3 | 2.3 s | $0.0001 |
| `gpt-6-luna` low | 2 | 0 | 0 | 0 | 2 | 1 | 3.4 s | $0.0002 |
| `gpt-6-luna` medium | 1 | 0 | 0 | 0 | 1 | 0 | 5.2 s | $0.0003 |
| **`gpt-6.1-sol` low** | 0 | 0 | 0 | 2 | 2 | **0** | **3.3 s** | $0.0024 |
| `gpt-6.1-sol` medium | 0 | 0 | 0 | 1 | 1 | 0 | 5.0 s | $0.0033 |

Files: `eval_results/email_model_comparison.jsonl`, `email_model_judged.jsonl` (E4 rewritten),
`email_model_v3_judged.jsonl` (before), side by side: `eval_results/email_model_comparison.md`.
Rounds 1 and 2: `email_model_v1_*`, `email_model_v2_*`.

The same point, eval_12, "Was the loose-terminal failure in the January PowerModuleZ batch isolated…":
- `gpt-5.4-mini` none: "whether the loose-terminal issue in the January PowerModuleZ batch was isolated";
- `gpt-6-luna` none: "whether the cause of the loose-terminal failure affects any products in this order";
- `gpt-6.1-sol` low: "how terminal integrity will be verified for the items listed above";
- `gpt-6.1-sol` medium: "whether any terminal-integrity considerations apply to them".

What the reading found in round 3:
- `gpt-5.4-mini` copied the past problems in at both efforts ("the reported quality inconsistencies",
  "the recent quality issues"), and once wrote an Armenian word into the email ("what … partial
  deliveries you can առաջարկ?"). The prompt does not fix it.
- `gpt-6-luna` softens the problem words but keeps them ("any product-standard discrepancies",
  "the expected recovery timeline and backlog clearance schedule"); the user: a slight difference of
  preference.
- `gpt-6.1-sol` rewrites them as ordinary questions about this order ("confirm compliance with the
  applicable product standards", "the production and order fulfilment schedule").
- The judge's E6 fails on Sol (eval_05, 18, 20): the points already ask for expedited or partial
  delivery by the requested date, so Sol, as the prompt says, added no dispatch question; the judge
  read "an earlier delivery" as earlier than the requested date rather than than the standard lead
  time.
- The judge wavered on Luna's "the cause of the loose-terminal failure" (passed twice, failed once):
  E2's second sentence ("a question about how a step of this order will proceed passes") outweighed
  its first.
- The validator passed every email again, the January PowerModuleZ sentence and the Armenian word
  included.

## Decision

**`gpt-6.1-sol` at effort low, decided by the user on 2026-10-25**, on round 3: no fault found by
reading, the fastest of the settings without one (3.3 s), and the same model and effort as the
purchase request. `gpt-6-luna` medium was as clean, at about a tenth of the cost, but slower and with
a history of malformed JSON; `gpt-5.4-mini` is out on copying past problems in.

The validator: still `gpt-4o-mini` with the keyword check; it caught nothing across 525 emails. To be
decided.

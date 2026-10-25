# Supplier email model comparison

**2026-10-25.** Which model should write the supplier email's requests (`AGENT_MODELS["email_draft"]`
and `["email_rewrite"]` in `services/agents.py`), and what the prompt had to say. `gpt-5.4-mini` had
held the role since the v5 prompt tuning of 2026-10-22 (`prompts/CHANGES.md`), chosen there on one or
two development cases, never on the same inputs as other models.

## In short

- **The deciding difference: how each model asks about a supplier's past problem.** The analyst's
  points often name what went wrong ("Was the loose-terminal failure in the January PowerModuleZ batch
  isolated...?"), and the email leaves the company. Sol rewrites such a point as an ordinary question
  about this order ("how terminal integrity will be verified for the items listed above"); Luna
  softens it but leaves the trace ("the cause of the loose-terminal failure"); `gpt-5.4-mini` copies
  it ("the loose-terminal issue in the January PowerModuleZ batch"), and kept doing so whatever the
  prompt said.
- **The prompt had to say it for every point, not as an exception.** A rule "use a point as it is
  where it reads well" put before the past-problem rule let every point through, since the points
  are questions; the user's correction (style for every point, no past problem whatever a point
  says) fixed it for Sol and Luna, not for `gpt-5.4-mini`.
- **The checks had to be corrected too.** The judge failed questions the prompt requires to be left
  out, counted required questions as repeats, and failed an indirect question for not asking about
  the problem itself; each wording was corrected and the emails scored again.
- **The validator caught nothing** in 525 emails, the PowerModuleZ sentence included.
- Sol low was chosen over Luna medium, the other clean setting: more expensive, faster, and it asks
  indirectly where it matters.

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
ones in `docs/Examples.zip` say "our stock review" and ask in lists); no "pipeline". Then two checks
were rewritten and the same emails scored again: E4, so that an indirect question about a past problem
covers its point (it had failed what E2 requires); E6, so that an earlier delivery means sooner than
the standard lead time allows, such as expedited or partial delivery by the requested date (it had
read "earlier" as earlier than the requested date, and failed Sol for not adding a dispatch question
the prompt says to leave out when a point already asks for that).

| Setting | E2 | E4 | E5 | E6 | Judge: any fail | Reading | Time | Cost per email |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gpt-5.4-mini` none | 6 | 0 | 3 | 0 | 6 | 7 | 2.4 s | $0.0010 |
| `gpt-5.4-mini` low | 2 | 1 | 4 | 0 | 6 | 6 | 2.5 s | $0.0013 |
| `gpt-6-luna` none | 2 | 0 | 1 | 0 | 3 | 3 | 2.3 s | $0.0001 |
| `gpt-6-luna` low | 2 | 0 | 0 | 0 | 2 | 1 | 3.4 s | $0.0002 |
| `gpt-6-luna` medium | 1 | 0 | 1 | 0 | 2 | 0 | 5.2 s | $0.0003 |
| **`gpt-6.1-sol` low** | 0 | 0 | 0 | 0 | **0** | **0** | **3.3 s** | $0.0024 |
| `gpt-6.1-sol` medium | 0 | 0 | 0 | 0 | 0 | 0 | 5.0 s | $0.0033 |

With both checks rewritten, the judge and the reading agree: Sol none at either effort; Luna's fails are
the problem words it keeps ("the cause of the loose-terminal failure", now failed, as read; "any
product-standard discrepancies"); `gpt-5.4-mini`'s are the past problems it copies in.

Files: `eval_results/email_model_comparison.jsonl`, `email_model_judged.jsonl` (E4 and E6 rewritten),
`email_model_v3_judged.jsonl` (before both), `email_model_v3b_judged.jsonl` (E4 only), side by side:
`eval_results/email_model_comparison.md`.
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
- Before E6 was rewritten, the judge failed Sol on eval_05, 18 and 20: the points already ask for
  expedited or partial delivery by the requested date, so Sol, as the prompt says, added no dispatch
  question; the judge read "an earlier delivery" as earlier than the requested date. With the new
  wording all six passed in every run.
- Before E6 was rewritten, the judge wavered on Luna's "the cause of the loose-terminal failure"
  (passed twice, failed once): E2's second sentence ("a question about how a step of this order will
  proceed passes") outweighed its first. In the last scoring it failed it.
- The validator passed every email again, the January PowerModuleZ sentence and the Armenian word
  included.

## Decision

**`gpt-6.1-sol` at effort low, decided by the user on 2026-10-25**, on round 3, against
`gpt-6-luna` medium, the other setting with no fault found by reading:

| | `gpt-6.1-sol` low | `gpt-6-luna` medium |
|---|---|---|
| Cost per email | $0.0024 | $0.0003 |
| Time per email | **3.3 s** | 5.2 s |
| How it asks about a supplier's past problem | **as an ordinary question about this order**: "how terminal integrity will be verified", "confirm compliance with the applicable product standards", "the production and order fulfilment schedule" | keeps the problem words, softened: "the cause of the loose-terminal failure" (none), "any product-standard discrepancies", "the expected recovery timeline and backlog clearance schedule" |
| JSON format | sound in all three rounds | three malformed in round 1, one past repair |

- **It costs more:** about eight times as much per email, but about $0.21 more per 100 suppliers,
  since each supplier gets one email.
- **It is faster:** 3.3 s against 5.2 s, the fastest of the settings without a fault.
- **It asks indirectly where it matters:** the email goes outside the company, and the analyst's
  points often name what went wrong with the supplier. Sol rewrites them as questions about this
  order that reveal nothing the company has heard; Luna softens them but leaves the trace (the user:
  a slight difference of preference, but the deciding one).
- It is also the same model and effort as the purchase request.

`gpt-5.4-mini` is out on copying past problems in.

The validator: still `gpt-4o-mini` with the keyword check; it caught nothing across 525 emails. To be
decided.

# Analysis model comparison

**First round, 2026-10-24.** Which model should write the analysis (`AGENT_MODELS["analysis"]`
in `services/agents.py`), weighing quality, cost and time together. GPT-4o had been the analysis
model since the start of the project; this is the first comparison against newer models.

## Why it was run

With prompt v5 the analysis decides each item's lead time and, in the same answer, works out the
figures that follow from it (risk level, gap, latest order date), which code then checks (C10,
`services/quality_checks.py`). GPT-4o got these figures wrong in every first answer across four
prompt revisions (7, 4, 4 and 8 of 20 values), mostly the order dates; each time a rewrite was
needed, and the rewrite was right only because C10's failure message gives the correct values.
A probe (`scripts/probe_order_dates.py`) showed GPT-4o understood the rule and its inputs, and
missed when it had to write the date without working (see "GPT-4o and date arithmetic"). Rather
than keep adjusting the prompt to one model, newer models at a similar or lower price were tried
on the same prompt.

## Method

- **Same prompt for every model**: `prompts/v5/analysis.md` as used in run
  `2026-10-24_003008` (one call; the output fields in the order of the work; `figures_after_changes`
  with `new_*` names). Only the analysis model changes (`scripts/run_dev_case.py --analysis-model`);
  its rewrites use the same model. Every other agent and the judge (`claude-haiku-5-5`) are as in
  production.
- **Two development cases** (`dev/cases/`):
  - *Kestrel*: a supplier-wide notice of lead times four weeks longer (18 → 22 weeks, so every
    item's figures change), one Critical item, a 50% demand rise under a contract.
  - *Nordic*: no lead-time change is warranted (a past customs hold the supplier says it has
    fixed), one Critical item, a 60% demand rise, a defective batch of one item.
- **One run per model and case.** Indicative only; see "Limits".
- **Measured**: analysis call time and total pipeline time; C10 errors in the first answer (of 20
  values, when a lead time changes); rewrites; analysis tokens and cost; judge cost; the verdict and
  warnings of the analysis checks and of the purchase request checks (P1-P3).
- **Prices** (standard tier, USD per 1M tokens, input / output): `gpt-4o` 2.50 / 10.00,
  `gpt-6.1-sol` 2.00 / 10.00, `gpt-6-luna` 0.10 / 0.50 (OpenAI pricing page, checked 2026-10-24);
  `claude-sonnet-5-5` 2.00 / 10.00, judge `claude-haiku-5-5` 0.10 / 0.50 (`docs/judge_model_costs.md`,
  2026-10-18). Reasoning ("thinking") tokens are billed as output and are included.

## Candidates

| Model | Tier (provider's description) | Reasoning | Why tried |
|---|---|---|---|
| `gpt-4o` | — (no longer in OpenAI's model list) | none | the current model |
| `gpt-6.1-sol` | mid tier: "near-Astra performance for complex work at a lower cost" | low to max, cannot be off | the newest model at GPT-4o's price |
| `gpt-6-luna` | efficiency tier: "focused, high-volume tasks" | none to max | 1/25 of GPT-4o's price |
| `claude-sonnet-5-5` | — | thinking | another provider at GPT-4o's price |

Reasoning effort was left at each model's default.

### How the candidates were chosen

**The pool looked at**: every model with a published price at the three providers whose APIs the
project already uses, from `docs/judge_model_costs.md` (13 models, priced 2026-10-18) and OpenAI's
model list and pricing page (checked 2026-10-24):

| Provider | Models looked at (input / output, $ per 1M tokens) |
|---|---|
| OpenAI | `gpt-6-astra` 10 / 50, `gpt-5.5` 5 / 30, `gpt-5.6-sol` 4 / 20, `gpt-6.1-sol` 2 / 10, `gpt-6-sol` 2 / 10, `gpt-5.4-mini` 0.75 / 4.50, `gpt-4.1-mini` 0.40 / 1.60, `gpt-5.6-luna` 0.20 / 1.20, `gpt-5.4-nano` 0.20 / 1.25, `gpt-6-luna` 0.10 / 0.50 |
| Anthropic | `claude-sonnet-5-5` 2 / 10, `claude-haiku-5-5` 0.10 / 0.50 |
| Google | `gemini-3.8-flash` 0.75 / 3.75, `gemini-3.6-flash` 0.75 / 3.75, `gemini-3.7-flash` (no published price), `gemini-3.5-flash-lite` 0.30 / 2.50, `gemini-3.1-flash-lite` 0.25 / 1.50 |

**The criteria**, in the order they were applied:
1. **No dearer per token than GPT-4o** (2.50 / 10): the question was whether a newer model at the
   same cost does better, so a model that costs more had to beat it on quality by a margin first.
   Out: `gpt-6-astra` (4-5 times), `gpt-5.5` (2-3 times), `gpt-5.6-sol` (2 times).
2. **The current generation of each line**: an older model at the same price as a newer one of the
   same line has no reason to win. Out: `gpt-6-sol` (same price as `gpt-6.1-sol`), `gpt-5.6-luna`
   (`gpt-6-luna` replaces it, at half the price), `gpt-5.4-mini`, `gpt-5.4-nano` and `gpt-4.1-mini`
   (older generations, between the Sol and Luna tiers in price).
3. **Not the judge's model**: `claude-haiku-5-5` is the judge; an analysis it also wrote would be
   judged by itself.
4. **One per price tier and provider**, to see whether price buys quality: the mid tier of OpenAI
   (`gpt-6.1-sol`) and of Anthropic (`claude-sonnet-5-5`), at GPT-4o's price, and OpenAI's efficiency
   tier (`gpt-6-luna`), at 1/25 of it. GPT-4o stays in as the baseline.

**Not settled: Google.** `gemini-3.8-flash` (0.75 / 3.75) passes criteria 1-3 and was on the first
list, then dropped when the GPT-6 models were added, without a test or a reason recorded. Reasons it
could be left out, from this project's own record, all about other Gemini Flash models or about
prices: on 2026-10-22 the `gemini-3.7-flash` judge ran 30-109 s per call and then returned 504 and
429 errors; `gemini-3.7-flash` has no published price; it and `gemini-3.6-flash` returned 503 at the
first attempt on 2026-10-18; the Flash prices double on 2027-01-01; and `gemini-3.8-flash` spent
over 800 thinking tokens per judge call, billed as output. None of this was measured on
`gemini-3.8-flash` as an analysis model.

**Picked for a test (2026-10-24): `gemini-3.8-flash`.** The 429 errors had come from an account with no
credit; credit was added. From Google's price list (updated 2026-10-22): `gemini-4-argon`, the Gemini 4
model, costs 2 / 10 now but 4 / 20 from 2026-12-01, twice GPT-4o, so it fails criterion 1; Gemini 3.8
Flash, the current Flash, costs 0.75 / 3.75 and 1.50 / 7.50 from 2027-01-01, within it either way;
the other text models are older generations or previews. First calls, 2026-10-24 about 02:25: through
LangChain, 504 DEADLINE_EXCEEDED on a one-line prompt; through Google's SDK, 503 UNAVAILABLE after
40.8 s ("This model is currently experiencing high demand"), while `gemini-3.5-flash-lite` answered
the same prompt in 15.3 s.

**What happened next (2026-10-24, 02:24-02:30).** Flash was taken as the fair counterpart to Sol
(both mid tier); Flash-Lite is a tier lower, Luna's counterpart.
- `gemini-3.5-flash-lite`, started before that was settled: its first run failed with 503 UNAVAILABLE
  ("high demand") on the analysis call; stopped.
- A one-line prompt again: `gemini-3.6-flash` answered in 2.0 s, `gemini-3.8-flash` in 168.8 s, past
  the pipeline's 60 s limit for a call.
- `gemini-3.6-flash` (the previous Flash, same price as 3.8) on both cases, two runs each: **three of
  four runs failed** with 503 UNAVAILABLE on the analysis call, after the pipeline's retry. The one
  that ran (Kestrel, `dev/runs/2026-10-24_022836_…`): every figure after the change right at first, no
  rewrite, ready with one warning (J2b); analysis call 33.7 s, 4,977 / 7,391 tokens (mostly thinking),
  $0.031, as much as Sol; one supplier question ("Can KestrelComponents offer expedited freight or
  allocate buffer stock from existing warehouses …") and four general actions ("Arrange expedited
  shipment or source BearingCap from alternative suppliers to bridge its 10-week stockout gap").

**Reading.** Gemini Flash could not be compared on the same footing because it did not answer often
enough to run the cases: of six analysis calls attempted on two Flash models and Flash-Lite, four
were refused for demand and one took 169 s for a one-line answer. In the one run it completed it was
accurate, at Sol's cost, with questions and actions closer to Luna's. Availability is itself a
reason against it for a pipeline that buyers run on demand. To close the gap, run it on
the same cases or record the decision to leave it out.

## Results

| Model | Case | Analysis calls (s) | Total (s) | C10 errors, first answer | Rewrites | Analysis tokens in / out | Analysis cost | Judge cost | Verdict | Analysis warnings | Purchase request warnings |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4o | Kestrel | 10.3 + 10.7 | 39.3 | 8 | 1 | 9,406 / 2,220 | $0.0457 | $0.0014 | ready | J11, J2a | P1a, P2 |
| gpt-4o | Nordic | 8.9 + 5.9 + 5.5 | 46.5 | 1 | 2 | 14,055 / 2,385 | $0.0590 | $0.0028 | ready | J11 | P1a, P1b, P1c |
| gpt-6.1-sol | Kestrel | 38.9 | 58.2 | 0 | 0 | 4,121 / 2,151 | $0.0298 | $0.0017 | ready | none | P1a, P1c, P2 |
| gpt-6.1-sol | Nordic | 25.9 | 45.0 | — | 0 | 4,049 / 1,308 | $0.0212 | $0.0016 | ready | none | none |
| gpt-6-luna | Kestrel | 22.1 | 40.6 | 0 | 0 | 4,121 / 2,746 | $0.0018 | $0.0014 | ready | none | P1c |
| gpt-6-luna | Nordic | 16.0 | 32.7 | — | 0 | 4,049 / 1,973 | $0.0014 | $0.0014 | ready | none | P2 |
| claude-sonnet-5-5 | Kestrel | 26.7 | 44.6 | 0 | 0 | 6,001 / 3,880 | $0.0508 | $0.0016 | ready | J11 | P1a, P1b, P1c |
| claude-sonnet-5-5 | Nordic | 19.2 | 36.4 | — | 0 | 5,912 / 2,695 | $0.0388 | $0.0015 | ready | none | P1c |

"—": no lead-time change, so no figures for C10 to check (except GPT-4o on Nordic, which made one).

Runs (`dev/runs/`, not in git): GPT-4o `2026-10-24_003008_…`, `2026-10-24_004929_…`; Sol
`2026-10-24_003704_…`, `_003806_…`; Luna `_003856_…`, `_003941_…`; Sonnet `_004621_…`, `_004513_…`.

### What the analyses said

- **GPT-4o.** Kestrel: 8 of 20 figures wrong at first (all four order dates copied from the
  18-week input, three risk levels a level too low, one order date before the snapshot date).
  Nordic: raised HexBoltM12's lead time to 22 weeks for the customs hold the supplier says is fixed
  and gave it an order date before the snapshot date (C10); the second answer kept the lead time
  (J1b, J6a, J6b failed); the third dropped it. Its final Nordic analysis asks no supplier question
  and leaves out the defective AnchorPlug batch altogether, the fault already noted for v5 in
  `prompts/CHANGES.md`.
- **All three newer models**: every figure right at first, on both cases; no lead-time change for
  the fixed customs hold; a question and an action on the AnchorPlug batch.
- **gpt-6.1-sol**: the most thorough. A stock finding for every item; the new order dates written
  into the actions (ValveSeat by 2025-04-05, GasketKit by 2025-04-19, PistonRing by 2025-05-31);
  asked whether the suggested GasketKit quantity already includes the demand rise, to avoid
  counting it twice. Long: its Kestrel purchase request ran to six sentences (P2). The slowest.
- **gpt-6-luna**: concise and correct; on Kestrel one question only, so less depth than the others.
- **claude-sonnet-5-5**: pointed out that ValveSeat's order is due on the snapshot date and made it
  an action. It is of the same family as the judge (`claude-haiku-5-5`), which may favour it; to be
  checked in judge validation.
- The newer models write the new order dates into their text, which is why the dates are worked
  out by the model and checked by code rather than filled in by code alone.

### Setting up the runs (not counted)

Three Sonnet runs failed through the pipeline's code, not the model, and are kept in `dev/runs/`:
`_004047_…` and `_004220_…` read Sonnet's answer, a list of thinking and text blocks, as plain text
(fixed: `AIMessage.text`); `_004402_…` hit the 4,000-token output limit, used up by thinking (now
16,000). Sonnet also accepts only its default temperature. `langchain-anthropic` was installed for
these runs, which moved `langchain-core` from 1.4.8 to 1.7.0.

## GPT-4o and date arithmetic

The probe (`dev/runs/2026-10-24_002402_probe-order-date-arithmetic/`) asked GPT-4o "2025-09-06
minus 22 weeks" in three phrasings, three times each: right 6 times, a day out 3 times
(2025-04-04, 2025-04-06), each time with its working written out (it paired month lengths with the
wrong months in one). Asked to explain its first analysis answer, it named the right starting date
and the right subtraction for each item and still gave the wrong dates. In the analysis it has to
write the date in the JSON without working, and there it was weeks out or left the 18-week date.

## Reading the results

- On these two cases the newer models removed the problem that started the comparison: no first
  answer needed a rewrite. GPT-4o needed one or two rewrites per case, which made it slower in
  total than Luna and more expensive than Sol despite its faster calls.
- The newer models are slower per call (thinking): 16-39 s against GPT-4o's 6-10 s, but with no
  rewrites the totals are similar (33-58 s against 39-47 s).
- Cost per analysis: Luna about 1/25 of GPT-4o, Sol about two thirds, Sonnet about the same (its
  thinking adds output).

## Limits

- One run per model and case, two cases: not enough to tell a model's error rate or how
  consistent it is. Neither case has an item whose risk level changes with the lead time (each
  Kestrel item has the same risk at 18 and at 22 weeks), so a wrong lead time used for risk would
  not show.
- The prompt was written and revised on GPT-4o.
- The judge has not yet been validated (`docs/judge_validation.md`).
- The judge (`claude-haiku-5-5`) is of Sonnet's family; LLM judges are known to favour their own
  outputs and text familiar to them (references in `docs/judge_validation.md`, Candidates), so the
  judge's verdicts may have favoured Sonnet. In the second round it in fact failed Sonnet once
  wrongly (J7a); the accuracy comparison above rests on the C10 figures, which code checks.

## Second round, 2026-10-24

**Changed since the first round:** the Kestrel case has seven items (ThrustWasher and SpringClip
added, whose risk rises with the 22-week lead time, so a figure worked out at 18 weeks shows);
Findings are grouped by source; the judge is shown the pipeline's 18-week figures and, apart, the
figures after the changes (it had been shown only the latter and so read a proposed lead time as
the input's). Same prompt for every model. Two runs per model and case, sixteen in all
(`dev/runs/2026-10-24_012103_…` to `_013445_…`, labels "model comparison round 2").

| Model | Case | Run | Analysis calls (s) | Total (s) | C10 errors, first answer | Rewrites | Analysis tokens in / out | Analysis cost | Judge cost | Verdict | Analysis warnings | Purchase request warnings | Blocking failures per attempt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4o | Kestrel | 1 | 11.7 + 10.8 | 39.5 | 8 | 1 | 10,086 / 2,506 | $0.0503 | $0.0015 | ready | none | none | 1: C6, C7, C10 · 2: none |
| gpt-4o | Kestrel | 2 | 13.9 + 11.3 | 43.0 | 7 | 1 | 10,049 / 2,723 | $0.0524 | $0.0016 | ready | none | P1a, P1b, P1c, P2 | 1: C6, C7, C10 · 2: none |
| gpt-4o | Nordic | 1 | 8.7 + 8.1 | 41.4 | — | 1 | 9,031 / 1,606 | $0.0386 | $0.0030 | ready | J11 | P1c | 1: J6a, J6b · 2: none |
| gpt-4o | Nordic | 2 | 9.5 + 8.6 + 10.2 | 62.8 | 0 | 2 | 13,990 / 2,855 | $0.0635 | $0.0045 | ready (wrongly) | none | P1a, P1c, P2 | 1: J6c · 2: C10 · 3: J6c |
| gpt-6.1-sol | Kestrel | 1 | 38.8 | 56.9 | 0 | 0 | 4,365 / 2,389 | $0.0326 | $0.0018 | ready | none | none | 1: none |
| gpt-6.1-sol | Kestrel | 2 | 36.4 | 55.7 | 0 | 0 | 4,365 / 2,148 | $0.0302 | $0.0017 | ready | none | P2 | 1: none |
| gpt-6.1-sol | Nordic | 1 | 22.9 | 39.7 | — | 0 | 4,035 / 1,282 | $0.0209 | $0.0015 | ready | none | P1c, P2 | 1: none |
| gpt-6.1-sol | Nordic | 2 | 23.1 | 38.4 | — | 0 | 4,035 / 1,346 | $0.0215 | $0.0014 | ready | none | none | 1: none |
| gpt-6-luna | Kestrel | 1 | 30.9 | 46.8 | 0 | 0 | 4,365 / 3,355 | $0.0021 | $0.0015 | ready | none | P2 | 1: none |
| gpt-6-luna | Kestrel | 2 | 36.1 + 23.3 | 84.1 | 0 | 1 | 10,150 / 6,464 | $0.0042 | $0.0034 | ready | none | none | 1: J6b (judge error) · 2: none |
| gpt-6-luna | Nordic | 1 | 20.6 | 37.1 | — | 0 | 4,035 / 2,225 | $0.0015 | $0.0014 | ready | J11 | P1b, P1c | 1: none |
| gpt-6-luna | Nordic | 2 | 32.3 | 50.4 | — | 0 | 4,035 / 3,165 | $0.0020 | $0.0014 | ready | none | P1b, P1c, P2 | 1: none |
| claude-sonnet-5-5 | Kestrel | 1 | 27.6 + 19.8 | 70.6 | 0 | 1 | 15,669 / 7,732 | $0.1087 | $0.0035 | ready | J11 | none | 1: J7a (judge error) · 2: none |
| claude-sonnet-5-5 | Kestrel | 2 | 31.4 | 47.2 | 0 | 0 | 6,364 / 4,671 | $0.0594 | $0.0017 | ready | none | P1a, P2 | 1: none |
| claude-sonnet-5-5 | Nordic | 1 | 24.1 | 39.3 | — | 0 | 5,895 / 2,930 | $0.0411 | $0.0014 | ready | J11 | P1a, P1c, P2 | 1: none |
| claude-sonnet-5-5 | Nordic | 2 | 21.5 | 38.3 | — | 0 | 5,895 / 3,074 | $0.0425 | $0.0014 | ready | none | P1b, P1c, P2 | 1: none |

"—": no lead-time change in the first answer, so nothing for C10 to check.

**By model, over its four runs:**

| Model | First answers passing every blocking check, as judged | Same, judge errors set aside | Kestrel first answers with every figure after the change right (C10) | Rewrites | Total time | Analysis cost per run |
|---|---|---|---|---|---|---|
| gpt-4o | 0 of 4 | 0 of 4 | 0 of 2 (8 and 7 wrong); in Nordic it made a wrong lead-time change both times | 5 | 39-63 s | $0.039-0.064 |
| gpt-6.1-sol | 4 of 4 | **4 of 4** | 2 of 2 | 0 | 38-57 s | $0.021-0.033 |
| gpt-6-luna | 3 of 4 | **4 of 4** | 2 of 2 | 1 | 37-84 s | **$0.0015-0.0042** |
| claude-sonnet-5-5 | 3 of 4 | **4 of 4** | 2 of 2 | 1 | 38-71 s | $0.041-0.109 |

**What happened in the runs that did not pass at once:**
- **GPT-4o, Kestrel (both runs):** ThrustWasher's risk and gap at 18 weeks (High, no gap, for
  Critical and a 2-week gap), so no action bridged its gap (C6) and the summary did not name it
  (C7); seven or eight figures wrong in all.
- **GPT-4o, Nordic run 1:** raised HexBoltM12's lead time to 22 weeks for the customs hold the
  supplier says it has fixed; J6a and J6b failed it; the rewrite dropped it.
- **GPT-4o, Nordic run 2: a wrong change reached the documents marked ready.** The same 22-week
  change, first for HexBoltM12 alone, then for three items. The judge passed the change itself
  (J6a, J6b) every time and failed J6c for the opposite reason, that the supplier-wide delay should
  apply to *all* items. With the rewrites used up, the attempt with the fewest blocking failures
  (attempt 1) was judged again, and this time J6c passed: the same output, failed and then passed.
- **Luna, Kestrel run 2: a judge error made the analysis worse.** J6b failed a correct 50% quantity
  rise with a wrong reason ("the 50% uplift is applied to a whole quantity that includes the stock
  already held"; the suggested quantity is the quantity to order). The rewrite followed it and
  proposed 5,400 instead of 7,800 ("apply the 50% uplift to the suggested quantity above the 4,800
  units of available stock"), which J6b then passed. In the same rewrite Luna kept OilSeal at 18
  weeks, since its delivery date (March 2026) falls after the end of 2025, when the supplier expects
  the longer lead times to end: a reasoned call the first answer had not made.
- **Sonnet, Kestrel run 1:** J7a failed it for writing that ValveSeat's order date is the snapshot
  date, which it is (2025-04-05); the rewrite changed nothing material.

**Reading the second round:**
- The newer models again got every figure after the lead-time change right in every first answer,
  now also for the two items whose risk changes (ThrustWasher Critical with a 2-week gap, SpringClip
  High). GPT-4o did not, in either run, and twice proposed a lead-time change for a fixed one-off
  delay.
- With the judge's errors set aside, all three newer models passed every first answer on substance.
  Sonnet's failure was plainly wrong (the snapshot date is 2025-04-05). Luna's J6b gave two reasons:
  the quantity one was plainly wrong; the other, that the supplier expects the longer lead times to
  end with 2025 while OilSeal's delivery falls in March 2026, is a matter of judgement, not an error
  in the first answer. So on these cases the three are equally accurate, and they differ in cost,
  time and depth: Luna about 1/10-1/25 of the others' cost, Sol the most thorough text, Sonnet the
  most expensive and of the judge's family.
- `gpt-6.1-sol` is the slowest per call (23-39 s) but its totals (38-57 s) are close to the others'.
- **The judge is now the larger source of error.** Of the five rewrites in this round, two were
  sent back by judge errors (Luna J6b, Sonnet J7a), one of which made the analysis worse; and the
  judge let GPT-4o's wrong lead-time change through, contradicting its own verdict on the same output.
  Judge validation (`docs/judge_validation.md`) has to come before the judge's verdicts are relied on,
  and a rewrite that follows a judge's note can do harm when the note is wrong.

## Decision

**`gpt-6.1-sol`, decided 2026-10-24** (`AGENT_MODELS["analysis"]` in `services/agents.py`).

With the judge's errors set aside, Sol, Luna and Sonnet were equally accurate on both cases: the same
lead times and quantities, every figure after the changes right at first. What decided it was what
the analysis is for. The value of the LLM in this pipeline is what it draws out of the history: the
questions to put to the supplier and to settle internally, and the actions to take. On that, Sol
was the most specific in every run, and Luna the least:

| | gpt-6.1-sol | gpt-6-luna | claude-sonnet-5-5 |
|---|---|---|---|
| Questions, Kestrel | 2-3, e.g. what expedited shipping, earlier production slots or split deliveries the supplier can offer for BearingCap and ThrustWasher, with quantities and dates; how the relocation affects each of the seven items' dates; whether the GasketKit forecast already includes the contract's demand (that question, or Nordic's equivalent for WasherSteel, in three of the four runs) | 1: confirm the lead time and delivery dates for this order | 5: whether the extension applies to every item and per-item dates; whether it will last to the end of 2025 or get worse; shorter lead time, partial early delivery or expediting for BearingCap and ThrustWasher; internally, an alternative source or reallocated stock for the gap; whether the suggested GasketKit quantity already reflects the 50% rise (asked in all four runs, Nordic's for WasherSteel) |
| Actions | 4-6, with dates and deadlines, e.g. "Place the ValveSeat order on 5 April 2025 and obtain a delivery commitment for no later than 6 September 2025; the revised lead time leaves no timing buffer"; the remaining orders brought forward to their new dates, item by item; in run 1 of each case a date by which bridging supply must arrive ("arriving by 28 June 2025", "by 2025-06-07") | 4, general, e.g. "Confirm the revised lead time and delivery dates with KestrelComponents before placing the orders" | 4-6, with the order dates ("Order SpringClip by 2025-05-17, PistonRing by 2025-05-31 and OilSeal by 2025-10-18, and re-check these dates if the supplier changes its lead times"), no date by which bridging supply must arrive; in Nordic run 1 an action states the suggested quantity "does not reflect the demand rise" while a question asks whether it does |
| Analysis cost per run | $0.021-0.033 | $0.0015-0.0042 | $0.041-0.109 |
| Analysis call | 23-39 s | 21-36 s | 22-31 s |

Sonnet asked the broadest questions (more on the supplier's risk and on internal cover) and Sol
gave the actions the firmest deadlines; Sonnet costs 1.5-3.5 times Sol and is of the judge's family
(`claude-haiku-5-5`), whose verdicts on it may be biased. Google's Gemini Flash, the other mid-tier candidate, failed on availability before quality could be
compared (three of four runs refused for demand; see "How the candidates were chosen"); in its one
completed run it cost as much as Sol with less specific questions and actions. So Sol costs about ten times Luna, and about two thirds of GPT-4o (whose rewrites made it dearer
still); the time is about the same as the others'. Luna stays the fallback where cost matters more.

### Sol on the prompt as it now stands: eight runs, 2026-10-24

After the stock-findings rule was tightened (only facts a question or action rests on), Sol ran four
times on each case (`dev/runs/2026-10-24_020443_…`, `_020540_…` and the six labelled "sol repeat").
Cost is the whole pipeline for one supplier: analysis, judge (analysis and purchase request checks),
the purchase request (`gpt-4o`) and the email (`gpt-5.4-mini`, validator `gpt-4o-mini`).

| Case | Total time | Analysis call | Rewrites | Cost per supplier (analysis / judge / purchase request / email) | Verdict | Stock findings, all cited | Questions / actions | Question on demand already in the suggested quantity |
|---|---|---|---|---|---|---|---|---|
| Kestrel (7 items, 3 documents) | 51.6-55.2 s; 82.1 s with a rewrite | 33.4-35.7 s | 1 of 4 runs | $0.040-0.041 ($0.068 with the rewrite); about 70% analysis | ready, no warnings, 4 of 4 | 3, yes | 2-3 / 3-5 | 3 of 4 runs |
| Nordic (4 items, 4 documents) | 39.8-43.9 s | 22.5-26.1 s | 0 | $0.029-0.031 | ready, no warnings, 4 of 4 | 1-2, yes | 4 / 4 | 3 of 4 runs |

The one rewrite: J8 failed a supplier question asking what partial-delivery and expedited-shipping
options the supplier could offer for BearingCap and ThrustWasher, "with quantities and arrival
dates", as revealing that stock is short and the order urgent. Asking a supplier to expedite says
that much by itself; whether such a question may go to the supplier is a rule for J8 to settle.

Still to do with Sol:
- Its stock findings cover every item, including those with nothing to report ("OilSeal … remains
  Low risk with no projected gap"), repeating the Items table; Findings are meant to hold the facts
  that matter.
- It writes dates as "5 April 2025" in the text while the tables use 2025-04-05.
- Its slowest call (38.9 s, seven items) is within the 60 s `llm_timeout_seconds`, with less margin
  than GPT-4o had. Proposed: `gpt-6-luna` first, `gpt-6.1-sol` where more depth is worth the time, after
repeated runs of both on these cases and on a case with an item whose risk changes with the lead
time. The full analysis-model benchmark (Still to do, `docs/changelog_2026-10.md`) follows the
staged comparison of `docs/eval_rubric.md`: blocking checks first, then warnings, then cost and
time.

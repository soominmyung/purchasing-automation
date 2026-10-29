# Choosing the analysis model

**2026-10-24.** How the analysis agent's model and effort were chosen.

## 1. The test cases

Two development cases (`dev/cases/`), each one supplier with its stock position and its history
documents, written so that the right answer is known. Both have the snapshot date 2025-04-05 and the
pipeline's standard lead time of 18 weeks (16 import + 2 internal). Risk follows from weeks to
stockout and the lead time: Critical below the lead time, High below lead time + 8, Medium below
lead time + 17, Low otherwise.

### Kestrel: changes clearly warranted (seven items)

KestrelComponents. The history gives a reason to change the lead time for every item and the
quantity of one, so it tests whether the model makes the changes and works out what follows from
them.

| Item | Available stock | Weeks to stockout | At 18 weeks: risk, gap, latest order date | At 22 weeks: risk, gap, latest order date |
|---|---:|---:|---|---|
| BearingCap | 600 | 12 | Critical, 6, 2025-04-05 (overdue) | Critical, **10**, 2025-04-05 (overdue) |
| ThrustWasher | 1,800 | 20 | High, 0, 2025-04-19 | **Critical, 2**, 2025-04-05 (overdue) |
| ValveSeat | 2,200 | 22 | High, 0, 2025-05-03 | High, 0, **2025-04-05 (due on the snapshot date)** |
| GasketKit | 4,800 | 24 | High, 0, 2025-05-17 | High, 0, 2025-04-19 |
| SpringClip | 2,800 | 28 | Medium, 0, 2025-06-14 | **High**, 0, 2025-05-17 |
| PistonRing | 1,500 | 30 | Medium, 0, 2025-06-28 | Medium, 0, 2025-05-31 |
| OilSeal | 3,500 | 50 | Low, 0, 2025-11-15 | Low, 0, 2025-10-18 |

History documents:
1. KestrelComponents, all items (2025-03-10): from 1 April 2025 lead times for all products are four
   weeks longer while production moves to a new plant in Gdansk; the supplier expects this to last
   until the end of 2025.
2. GasketKit, any supplier (2025-02-20): demand 50% higher since January 2025, after a fleet operator
   signed a contract for monthly deliveries running until December 2027.
3. OilSeal, any supplier (2024-12-15): demand steady over the past year.

The right answer: every item's lead time 18 → 22 weeks; GasketKit's quantity 5,200 → 7,800 (×1.5);
the figures at 22 weeks as in the right-hand column, including the two items whose risk rises
(ThrustWasher becomes Critical with a 2-week gap, SpringClip High), which were added so that a
figure worked out at 18 weeks cannot pass unseen; an action bridging each Critical item's gap;
ValveSeat to be ordered now.

### Nordic: no lead-time change warranted (four items)

NordicFasteners. Each kind of history document once, and a delay that should not change the lead
time, so it tests restraint and coverage.

| Item | Available stock | Weeks to stockout | Risk, gap, latest order date (18 weeks) |
|---|---:|---:|---|
| HexBoltM12 | 900 | 9 | Critical, 9, 2025-04-05 (overdue) |
| WasherSteel | 2,000 | 20 | High, 0, 2025-04-19 |
| AnchorPlug | 3,000 | 30 | Medium, 0, 2025-06-28 |
| CableTie | 4,400 | 44 | Low, 0, 2025-10-04 |

History documents:
1. NordicFasteners, all items (2024-11-20): shipments held for four weeks in November 2024 after a
   customs documentation error; the supplier says it has since changed how its export paperwork is
   prepared.
2. WasherSteel, any supplier (2025-02-10): demand up about 60% since January 2025 (a new construction
   customer's monthly orders).
3. AnchorPlug, from this supplier (2025-01-08): a December 2024 batch had about 3% of units with
   cracked sleeves; replacements arrived two weeks later.
4. CableTie, any supplier (2024-12-01): demand steady over the past year.

The right answer: no lead-time change (a one-off hold the supplier says it has fixed);
WasherSteel's quantity 2,600 → 4,160 (×1.6); a question to the supplier on whether the paperwork
change has held, and one on the AnchorPlug defect; an action bridging HexBoltM12's 9-week gap.

### How they were run

The same prompt (`prompts/v5/analysis.md`) for every model; only the analysis model and its effort
change (`scripts/run_dev_case.py --analysis-model --analysis-effort`), the rest of the pipeline and
the judge as in production. Each configuration two to four times per case. Every run is recorded
with its prompts, settings, models, efforts, tokens and time (`scripts/run_dev_case.py`).

## 2. What was compared

- **How the candidates were chosen:** no dearer per token than GPT-4o; the current generation of
  each line; not the judge's model; one per price tier and provider.
- **Candidates:** `gpt-4o`, `gpt-6.1-sol`, `gpt-6-luna`, `claude-sonnet-5-5`,
  Gemini Flash.
- **Effort:** each model's default first (second round: medium for Sol and Luna, high for Sonnet),
  then another level (third round: Luna high, Sonnet medium).

## 3. Results by configuration

| Model · effort | First answers passing (judge errors set aside) | Figures after the change right (C10) | Rewrites | Analysis call | Total | Analysis cost per run | Availability |
|---|---|---|---|---|---|---|---|
| `gpt-4o` | **0 of 4** | no (Kestrel: 8 and 7 of 20 wrong) | 5 | 8-14 s | 39-63 s | $0.039-0.064 | normal |
| **`gpt-6.1-sol` · medium** | **4 of 4** | yes | 0 | 23-39 s | 38-57 s | **$0.021-0.033** | normal |
| `gpt-6-luna` · medium | 4 of 4 | yes | 1 (a judge error) | 21-36 s | 37-84 s | $0.0015-0.0042 | normal |
| `gpt-6-luna` · high | 3 of 3 that finished | yes | 1 (a judge error) | 37-94 s | 52-110 s | $0.0022-0.0042 | **3 timeouts** |
| `claude-sonnet-5-5` · high | 4 of 4 | yes | 1 (a judge error) | 20-31 s | 36-71 s | $0.039-0.109 | normal |
| **`claude-sonnet-5-5` · medium** | **4 of 4** | yes | 0 | **10-22 s** | **24-38 s** | $0.026-0.046 | normal |
| Gemini Flash | 1 of 1 that finished | yes | 0 | 34 s | — | $0.031 | **4 of 6 calls refused** |

- **GPT-4o** got the figures wrong and turned a fixed customs hold into a lead-time change: out.
- **Gemini Flash** out on availability.
- **Luna at high** only got slower, and no more specific: out.
- **Left:** Sol at medium and Sonnet at medium, and, on cost alone, Luna at medium.

## 4. The deciding criterion: how specific the questions and actions are

What the analysis is for is what it draws out of the history: the questions to put to the supplier
and to settle internally, and the actions to take.

| | Sol · medium | Sonnet · medium | Luna · medium |
|---|---|---|---|
| Kestrel questions | 2-3 | **3-4** | 1 |
| Kestrel actions | 4-6 | 5-6 | 4 |
| A date by which bridging supply must arrive | **yes** ("arriving by 28 June 2025") | rarely | no |
| Order dates | each item's new date | each item's new date ("by 19 April 2025") | no |
| Whether the suggestion already includes the demand rise | most runs | all four runs | rarely |
| Other points raised | ValveSeat due today | ValveSeat due today; whether the extension ends with 2025; an alternative source or internal reallocation | — |

- **Sol** writes the most actionable actions: what to secure, by when, as dates.
- **Sonnet** asks the broadest questions, into the supplier's risk and internal alternatives.
- **Luna** is accurate but clearly less specific, short of what the analysis is for.

## 5. Cost and speed (same prompt, one supplier)

| | Sol · medium | Sonnet · medium |
|---|---|---|
| Analysis call | Kestrel 33-36 s, Nordic 23-26 s | Kestrel 22 s, Nordic 10-11 s |
| Analysis cost per run | Kestrel $0.028, Nordic $0.021 | Kestrel $0.046, Nordic $0.026 |
| Per 1,000 analyses | about $25 | about $36 |

Sonnet costs about 1.3-1.6 times as much and is 1.5-2 times as fast.

## 6. What else bears on the choice

- **The judge's family matters most.** The production judge, `claude-haiku-5-5`
  (`docs/judge_selection.md`), is a Claude model.
  - With Sonnet, the analysis and the judge would be of one family, the case the self-preference
    literature warns about (Zheng et al., NeurIPS 2023; Panickssery, Bowman and Feng, NeurIPS 2024;
    Wataoka, Takahashi and Ri, 2024). If the judge passes Sonnet's output too readily, the blocking
    checks mean less.
  - With Sol (OpenAI), the analysis and the judge are of different families, and the problem does
    not arise.
- **Limits:**
  - Two cases, two to four runs per configuration.
  - The judge used in these comparisons was Haiku without thinking, the weaker setting found in
    judge validation. The figure accuracy (C10) is checked by code and is unaffected.

## 7. What each round showed

- **Round 1 (one run per model and case): changing the model fixed what prompt revisions had not.**
  GPT-4o got 8 of Kestrel's 20 figures after the lead-time change wrong at first (order dates copied
  from the 18-week input, risk levels a level too low) and turned Nordic's customs hold, which the
  supplier says is fixed, into a 22-week lead time. A probe showed why: asked "2025-09-06 minus 22 weeks", it was right 6 times in 9 with
  its working written out, and in the analysis it must write the date without working. The three
  newer models had every figure right at first, kept Nordic at 18 weeks and raised the defective
  AnchorPlug batch, which GPT-4o left out.
- **Round 2 (two runs each, two items added whose risk rises at 22 weeks): equally accurate, so the
  difference is what they draw out of the history.** Sol, Luna and Sonnet got every figure right,
  including ThrustWasher becoming Critical with a 2-week gap; GPT-4o did not, in either run. The
  failures left were the judge's: it failed Luna's correct 50% quantity rise with a wrong reason, and
  Luna's rewrite followed it to a wrong 5,400; it passed GPT-4o's wrong lead time, then failed and
  passed the same output. This made judge validation the next step.
- **Round 3 (other efforts): more reasoning did not make Luna more specific.** At high effort Luna's
  actions were as general as at medium ("Confirm order-specific delivery dates with KestrelComponents
  in light of the announced lead-time extension") and three calls passed the 60-second limit. Sonnet
  at medium was the fastest and as accurate, with the broadest questions.
- **The deciding difference, read side by side (Kestrel, same prompt):**
  - Sol: "Place the ValveSeat order on 5 April 2025 and obtain a delivery commitment for no later than
    6 September 2025; the revised lead time leaves no timing buffer"; bridging supply "arriving by
    28 June 2025".
  - Sonnet: "Order SpringClip by 2025-05-17, PistonRing by 2025-05-31 and OilSeal by 2025-10-18", but
    no date by which bridging supply must arrive.
  - Luna: "Confirm the revised lead time and delivery dates with KestrelComponents before placing the
    orders".
- **Later, with Sol in place (2026-10-24 and 25):** when the prompt asked when each check comes, Sol
  first put supplier-wide checks "before placing each order" next to orders it said to place
  immediately; with the rule changed to judge the timing against each item's latest order date, it
  split the same check by item
  ("alongside the immediate HexBoltM12 order, without delaying it. Complete these checks before
  placing the remaining orders"), overdue items with a check put first went 9 to 0 in 25 analyses and
  dates in actions 38 to 72. Given the company's market, it told a UK demand rise from a Korean one
  ("Use the Korean demand surge to assess supply availability, not to increase the UK order
  quantity").

## 8. Decision

**`gpt-6.1-sol` at medium effort, kept.**

**Why Sol: its actions say what to secure and by when, as dates.** The value of the LLM in this
pipeline is what it draws out of the history for the buyer to act on; once the newer models were
equally accurate, that was the difference (examples above). An action with a deadline can be done
and checked; "confirm the lead time" cannot.
1. It writes the most actionable actions, with arrival deadlines and order dates as dates.
2. It costs less than Sonnet, with the same first-answer record (4 of 4).
3. It is of a different family from the production judge (Haiku), so the judge can check it fairly.

**The alternative is Sonnet at medium**, if speed matters most or broader questions are wanted. The
judge would then have to be a non-Claude model to be fair, and Haiku is the only judge that has met
the validation thresholds so far, so that is not practical now.

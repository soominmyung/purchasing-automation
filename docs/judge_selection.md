# Choosing the judge model

**2026-10-24.** The case for the LLM judge's model and settings, in one place. The full record of
every run is in `docs/judge_validation.md`; this is the summary to cite.

## 1. What the judge does

After the code checks (C1–C10), one LLM call reads each supplier's analysis against its input and
history documents and gives a pass or fail, with the evidence, on every applicable judge check
(J1–J10: facts traced to a document, changes and their reasons, questions, actions, summary). A
second call checks the purchase request's Purpose and Justification (P1a–c: nothing added, no
judging words, no overstatement). A failed analysis check sends the analysis back for a rewrite; a
failed purchase request check is a warning, shown in the draft's red note for the buyer.

## 2. Candidates and why

- **Not of the analysis model's family.** The analysis is written by `gpt-6.1-sol`. LLM judges favour
  text from their own model or family: Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and
  Chatbot Arena" (NeurIPS 2023 Datasets and Benchmarks, arXiv:2306.05685); Panickssery, Bowman and
  Feng, "LLM Evaluators Recognize and Favor Their Own Generations" (NeurIPS 2024, arXiv:2404.13076);
  Wataoka, Takahashi and Ri, "Self-Preference Bias in LLM-as-a-Judge" (arXiv:2410.21819, preprint).
- **Cheap, since it runs on every attempt of every supplier.** `claude-haiku-5-5` and
  `gemini-3.1-flash-lite`; `gpt-6-luna` as a reference from the analysis model's family. Sonnet and
  Gemini Flash were left out on cost (and Gemini's availability: 429/503/504 errors in testing).

## 3. The tests

1. **Seeded faults** (`eval_data/judge_calibration_v5.jsonl`): 51 samples, each a real output with
   one fault planted for one check (46) or a real fault (5). Detection, and flip (the verdict
   changing across three runs).
2. **False alarms**: six unmodified outputs, one of each kind of scenario, scored by hand on every
   check by the user (89 verdicts, `eval_data/hand_scoring_sheet_v5.md`). A judge's fail where the
   person passed is a false alarm.

Rule set before the runs: detection ≥ 90%, false alarms ≤ 10%, flip ≤ 5%; then the cheapest.

## 4. Results

| Judge | Detection | Flip (seeded) | False alarms | Cost per judgement | Time |
|---|---:|---:|---:|---:|---:|
| Haiku, forced tool (no thinking) | 83.7% | 17.6% | 2.7% | $0.0018 | 6 s |
| `gpt-6-luna` | 89.1% | — | 17.0% | $0.0020 | 34 s |
| Haiku, thinking, medium | 91.2% | 5.9% | 9.5% | $0.0018 | 12 s |
| **Haiku, thinking, high** | **93.9%** | **3.9%** | **8.0%** | **$0.0024** | **16 s** |

Gemini Flash-Lite did not finish a round reliably. Round 1 called Haiku through a forced tool call,
which leaves it no room to think while Luna reasoned; the user caught that this was unfair, and
round 2 let Haiku think first and answer in the scoring sheet's shape (structured outputs).

**One wording fixed on the way (P1a).** In the hand-scored set, the only fail was a purchase request
adding "supporting the decision to proceed with this supplier". No judge caught it. P1a now says
"Stating a conclusion that the input does not draw ... adds an assessment"; a broader first wording
also failed a plain summary of a document. With the kept wording: caught 3/3, the summary passed
3/3, seeded P1 faults 23/24.

## 5. What each round showed

- **Why validate at all:** in the analysis model comparison the judge had become the larger source of
  error. It failed a correct 50% quantity rise with a wrong reason and the rewrite followed it to a
  wrong figure; it passed a wrong lead time and then, on the same output, failed and passed it.
- **Round 1: the model was not the problem; the way it was asked was.** Haiku caught 83.7% of the
  seeded faults and its verdicts flipped on 17.6% of samples; Luna caught 89.1%. The user saw that the
  comparison was unfair: the judge's answer was forced through a tool call, and with a forced tool
  Haiku returns only a `tool_use` block, no thinking (1,637 output tokens against 3,346 with thinking
  on the same sample), while Luna reasoned. Gemini Flash-Lite answered too seldom to judge (429
  errors).
- **Round 2: let it think.** The same Haiku, answering in the scoring sheet's shape so that it thinks
  first: 91.2% at medium effort, **93.9% at high, flip 3.9%**. The checks it had missed most improved
  most: J6a (direction of a change) 5/9 to 8/9, J6b (size) 3/9 to 6/9, J10 (a problem document left
  unused) 1/6 to 4/6, P1c (meaning kept) 6/9 to 9/9. GPT-4o's wrong lead time for a fixed customs
  hold, passed every time in round 1, was now caught two times in three.
- **False alarms, on 89 verdicts the user scored by hand:** Haiku high 8.0%, most of them stricter
  readings than a person's ("only 5 weeks to stockout" as a judging word); Haiku without thinking
  2.7% but it catches less; Luna 17.0% with 19.1% flips. No setting caught the one real fault (an
  added assessment of the supplier) until P1a was reworded to "Stating a conclusion that the input
  does not draw"; a broader rewording also failed a plain summary, so the narrow one was kept.
- **What this taught:** a judge's verdicts depend on how it is made to answer as much as on the
  model; and with warnings that the buyer settles, a judge that flags a little more than a person is
  better than one that misses what a person would catch (recall over precision).

## 6. Decision

**`claude-haiku-5-5`, thinking at high effort, answer in the scoring sheet's shape, one call per
output** (`anthropic:claude-haiku-5-5?effort=high&format=schema`, `evaluator_model` in `config.py`).

**Why Haiku with thinking: it catches the most faults with the fewest flips.** 93.9% of the seeded
faults (against 83.7% for the same model forced to answer without thinking and 89.1% for Luna), flip
3.9%, false alarms 8.0%: the only setting to meet all three thresholds. It is also of a different
family from the analysis model, and costs about $0.0035 per supplier with both calls.

**Favouring recall over precision.** Most remaining false alarms are stricter readings than a
person's ("only 5 weeks to stockout" read as a judging word, "imminent" for 19 weeks out). They are
left as they are: a missed fault reaches the approvers unnoticed, while a false alarm costs the buyer
one look at a line in the red note. With error costs that unequal, a judge that flags somewhat more
than a person is preferred to one that misses what a person would catch.

Still weak: J2a, J6b and J10 on seeded faults (two or three misses each).

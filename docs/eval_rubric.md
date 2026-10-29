# Quality checks

Every draft the pipeline writes is checked before the buyer sees it: the analysis, the purchase
request and the supplier email. Code checks what can be computed (format, items, figures,
citations); an LLM judge checks what needs reading (whether the history is reported faithfully,
whether a change is justified, whether a question leaks internal information). The definitions
are in `services/quality_checks.py`; the email validator is `check_email` in `services/agents.py`.

The same checks were used to choose the models (`docs/*_selection.md`).

## Severity and verdict

- **Blocking:** the draft must not be used as it is if this fails (invented or distorted history,
  a wrong figure, an item not in the input, internal information in a question for the supplier).
- **Warning:** usable, but a person should look.

A draft is **ready** only if every blocking check that applies passes; otherwise it **needs
review**. Warnings are listed but do not change the verdict. A check the judge could not answer
counts as not passed. Each check applies only when there is something to check (no proposed
change, no J3; no question for the supplier, no J5).

What happens on a failure:

| Draft | Blocking failure | Warning |
|---|---|---|
| Analysis | Sent back to the analysis agent with the reasons, up to two rewrites; if it still fails, the attempt with the fewest blocking failures is used and re-checked | Shown in the note at the top of the report and the purchase request |
| Purchase request | Shown in the note at the top of the draft (no rewrite) | Same |
| Supplier email | Sent back with the reason, up to two rewrites; a reason left after the last one is shown in the email's note | Shown in the email's note |

Every result, with its reason, also goes into the quality-check document kept for the operator.

## Analysis

The analysis agent returns a summary, numbered findings (facts from the history documents or the
stock figures, each with its source) and judgements that cite them: questions, proposed lead-time
and quantity changes, and recommended actions. Code works out the figures after a change.

### Code checks (C1–C10)

| ID | Passes when | Severity |
|---|---|---|
| C1 | The output parses and has every field with the right type | Blocking |
| C2 | Every item code it names is in the input | Blocking |
| C3 | Each quantity change starts from the pipeline's suggested quantity | Blocking |
| C4 | Every finding rests on documents that were given and apply to what it is about, or on the stock position; every question, change and action cites at least one finding that exists and concerns its items or the supplier | Blocking |
| C5 | Each change is well formed and adds up: a lead time above 0 weeks, each item at most once, a quantity factor above 0 and not 1, the new quantity equal to the old one times the factor, rounded up | Blocking |
| C6 | Each Critical item has an action that cites a stock finding about it | Blocking |
| C7 | The summary names every Critical item | Warning |
| C8 | The summary is within about 100 words | Warning |
| C9 | Every "current value → proposed value" written in the text is one of the proposed changes | Blocking |
| C10 | The figures after each lead-time change (risk level, gap, latest order date) are those the pipeline's rules give | Blocking |

### Judge checks (J1a–J8)

| ID | Passes when | Applies when | Severity |
|---|---|---|---|
| J1a | Every statement about supplier or item history is found in the documents given | Always | Blocking |
| J1b | No statement distorts what a document says (nothing made stronger, weaker or different) | Always | Blocking |
| J1c | No document's content is attributed to a supplier or item it does not name | Always | Blocking |
| J2a | At least one question addresses the supplier's problem in the history | A document concerns the supplier | Warning |
| J2b | At least one question addresses an item's problem in the history | A document names an item | Warning |
| J2c | No question presumes a problem the inputs do not show | Always | Warning |
| J3a | Each proposed change goes the way its cited findings point | A change is proposed | Blocking |
| J3b | Each change's size fits what the findings say; where they give no size, the reason says the size is an estimate to be confirmed | A change is proposed | Blocking |
| J3c | No item that a change's reason applies to is left without the change, unless the findings support it | A change is proposed | Blocking |
| J4 | Every stock figure quoted in the text matches the item's figures after the proposed changes | Always | Blocking |
| J5 | No question for the supplier reveals anything internal (stock levels, weeks to stockout, budget, the company's own analysis); asking what faster or partial delivery the supplier can offer passes | A question is marked for the supplier | Blocking |
| J6 | Each question and action follows from the findings it cites | Questions or actions exist | Warning |
| J7 | No document left unused describes a problem that matters for this order | A document is unused | Warning |
| J8 | Findings state facts only, with no judgement of what they mean for the order | Findings exist | Warning |

## Purchase request

The approval agent writes the Purpose and Justification; code writes the item table, the
timing and the signature block.

| ID | Passes when | By | Severity |
|---|---|---|---|
| P1a | Nothing is added that the input does not give: no new fact, risk or assessment (for example, saying a supplier's record supports buying from it) | Judge | Warning |
| P1b | Figures are quoted as the input gives them, with no word that judges them | Judge | Warning |
| P1c | Where other words are used, the meaning does not change ("about 60%" is not "60%") | Judge | Warning |
| P2 | Purpose has 1–2 sentences, Justification 2–4 | Code | Warning |
| P3 | Only the history documents given are cited | Code | Blocking |

## Supplier email

The email agent writes only the requests; code adds the greeting, the purpose, the item lines and
the signature. The agent never receives stock figures, risk levels or internal questions.

| ID | Passes when | By | Severity |
|---|---|---|---|
| E1 | Nothing internal is revealed: no internal terms, stock quantities, weeks to stockout, risk levels or mention of an internal analysis | Code, then `claude-haiku-5-5` | Blocking |
| E2 | No past problem of the supplier, or when it happened, is mentioned; a question about how a step of this order will proceed or be checked passes | `claude-haiku-5-5` | Blocking |
| E3 | No item outside this order is mentioned, by name, code or other words | Code, then `claude-haiku-5-5` | Blocking |
| E4 | Every point to clarify is covered by a question | Judge | Warning |
| E5 | No question is asked twice | Judge | Warning |
| E6 | For each Critical item, the earliest dispatch or a faster delivery is asked | Judge | Warning |
| E7 | No purchase order is promised, the item list is not repeated, and it reads as a polite request to a business partner | Judge | Warning |

## The judge

`claude-haiku-5-5`, thinking at high effort, answering in the shape of a scoring sheet, one call
per draft (`evaluator_model` in `config.py`). It is from a different model family from the
writers (`gpt-6.1-sol`), so no draft is judged by a model of its own family. Before use it was
validated on outputs with known faults and on hand-scored outputs (`docs/judge_selection.md`).

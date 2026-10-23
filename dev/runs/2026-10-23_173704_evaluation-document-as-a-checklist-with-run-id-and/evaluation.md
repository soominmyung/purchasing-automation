# Analysis Quality Check

**Supplier:** KestrelComponents  
**Run id:** 2026-10-23_173704_evaluation-document-as-a-checklist-with-run-id-and  
**Snapshot date:** 2025-04-05  
**Result:** Ready: all 16 blocking checks passed; no warnings.  
**Checked by:** code, and `anthropic:claude-haiku-5-5` for the judgement checks

## Attempts

| Attempt | Judge run | Blocking checks failed | Sent back for a rewrite |
| --- | --- | --- | --- |
| 1 | no (a code check failed first) | C5 (Proposed changes add up): 520001 proposed order date 2025-03-08 is before the snapshot date | yes |
| 2 | yes | none | no |

## Analysis Checks

| Check | Severity | Result | Detail |
| --- | --- | --- | --- |
| Output format (C1) | blocking | Pass | all required fields present |
| Items match the input (C2) | blocking | Pass | every item named is in the input |
| Figures match the input (C3) | blocking | Pass | changes start from the pipeline's figures |
| Every judgement cites findings that apply (C4) | blocking | Pass | every judgement cites findings that apply |
| Proposed changes add up (C5) | blocking | Pass | the changes add up |
| Each Critical item has an action on its stock position (C6) | blocking | Pass | each Critical item has one |
| Summary names every Critical item (C7) | warning | Pass | names every Critical item |
| Summary within about 100 words (C8) | warning | Pass | 52 words |
| Arrows in the text match the proposed changes (C9) | blocking | Pass | every arrow matches a change |
| History statements found in the documents (J1a) | blocking | Pass | Findings restate documents [1] (four-week longer lead times from 1 April 2025, until end 2025), [2] (GasketKit demand 50% higher since January 2025, contract to December 2027), and [3] (OilSeal demand steady). No invented history. |
| History not distorted (J1b) | blocking | Pass | Finding 1 matches document 1 closely; finding 3 matches document 2 (50% higher, contract to Dec 2027); finding 4 matches document 3 ('steady over the past year'). Nothing strengthened or weakened. |
| History not attributed to what it does not name (J1c) | blocking | Pass | Document 1 applies to supplier-wide and is cited as supplier-wide in finding 1 with items []. Document 2 is cited for GasketKit (520003) only. Document 3 is cited for OilSeal (520005) only. |
| No question presumes a problem the data does not show (J2c) | warning | Pass | Critical questions: 'How will the extended lead times ... affect the delivery of ...' (neutral) and 'Is the increased GasketKit demand expected to continue beyond the current contract?' (the demand increase is shown by document 2). Neither presumes an unshown problem. |
| Stock figures quoted match the input (J7a) | blocking | Pass | Summary says 'a 6-week gap' for BearingCap (expected_stockout_weeks 6, matches). Finding 2 says 'gap is 6 weeks' (matches). No other stock figures are quoted in prose. |
| Each proposed change goes the way its findings point (J6a) | blocking | Pass | Finding 1 (lead times +4 weeks) supports moving PO dates earlier. Shifts of -4 weeks were applied to 520002-520005 and 0 to 520001 (already overdue at 2025-04-05 per system). Finding 3 (GasketKit demand +50%) supports the 1.5 factor increase for 520003. |
| Each proposed change is the size its findings support (J6b) | blocking | Pass | Shift of -4 weeks matches the 4-week lead time increase in finding 1. Factor 1.5 matches the 50% demand increase in finding 3. The reasons state these sizes. Check 520002: 2025-05-03 minus 4 weeks = 2025-04-05, matches. 520003: 2025-05-17 minus 4 weeks = 2025-04-19, matches. 520004: 2025-06-28 minus 4 weeks = 2025-05-31, matches. 520005: 2025-11-15 minus 4 weeks = 2025-10-18, matches. |
| No item the same reason applies to is left out (J6c) | blocking | Pass | Lead-time reason applies to all products, and all five items have po_date_changes entries. 520001 has shift 0 because its date (2025-04-05) is already at or before the reasonable window; it is not moved earlier beyond what is possible, consistent with the exception. The demand reason applies only to GasketKit. |
| Changed figures written as pipeline value and proposed value (J7b) | blocking | Pass | Changed figures are mentioned in prose only as 'the gap is 6 weeks' and 'demand increased by 50%'. No PO date or quantity with a changed value appears in prose, e.g. the 5200 quantity is only referenced in the change field. Summary and actions do not state the order date or quantity. |
| Questions marked for the supplier reveal nothing internal (J8) | blocking | Pass | The supplier question (critical_questions[0]) asks how extended lead times affect delivery of listed items. It does not reveal stock levels, weeks to stockout, urgency, budget or internal analysis. The summary's stockout wording is internal but is not a supplier question. |
| Each question and action follows from its findings (J9) | warning | Pass | Supplier question about lead-time impact follows from finding 1. Internal GasketKit question follows from finding 3. Action 'expedite BearingCap or cover from alternative source' follows from finding 2 (6-week gap). Action to confirm delivery dates follows from finding 1. Action to review GasketKit quantity follows from finding 3. |
| No document with a problem for this order left unused (J10) | warning | Pass | Document [3] (OilSeal demand steady) is cited only in finding 4, which is a plain statement, and no question, change or action cites finding 4 or document 3. |
| Findings state facts only (J11) | warning | Pass | Finding texts restate document facts or stock figures ('gap is 6 weeks'). Finding 2 states 'BearingCap runs out before a new order can arrive', which is a stock-based fact drawn from the expected stockout and lead time. No judgement of order implications appears in the findings. |
| Supplier problem raised as a question (J2a) | warning | Pass | Document 1 describes a supplier-wide lead-time extension that matters for ordering. Critical question 1 addresses it by asking the supplier how the extended lead times affect deliveries. |
| Item problem raised as a question (J2b) | warning | Pass | Document 2 describes GasketKit demand rising 50% (a problem for ordering). Critical question 2 asks whether the increased GasketKit demand continues beyond the contract. Document 3 describes steady OilSeal demand, which is not a problem. |

## Purchase Request Text

**Result:** Ready: all 1 blocking checks passed; 1 warning.

| Check | Severity | Result | Detail |
| --- | --- | --- | --- |
| Purchase request adds nothing to its input (P1a) | warning | Fail | Text says BearingCap is critical with a six-week stockout expected, matching input risk_level Critical and gap of 6 weeks. The 'confirming delivery dates... is essential' statement is a condition the request rests on, which is allowed. However, the text adds 'imminent stockouts' in purpose; input gives weeks_to_stockout 12 for BearingCap and 22-30 for others, so 'imminent' is an added assessment. Also 'to address imminent stockouts' is not supported for ValveSeat, GasketKit, PistonRing (expected_stockout_weeks 0.0). The justification also says the order dates are adjusted, which matches the changes list. |
| Purchase request quotes figures as given (P1b) | warning | Pass | Text quotes 'four-week increase in lead times, effective from 1 April 2025', 'six-week stockout', '50% rise in demand', quantities 5200 and 7800. No judging words like quick or slow are applied to figures. Figures are exact. |
| Purchase request keeps the input's meaning (P1c) | warning | Pass | '5200' and '7800' written without commas versus input 5200 and 7800; 'six-week' for 6 weeks and 'four-week' for four weeks are same meaning. 'New plant in Gdansk' matches input. 'Alternative supplier' matches 'alternative source'. 'Expedited shipping' matches. 'Rise in demand' matches 50% higher. No meaning change. |
| Purchase request paragraphs within length (P2) | warning | Pass | within length |
| Purchase request cites only documents given (P3) | blocking | Pass | all cited documents given |

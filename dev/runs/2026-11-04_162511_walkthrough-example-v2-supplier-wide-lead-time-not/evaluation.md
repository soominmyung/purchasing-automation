# Analysis Quality Check

**Supplier:** MeridianSeals  
**Run id:** 2026-11-04_162511_walkthrough-example-v2-supplier-wide-lead-time-not  
**Snapshot date:** 2025-04-05  
**Result:** Ready: all 16 blocking checks passed; no warnings.  
**Checked by:** code, and `anthropic:claude-haiku-5-5?effort=high&format=schema` for the judgement checks

## Attempts

| Attempt | Judge run | Blocking checks failed | Sent back for a rewrite |
| --- | --- | --- | --- |
| 1 | yes | none | no |

## Analysis Checks

| Check | Severity | Result | Reason |
| --- | --- | --- | --- |
| Output format (C1) | blocking | Pass | all required fields present |
| Items match the input (C2) | blocking | Pass | every item named is in the input |
| Figures match the input (C3) | blocking | Pass | changes start from the pipeline's figures |
| Every judgement cites findings that apply (C4) | blocking | Pass | every judgement cites findings that apply |
| Proposed changes well formed and add up (C5) | blocking | Pass | the changes are well formed and add up |
| Each Critical item has an action on its stock position (C6) | blocking | Pass | each Critical item has one |
| Summary names every Critical item (C7) | warning | Pass | names every Critical item |
| Summary within about 100 words (C8) | warning | Pass | 80 words |
| Arrows in the text match the proposed changes (C9) | blocking | Pass | every arrow matches a change |
| Figures after the lead-time changes follow the rules (C10) | blocking | Pass | the figures follow the rules |
| History statements found in the documents (J1a) | blocking | Pass | Findings 1, 2 and 3 restate documents 1 (four-week lead time rise from 1 April 2025 until end of 2025), 2 (three-week November 2024 hold, revised paperwork checks) and 3 (OilRing demand steady). No other history claims appear. |
| History not distorted (J1b) | blocking | Pass | Finding 1 says lead times for all products increase by four weeks until end 2025, matching document 1. Finding 2 matches document 2's wording. Finding 3 says demand has been steady, matching document 3. |
| History not attributed to what it does not name (J1c) | blocking | Pass | Document 1 and 2 are supplier-wide and the findings attribute them to all products or to MeridianSeals. Document 3 is cited only for OilRing (610005), which it names. |
| No question presumes a problem the data does not show (J2c) | warning | Pass | Question 1 asks about expedite options for BushingSleeve and GasketRing, whose inputs show Critical gaps. Question 2 asks whether the four-week extension is still the limit, which document 1 supports. Question 3 asks about checks against a recurrence of the documented Felixstowe hold. Each addresses a shown issue. |
| Stock figures quoted match the input (J4) | blocking | Pass | Finding 4: 10 weeks, 22 weeks, gap 12, Critical, PO 5 April 2025 match post-change figures for 610001. Finding 5: 20 weeks, 22, gap 2, Critical, PO 5 April 2025 match 610002. Finding 6: 24, 22, gap 0, High, PO 19 April 2025 match 610003. Finding 7: 28, 22, gap 0, High, PO 17 May 2025 match 610004. Summary and actions: gaps 12 and 2, dates 19 April, 17 May and 18 October match post-change values. |
| Each proposed change goes the way its findings point (J3a) | blocking | Pass | supplier_lead_time and lead_time_changes set 22 weeks (18 + 4) citing finding 1, which states a four-week increase for all products. Finding 2 is cited but its hold is correctly not added because the supplier reports revised checks. |
| Each proposed change is the size its findings support (J3b) | blocking | Pass | The size of the change is exactly the four weeks stated in document 1. The reason explains that the four weeks is added to the standard 18-week lead time and that the earlier customs hold is not a continuing delay. |
| No item the same reason applies to is left out (J3c) | blocking | Pass | All five items (610001 to 610005) receive the 22-week lead time and are listed in lead_time_changes under the same refit reason, so no item is missing it. |
| Questions marked for the supplier reveal nothing internal (J5) | blocking | Pass | Supplier questions ask about expedited production, partial shipments, transport, delivery commitments, the refit limit and export-document checks. None reveals stock levels, weeks to stockout, budget or internal plans. |
| Each question and action follows from its findings (J6) | warning | Pass | Question 1 and actions 1 and 2 follow from findings 4 and 5 (Critical gaps) and the refit in finding 1. Question 2 and action 3 follow from findings 1 and 2. Action 4 brings ValveCap and SealPlate forward, following from the refit's earlier post-change PO dates in findings 6 and 7. |
| No document with a problem for this order left unused (J7) | warning | Pass | Document 3 (OilRing demand steady) is cited only by finding 3. Finding 3 is not listed in any critical question, change, or action findings array, so no action rests on it. |
| Findings state facts only (J8) | warning | Pass | Findings 4 to 7 state stock figures, risk levels, gaps and dates. Findings 1 to 3 restate documents. The 'already overdue' status of the 5 April order date matches the pipeline's overdue timing for 610001 and is a status, not a judgement about the order. |
| Supplier problem raised as a question (J2a) | warning | Pass | The history shows a supplier-wide refit extension (finding 1), addressed by question 2 and action 3, and a customs hold (finding 2), addressed by question 3. |
| Item problem raised as a question (J2b) | warning | Pass | Document 3 reports steady OilRing demand, which describes no problem with an item, so the check passes. |

## Purchase Request Text

**Result:** Ready: all 1 blocking checks passed; no warnings.

| Check | Severity | Result | Reason |
| --- | --- | --- | --- |
| Purchase request adds nothing to its input (P1a) | warning | Pass | The text restates the input's lead-time change (18 to 22 weeks), the four order-date changes, the gaps (12 and 2 weeks), the zero-gap items, the conditional commitments step, and the customs hold as not an additional delay. No new fact, risk or assessment appears. |
| Purchase request quotes figures as given (P1b) | warning | Pass | Figures match the input exactly: 1300, 2600, 10, 20, 24, 28 weeks; 12-week and two-week gaps; four-week increase; three-week hold; 18 to 22 weeks; dates 5 April, 19 April, 17 May, 14 June 2025. No judging word such as quick or slow is used. |
| Purchase request keeps the input's meaning (P1c) | warning | Pass | Reworded phrases keep the input's meaning: 'shipments were held' becomes 'shipment hold', 'before placing the remaining orders' becomes 'conditional on obtaining those commitments beforehand', 'do not defer the main order' becomes 'without waiting for parallel arrangements'. No meaning changes. |
| Purchase request paragraphs within length (P2) | warning | Pass | within length |
| Purchase request cites only documents given (P3) | blocking | Pass | all cited documents given |

## Supplier Email

**Exposure check (E1-E3, the validator):** passed (1 draft).  
**Other checks (warnings):** 0 of 4 not passed.

| Check | Severity | Result | Reason |
| --- | --- | --- | --- |
| Email covers every point to clarify (E4) | warning | Pass | Point 1 (expedited options for BushingSleeve and GasketRing) is covered by 'what expedited production, partial shipment or faster transport options can you offer'. Point 2 (refit delivery commitments and the four-week limit) is covered by 'confirm the delivery commitments you can offer for this order during the moulding-line refit, and whether any lead-time extension will be limited to four weeks'. Point 3 (export-document checks and delays) is covered by 'What export-document checks will you carry out for this order, and do you expect any documentation-related delays?'. OilRing is not asked about. |
| Email asks no question twice (E5) | warning | Pass | Each question is distinct: availability, lead times and terms; expedited options for two items; delivery dates for the two items; refit commitments; the four-week limit; export-document checks; documentation delays. No question is repeated. |
| Email asks the earliest dispatch where needed (E6) | warning | Pass | 'For BushingSleeve and GasketRing, what expedited production, partial shipment or faster transport options can you offer, and what quantities and delivery dates can you commit to?' |
| Email promises no order and reads as a request (E7) | warning | Pass | No purchase order is promised and the item list is not rewritten; items are referred to by name only. The text is written as polite flowing sentences beginning 'Could you please confirm' and 'Could you also confirm'. |

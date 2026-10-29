# Analysis Quality Check

**Supplier:** KestrelComponents  
**Run id:** 2026-10-29_004531_judge-checks-renumbered-without-gaps-j3a-c-j4-j8  
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
| Summary within about 100 words (C8) | warning | Pass | 75 words |
| Arrows in the text match the proposed changes (C9) | blocking | Pass | every arrow matches a change |
| Figures after the lead-time changes follow the rules (C10) | blocking | Pass | the figures follow the rules |
| History statements found in the documents (J1a) | blocking | Pass | Finding 1 matches document 1 (four weeks longer for all products from 1 April 2025, expected to last to end of 2025); finding 2 matches document 2 (50% higher since January 2025, fleet contract to December 2027); finding 3 matches document 3 (steady demand). |
| History not distorted (J1b) | blocking | Pass | Each finding restates its document's content without strengthening, weakening or changing it; finding 2 correctly says the document does not identify the market. |
| History not attributed to what it does not name (J1c) | blocking | Pass | Document 1 is supplier-wide and is cited with items []; document 2 is cited only for GasketKit (520003); document 3 is cited only for OilSeal (520005). |
| No question presumes a problem the data does not show (J2c) | warning | Pass | Questions address the plant-move lead-time extension (shown in document 1), the short stock cover for BearingCap and ThrustWasher (shown in the stock figures), and the GasketKit demand rise (shown in document 2); none presumes an unshown problem. |
| Stock figures quoted match the input (J4) | blocking | Pass | Finding 4: 12 weeks, lead time 22, gap 10, order 5 April 2025 match 520001 after changes. Finding 5: 20 weeks, gap 2, order 5 April 2025 match 520006. Finding 6: 22 weeks, gap 0, order 5 April 2025 match 520002. Summary risk levels (BearingCap and ThrustWasher Critical) match. Action 4 date 19 April 2025 and quantity 5200 match 520003. |
| Each proposed change goes the way its findings point (J3a) | blocking | Pass | lead_time_changes sets 22 weeks (18 plus 4) for all seven items, citing finding 1, which states the four-week supplier-wide extension. The direction and reason are supported. |
| Each proposed change is the size its findings support (J3b) | blocking | Pass | The document gives the size as four weeks, and 18 plus 4 gives 22. No item-specific delay is added, so the supplier-wide reason is not double-counted. Latest order dates fall before the end of 2025, when the extension is expected to end. |
| No item the same reason applies to is left out (J3c) | blocking | Pass | All seven items, including every item in supplier_lead_time's scope, move to 22 weeks, so no item is left at 18 weeks without a reason. |
| Questions marked for the supplier reveal nothing internal (J5) | blocking | Pass | Supplier questions ask about plant-move effects on availability and firm delivery dates, and about expedited shipping or partial deliveries for BearingCap and ThrustWasher. They reveal no stock levels, weeks to stockout, budget or internal plans. |
| Each question and action follows from its findings (J6) | warning | Pass | Question 1 cites finding 1 (plant move). Question 2 and action 2 cite findings 1, 4 and 5 (shortage cover). Action 1 cites findings 4, 5 and 6 (order dates). Action 3 cites finding 1. Question 3 and action 4 cite finding 2 (GasketKit demand, market not identified). |
| No document with a problem for this order left unused (J7) | warning | Pass | Document 3 (OilSeal steady demand) is cited only by finding 3. No question, proposed change or action cites it; OilSeal appears only under finding 1 and the supplier-wide items. |
| Findings state facts only (J8) | warning | Pass | Findings 1 to 6 state document content or stock figures (for example, 'has 12 weeks to stockout', 'leaving a 10-week gap', 'does not identify the market') without judging what they mean for the order. |
| Supplier problem raised as a question (J2a) | warning | Pass | The history describes a supplier-wide lead-time extension that matters for ordering, and question 1 addresses the plant move's effect on availability and delivery dates. |
| Item problem raised as a question (J2b) | warning | Pass | The history describes a GasketKit demand rise (document 2) that matters for ordering, and question 3 addresses it. OilSeal's steady demand (document 3) describes no problem. |

## Purchase Request Text

**Result:** Ready: all 1 blocking checks passed; no warnings.

| Check | Severity | Result | Reason |
| --- | --- | --- | --- |
| Purchase request adds nothing to its input (P1a) | warning | Pass | Every stated fact traces to the input: the four-week increase from 1 April 2025 for the Gdansk move, the 5 April, 19 April, 17 May and 31 May order dates, the 12-, 20- and 22-week stockout figures, the 50% GasketKit demand rise with no market identified, and the pre-order UK review. The 'conditional' wording on later orders matches the input's 'before placing the later orders'. |
| Purchase request quotes figures as given (P1b) | warning | Pass | Figures are quoted as given: '12 weeks to stockout', '10-week gap', '20 weeks', '2-week gap', '22 weeks', '50% higher', '5200'. No judging words are attached to any figure. |
| Purchase request keeps the input's meaning (P1c) | warning | Pass | Reworded terms keep the input's meaning: 'latest order date' for order_by, 'market is not identified' for 'does not identify the market', 'establish whether... is in the UK' as a pre-order review, and 'before placing the later orders' expressed as a condition the request rests on. |
| Purchase request paragraphs within length (P2) | warning | Pass | within length |
| Purchase request cites only documents given (P3) | blocking | Pass | all cited documents given |

## Supplier Email

**Exposure check (E1-E3, the validator):** passed (1 draft).  
**Other checks (warnings):** 0 of 4 not passed.

| Check | Severity | Result | Reason |
| --- | --- | --- | --- |
| Email covers every point to clarify (E4) | warning | Pass | Point 1 (plant move affecting availability and firm dates) is covered indirectly by "confirm the steps for securing availability and firm delivery dates for this order"; point 2 is covered by "What expedited shipping options or earlier partial deliveries can you offer for BearingCap and ThrustWasher, and on what dates?"; OilSeal is not mentioned anywhere in the requests. |
| Email asks no question twice (E5) | warning | Pass | The three questions are distinct: "confirm current availability, lead times and commercial terms ... whether you can meet the requested delivery dates?", "confirm the steps for securing availability and firm delivery dates", and "What expedited shipping options or earlier partial deliveries ... on what dates?". No question is repeated. |
| Email asks the earliest dispatch where needed (E6) | warning | Pass | "What expedited shipping options or earlier partial deliveries can you offer for BearingCap and ThrustWasher, and on what dates?" asks for sooner delivery than standard lead time for both items in ask_earliest_dispatch. |
| Email promises no order and reads as a request (E7) | warning | Pass | No purchase order is promised and the item list is not rewritten; items are referred to only as "the items listed above" and "BearingCap and ThrustWasher". The text uses polite flowing sentences such as "Could you please confirm..." and "Could you also confirm...". |

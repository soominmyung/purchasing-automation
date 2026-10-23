# Analysis Quality Check

**Supplier:** KestrelComponents  
**Result:** Ready: all 12 blocking checks passed; no warnings.  
**Checked by:** code, and `anthropic:claude-haiku-5-5` for the judgement checks

| Check | Severity | Result | Detail |
| --- | --- | --- | --- |
| Output format (C1) | blocking | Pass | all required fields present |
| Items match the input (C2) | blocking | Pass | every item named is in the input |
| Figures match the input (C3) | blocking | Pass | changes start from the pipeline's figures |
| Every judgement cites findings that apply (C4) | blocking | Pass | every judgement cites findings that apply |
| Proposed changes add up (C5) | blocking | Pass | the changes add up |
| Each Critical item has an action on its stock position (C6) | blocking | Pass | each Critical item has one |
| Summary names every Critical item (C7) | warning | Pass | names every Critical item |
| Summary within about 100 words (C8) | warning | Pass | 51 words |
| Arrows in the text match the proposed changes (C9) | blocking | Pass | every arrow matches a change |
| No invented or distorted history (J1) | blocking | Pass | Finding 1 restates document 1 correctly (supplier-wide, four weeks longer from 1 April 2025, until end of 2025). Finding 3 restates document 2 (GasketKit, 50% higher since January 2025, contract to December 2027) and finding 4 restates document 3 (OilSeal steady) correctly. Items and supplier attributions match the documents. |
| No question presumes a problem the data does not show (J2c) | warning | Pass | Critical question 1 asks how the extended lead times will affect the delivery schedule, which the history shows (document 1). Critical question 2 asks whether GasketKit demand continues beyond the current contract; this is neutral and the history supports a demand increase. No question presumes an unshown problem. |
| Stock figures quoted match the input (J7) | blocking | Pass | Summary and findings quote BearingCap 6-week gap (expected_stockout_weeks 6, matches). Finding 2 and summary do not quote stock figures otherwise. Quantity change reason for 520003 says '50%' which is from history, not stock data. Action text for 520001 mentions the 6-week gap, which matches. No mismatched stock figures found in prose. Changed PO dates and quantity appear only in change fields, not as '<pipeline> → <proposed>' prose mentions; the reasons do not quote stock values. |
| Each proposed change goes the way its findings point (J6a) | blocking | Pass | Document 1 says lead times lengthen by four weeks, so earlier order dates for 520002-520005 follow. However, 520001 BearingCap is missing: it is already overdue and was not shifted, which is acceptable. Quantity change for 520003 is upward, consistent with the 50% demand rise in document 2. |
| Each proposed change is the size its findings support (J6b) | blocking | Pass | Shift is -4 weeks for all four PO changes, matching the four-week lead-time increase. GasketKit factor 1.5 matches the 50% demand increase. The reasons explain these sizes. |
| No item the same reason applies to is left out (J6c) | blocking | Pass | The lead-time reason applies to all items with KestrelComponents: 520001 (BearingCap) is overdue and cannot move earlier, which is the allowed exception. 520002, 520003, 520004 and 520005 all received -4 week shifts. The only item not shifted is 520001, which is covered by the exception. |
| Supplier problem raised as a question (J2a) | warning | Pass | History document 1 describes a supplier-wide four-week lead-time increase from 1 April 2025 to end of 2025, which matters for ordering. Critical question 1 addresses it directly with the supplier. |
| Item problem raised as a question (J2b) | warning | Pass | History document 2 describes GasketKit demand rising 50%, which matters for ordering. Critical question 2 addresses it. OilSeal (document 3) shows steady demand, so no problem is described for it. |

## Purchase Request Text

**Result:** Ready: all 1 blocking checks passed; no warnings.

| Check | Severity | Result | Detail |
| --- | --- | --- | --- |
| Purchase request adds nothing to its input (P1) | warning | Pass | Purpose and justification are checked against the input. Claims that the BearingCap is 'at critical risk of stockout' with 'a six-week gap' match the input (risk_level Critical, gap 6 weeks). 'Order dates... advanced by four weeks due to KestrelComponents' extended lead times following their plant relocation' matches the changes list and the findings (four weeks longer from 1 April 2025, plant move to Gdansk). 'Increased by 50% to meet the heightened demand from a new contract' matches the GasketKit finding. 'Approval is contingent upon confirming the impact... with KestrelComponents' matches the action list. The text does not claim the gap is 'expected before new stock arrives' beyond the input's 'runs out before a new order can arrive'. Input says the ValveSeat order date moved from 2025-05-03 to 2025-04-05, which is four weeks earlier. The text says 'the order dates... have been advanced', consistent with changes. However, the justification says the 'BearingCap ... necessitating expedited shipping or sourcing from an alternative supplier', which matches the action. The text states 'relocation' where the input says 'plant move to Gdansk'; this is a near-synonym that does not change meaning. The purpose says 'to ensure continuity of supply', a reasonable framing of the actions. No figure is invented or judged. The 'Critical' risk is not explicitly named in the text, but the text's 'critical risk' matches the input. |
| Purchase request paragraphs within length (P2) | warning | Pass | within length |
| Purchase request cites only documents given (P3) | blocking | Pass | all cited documents given |

# Analysis Quality Check

**Supplier:** KestrelComponents  
**Result:** Ready: all 9 blocking checks passed; no warnings.  
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
| No invented or distorted history (J1) | blocking | Pass | Finding 1 says lead times for all products are four weeks longer from 1 April 2025, matching document [1] (supplier-wide). Finding 3 says GasketKit demand is 50% higher since January 2025 after a fleet contract, matching [2]. Finding 4 says OilSeal demand is steady, matching [3]. None is distorted or misattributed. |
| No question presumes a problem the data does not show (J2c) | warning | Pass | Question 1 asks how the extended lead times will affect deliveries, a problem the inputs show (document [1]). Question 2 asks whether the increased GasketKit demand continues beyond the contract, a problem the inputs show (document [2]). No question presumes an unshown problem. |
| Stock figures quoted match the input (J7) | blocking | Pass | Summary and finding 2 quote the BearingCap gap as 6 weeks, matching expected_stockout_weeks = 6. The only changed quantity (GasketKit 5200 → 7800) and the shifted PO dates appear only in the structured quantity_changes and po_date_changes fields, not in any text. No text mention lacks its arrow form. |
| Supplier problem raised as a question (J2a) | warning | Pass | The supplier-wide four-week lead-time increase from document [1] matters for ordering, and critical question 1 (ask: supplier) addresses it. |
| Item problem raised as a question (J2b) | warning | Pass | The item-level history that matters for ordering is GasketKit's 50% demand increase from document [2], and critical question 2 addresses it. OilSeal's 'steady' demand in [3] describes no problem. |

## Purchase Request Text

**Result:** Ready: all 1 blocking checks passed; 1 warning.

| Check | Severity | Result | Detail |
| --- | --- | --- | --- |
| Purchase request adds nothing to its input (P1) | warning | Fail | The purpose says the request addresses "imminent stockouts" for all four items. The input gives no such characterization, and it lists ValveSeat at 22 weeks to stockout, GasketKit at 24 and PistonRing at 30 (only BearingCap is Critical with 12 weeks). The justification also cites [1] for the BearingCap 6-week gap, but the input attaches no document to that finding (documents: []), so the citation is not supported by the input. |
| Purchase request paragraphs within length (P2) | warning | Pass | within length |
| Purchase request cites only documents given (P3) | blocking | Pass | all cited documents given |

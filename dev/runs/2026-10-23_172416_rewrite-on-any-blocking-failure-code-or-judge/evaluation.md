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
| Summary within about 100 words (C8) | warning | Pass | 52 words |
| Arrows in the text match the proposed changes (C9) | blocking | Pass | every arrow matches a change |
| No invented or distorted history (J1) | blocking | Pass | Finding 1 restates document 1 accurately (four weeks longer from 1 April 2025, plant move to Gdansk, until end of 2025). Finding 3 restates document 2 accurately (50% higher since January 2025, fleet operator contract until December 2027). Finding 4 restates document 3 ('steady over the past year'). No distortion or misattribution found. Note the summary and finding 2 cite stock data, not history. |
| No question presumes a problem the data does not show (J2c) | warning | Pass | Critical question 1 asks how the extended lead times will affect delivery schedules; this is a neutral question about a documented change. Critical question 2 asks whether GasketKit demand continues beyond the current contract; the contract runs to December 2027, so this is a reasonable question about a documented demand rise. No presumed unshown problem. |
| Stock figures quoted match the input (J7) | blocking | Pass | Summary: BearingCap '6-week gap' matches expected_stockout_weeks 6. Finding 2: '6 weeks' matches. Other figures quoted in prose: none beyond these. Summary and finding 3 cite the 50% demand rise from document 2, not stock data. Quantity change reason cites 50% (history figure). PO date changes' reasons cite four weeks (history figure). No stock figure in prose is mis-stated. However, GasketKit quantity change (5200 → 7800) is not written in the prose as '<pipeline> → <proposed>'; the action text for 520003 mentions 'order quantity' without figures, and the quantity change reason does not restate the quantity. The PO date changes' reasons do not quote dates. So no prose instance of a changed figure lacks the arrow form. |
| Each proposed change goes the way its findings point (J6a) | blocking | Pass | Four-week lead time extension (document 1, all products) is cited for PO date moves earlier by 4 weeks for items 520002, 520003, 520004, 520005. Consistent direction: longer lead time means ordering earlier. GasketKit quantity increase 1.5x cites finding 3 (50% demand rise), direction matches. |
| Each proposed change is the size its findings support (J6b) | blocking | Pass | Shift of -4 weeks matches the four-week lead time increase. Factor 1.5 matches the 50% demand increase. Reasons explain these sizes. |
| No item the same reason applies to is left out (J6c) | blocking | Pass | Lead time extension applies to all products, but BearingCap (520001) has no po_date_change, and ValveSeat, GasketKit, PistonRing, OilSeal have changes. BearingCap's recommended latest PO date is already overdue (2025-04-05 is before the snapshot effectively, shown as 'overdue'), so a move earlier cannot be made. That is the allowed exception. All other items receive the change. |
| Supplier problem raised as a question (J2a) | warning | Pass | Document 1 describes a supplier-wide lead time increase of four weeks until end of 2025. Critical question 1 addresses it directly, asking how the extended lead times affect delivery schedules. |
| Item problem raised as a question (J2b) | warning | Pass | Document 2 describes a 50% demand rise for GasketKit affecting ordering; critical question 2 asks whether it continues beyond the contract. Document 3 describes steady OilSeal demand, no problem. Addressed. |

## Purchase Request Text

**Result:** Ready: all 1 blocking checks passed; 2 warnings.

| Check | Severity | Result | Detail |
| --- | --- | --- | --- |
| Purchase request adds nothing to its input (P1) | warning | Not checked |  |
| Purchase request paragraphs within length (P2) | warning | Fail | justification has 5 sentences (2-4) |
| Purchase request cites only documents given (P3) | blocking | Pass | all cited documents given |

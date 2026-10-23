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
| No invented or distorted history (J1) | blocking | Pass | Finding 1 restates document 1 accurately (four weeks longer from 1 April 2025, plant move to Gdansk, until end of 2025, supplier-wide). Finding 3 restates document 2 accurately (50% higher since January 2025, fleet operator contract to December 2027, attributed to GasketKit). Finding 4 says OilSeal demand has been steady over the past year, matching document 3 and attributed to OilSeal. No distortion or misattribution found. |
| No question presumes a problem the data does not show (J2c) | warning | Pass | Critical question 1 asks how the extended lead times from the plant move will affect delivery, which the history shows (document 1). Critical question 2 asks whether the GasketKit demand increase continues beyond the contract, which is a question about a problem the inputs show (document 2). Neither presumes an unshown problem. |
| Stock figures quoted match the input (J7) | blocking | Pass | Summary and finding 2 give BearingCap's 6-week gap, matching expected_stockout_weeks 6. No stock figures are quoted for other items. The quantity change for 520003 is described in the reason as a 50% increase from the history, with no stock figure quoted in prose. The quantity_changes entry has original 5200 and proposed 7800, but the prose itself does not state the '<pipeline value> → <proposed value>' form for quantity. The reason field says 'Demand has increased by 50%', which is a history figure and not judged here. The prose mentions 'the GasketKit order quantity' in action text without a figure. However, the requirement is that wherever the prose mentions the changed figure, it is written as a pair; the action 'Review the GasketKit order quantity' mentions the quantity without a figure, so no violation is strictly evident in the prose. |
| Supplier problem raised as a question (J2a) | warning | Pass | Document 1 describes a supplier-wide lead time increase of four weeks from 1 April 2025 until end of 2025, which matters for ordering. Critical question 1 addresses it directly by asking how the extended lead times affect deliveries. |
| Item problem raised as a question (J2b) | warning | Pass | Document 2 describes a 50% demand increase for GasketKit, which matters for ordering. Critical question 2 addresses whether that increased demand continues beyond the current contract. Document 3 (OilSeal steady) describes no problem. The BearingCap gap is from stock data, not history. |

## Purchase Request Text

**Result:** Ready: all 1 blocking checks passed; 1 warning.

| Check | Severity | Result | Detail |
| --- | --- | --- | --- |
| Purchase request adds nothing to its input (P1) | warning | Not checked |  |
| Purchase request paragraphs within length (P2) | warning | Pass | within length |
| Purchase request cites only documents given (P3) | blocking | Pass | all cited documents given |

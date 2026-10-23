# Analysis Quality Check

**Supplier:** KestrelComponents  
**Result:** Ready: all 16 blocking checks passed; 1 warning.  
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
| History statements found in the documents (J1a) | blocking | Pass | Finding 1 (four-week lead time increase, Gdansk plant move, until end of 2025) matches document [1]; finding 3 (GasketKit demand 50% higher since January 2025, contract until December 2027) matches [2]; finding 4 (OilSeal demand steady) matches [3]. No invented history. |
| History not distorted (J1b) | blocking | Pass | Finding 1 says 'four-week increase in lead times for all products' matching [1]. Finding 3 states the 50% higher demand and contract end date matching [2]. Finding 4 'steady over the past year' matches [3]. No distortion. |
| History not attributed to what it does not name (J1c) | blocking | Pass | Document [1] is supplier-wide and is cited as supplier-wide in finding 1. Document [2] applies to GasketKit (any supplier) and is attributed to GasketKit 520003. Document [3] applies to OilSeal and is attributed to 520005. No misattribution. |
| No question presumes a problem the data does not show (J2c) | warning | Pass | Critical questions: 'How will the extended lead times... affect the delivery schedule...' (supplier) and 'Is the increased GasketKit demand expected to continue beyond the current contract?' (internal). Both address problems shown in inputs or are neutral. No presumed unshown problem. |
| Stock figures quoted match the input (J7a) | blocking | Pass | Summary and findings quote '6-week gap' for BearingCap, matching expected_stockout_weeks 6. No other stock figures quoted in prose. Finding 2 text matches the stock data. |
| Each proposed change goes the way its findings point (J6a) | blocking | Pass | po_date_changes for 520002, 520003, 520004, 520005 move earlier by 4 weeks, citing finding 1 (lead times +4 weeks), which supports earlier ordering. 520003 quantity raised by factor 1.5 citing finding 3 (demand 50% higher), which supports more quantity. |
| Each proposed change is the size its findings support (J6b) | blocking | Pass | Lead time increase is four weeks, and shifts are -4 weeks. Demand increase is 50%, and factor is 1.5. Reasons explain these sizes. |
| No item the same reason applies to is left out (J6c) | blocking | Pass | Lead time finding applies to all Kestrel items. 520001 has shift 0 with proposed date 2025-04-05, which equals the snapshot-adjacent date and is the item whose order date is already overdue (recommended_latest_po_timing 'overdue'), so it cannot move earlier. Others all shifted. 520001 is the exception allowed. |
| Changed figures written as pipeline value and proposed value (J7b) | blocking | Pass | The prose in the summary and findings does not mention specific order dates or quantities in pipeline-to-proposed form; quantity 5200 and dates appear only in change fields. Summary mentions 'review order quantities' without figures. |
| Questions marked for the supplier reveal nothing internal (J8) | blocking | Pass | Supplier question: 'How will the extended lead times due to the plant move affect the delivery schedule...' reveals no internal stock levels, weeks to stockout, budget, or urgency. |
| Each question and action follows from its findings (J9) | warning | Pass | Supplier question on lead time impact cites finding 1 (lead time extension), and internal GasketKit demand question cites finding 3. Actions: expedite BearingCap (finding 2 stockout gap), confirm lead time impact (finding 1), review GasketKit quantity (finding 3). Each follows. |
| No document with a problem for this order left unused (J10) | warning | Pass | Document [3] (OilSeal demand steady) is cited only in finding 4, which states steady demand. No question, change or action cites [3]. Finding 4 is not a question, change or action, so no problem is described that calls for action. |
| Findings state facts only (J11) | warning | Fail | Finding 1 states the lead time increase and its duration. Finding 2 states BearingCap runs out before a new order can arrive, with a 6-week gap. Finding 3 and 4 state document facts. However, finding 2 'runs out before a new order can arrive' is a judgement about the order, though derived from stock figures. Finding 3 and 4 are factual. |
| Supplier problem raised as a question (J2a) | warning | Pass | History [1] describes a supplier-wide four-week lead time increase until end of 2025, which matters for ordering. Critical question 1 addresses it by asking the supplier how the extended lead times affect the delivery schedule. |
| Item problem raised as a question (J2b) | warning | Pass | History [2] describes GasketKit demand increase, which matters for ordering. Critical question 2 asks whether demand will continue beyond the contract. History [3] (OilSeal steady) describes no problem. So the GasketKit problem is addressed. |

## Purchase Request Text

**Result:** Ready: all 1 blocking checks passed; 3 warnings.

| Check | Severity | Result | Detail |
| --- | --- | --- | --- |
| Purchase request adds nothing to its input (P1a) | warning | Not checked |  |
| Purchase request quotes figures as given (P1b) | warning | Not checked |  |
| Purchase request keeps the input's meaning (P1c) | warning | Not checked |  |
| Purchase request paragraphs within length (P2) | warning | Pass | within length |
| Purchase request cites only documents given (P3) | blocking | Pass | all cited documents given |

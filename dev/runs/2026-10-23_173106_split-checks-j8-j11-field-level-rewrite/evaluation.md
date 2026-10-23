# Analysis Quality Check

**Supplier:** NordicFasteners  
**Result:** Needs review: 3 of 16 blocking checks did not pass; 2 warnings.  
**Checked by:** code, and `anthropic:claude-haiku-5-5` for the judgement checks

| Check | Severity | Result | Detail |
| --- | --- | --- | --- |
| Output format (C1) | blocking | Pass | all required fields present |
| Items match the input (C2) | blocking | Pass | every item named is in the input |
| Figures match the input (C3) | blocking | Pass | changes start from the pipeline's figures |
| Every judgement cites findings that apply (C4) | blocking | Fail | question cites finding 3, which does not exist; action cites finding 2, which does not exist; action cites finding 3, which does not exist; quantity change cites finding 3, which does not exist |
| Proposed changes add up (C5) | blocking | Pass | the changes add up |
| Each Critical item has an action on its stock position (C6) | blocking | Fail | no action on the stock position of HexBoltM12 (410001) |
| Summary names every Critical item (C7) | warning | Pass | names every Critical item |
| Summary within about 100 words (C8) | warning | Pass | 52 words |
| Arrows in the text match the proposed changes (C9) | blocking | Pass | every arrow matches a change |
| History statements found in the documents (J1a) | blocking | Pass | Summary and finding 1 state the four-week November 2024 customs hold, and the supplier's paperwork change, which match document [1]. WasherSteel 60% demand rise matches [2]. No invented history. |
| History not distorted (J1b) | blocking | Pass | Document [1] says the supplier says it changed paperwork; output says 'claims to have since changed', which is faithful. WasherSteel 60% rise matches [2]. No distortion found. |
| History not attributed to what it does not name (J1c) | blocking | Pass | Finding 1 is supplier-wide and is correctly listed with items []. The WasherSteel demand note is attributed to WasherSteel, which [2] names. The AnchorPlug and CableTie documents are not cited as problems for other items. |
| No question presumes a problem the data does not show (J2c) | warning | Pass | Critical question 1 asks whether the customs process from November 2024 has been resolved, which is a problem the history shows. Question 2 is internal and asks whether demand will continue. Neither presumes an unshown problem. |
| Stock figures quoted match the input (J7a) | blocking | Pass | Summary and text cite no stock figures beyond the 9-week gap for HexBoltM12, which matches expected_stockout_weeks 9. Action text mentions 9-week gap, matching input. No mismatches found. |
| Each proposed change goes the way its findings point (J6a) | blocking | Fail | quantity_changes for 410002 raises quantity from 2600 to 4160 citing finding 3, but finding 3 is the AnchorPlug/WasherSteel demand note in the findings list. The cited finding ids do not match the documents, so the change's link to its reason is wrong. Finding 2 is the one that describes WasherSteel demand. |
| Each proposed change is the size its findings support (J6b) | blocking | Pass | Factor 1.6 matches the stated 60% demand rise, and the reason explains it. However, the cited findings id 3 does not support it; the reason itself is consistent. |
| No item the same reason applies to is left out (J6c) | blocking | Pass | Only WasherSteel is changed for the demand rise. No other item is affected by document [2], so no omission. |
| Changed figures written as pipeline value and proposed value (J7b) | blocking | Pass | Quantity change 410002 is 2600 → 4160 in the change fields, but the prose only mentions 'larger than suggested' without writing the figure as a pipeline-to-proposed pair; the 2600 and 4160 figures appear only in the change fields. Prose in action 3 does not mention the quantity figures. |
| Questions marked for the supplier reveal nothing internal (J8) | blocking | Pass | Supplier question 1 asks about the customs process and does not reveal stock levels, weeks to stockout, budget, or internal plans. |
| Each question and action follows from its findings (J9) | warning | Fail | Question 1 follows from finding 1 about customs delay. Action 1 (expedite HexBoltM12 for the 9-week gap) cites finding 2 (WasherSteel demand) instead of the stock data, so its link is wrong; the 9-week gap is in the stock input, not in finding 2. Question 2 cites finding 3 (AnchorPlug/WasherSteel demand), which does not match. |
| No document with a problem for this order left unused (J10) | warning | Pass | Document [3] (AnchorPlug cracked sleeves) is not used in any question, change or action. Document [4] (CableTie steady) is not used. Document [2] is used in the WasherSteel change, which is appropriate. No problem-document is linked wrongly except via the finding ids. |
| Findings state facts only (J11) | warning | Pass | Finding 1 states only the document facts: four-week hold, November 2024, customs error, and the supplier's claim. No judgement of impact on the order. |
| Supplier problem raised as a question (J2a) | warning | Pass | History describes a customs hold that matters for ordering, and critical question 1 addresses it directly. |
| Item problem raised as a question (J2b) | warning | Fail | Document [3] describes a cracked-sleeve batch for AnchorPlug (about 3% units), which may matter for ordering, but no critical question addresses it. Question 1 is supplier-wide and lists AnchorPlug only in the question text, not as the AnchorPlug quality issue. |

## Purchase Request Text

**Result:** Ready: all 1 blocking checks passed; 3 warnings.

| Check | Severity | Result | Detail |
| --- | --- | --- | --- |
| Purchase request adds nothing to its input (P1a) | warning | Not checked |  |
| Purchase request quotes figures as given (P1b) | warning | Not checked |  |
| Purchase request keeps the input's meaning (P1c) | warning | Not checked |  |
| Purchase request paragraphs within length (P2) | warning | Pass | within length |
| Purchase request cites only documents given (P3) | blocking | Pass | all cited documents given |

# Analysis Quality Check

**Supplier:** KestrelComponents  
**Result:** Needs review: 2 of 9 blocking checks did not pass; 3 warnings.  
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
| No invented or distorted history (J1) | blocking | Not checked | judge error: Expecting ',' delimiter: line 1 column 1291 (char 1290) |
| No question presumes a problem the data does not show (J2c) | warning | Not checked | judge error: Expecting ',' delimiter: line 1 column 1291 (char 1290) |
| Stock figures quoted match the input (J7) | blocking | Not checked | judge error: Expecting ',' delimiter: line 1 column 1291 (char 1290) |
| Supplier problem raised as a question (J2a) | warning | Not checked | judge error: Expecting ',' delimiter: line 1 column 1291 (char 1290) |
| Item problem raised as a question (J2b) | warning | Not checked | judge error: Expecting ',' delimiter: line 1 column 1291 (char 1290) |

## Purchase Request Text

**Result:** Ready: all 1 blocking checks passed; 1 warning.

| Check | Severity | Result | Detail |
| --- | --- | --- | --- |
| Purchase request adds nothing to its input (P1) | warning | Fail | The purpose says the request is "to address imminent stockouts and increased demand." The input gives only weeks_to_stockout values of 12, 22, 24 and 30, and the findings do not call any stockout imminent. Applying "imminent" to all four items, including those with 22 to 30 weeks of cover, adds a risk assessment the input does not give. |
| Purchase request paragraphs within length (P2) | warning | Pass | within length |
| Purchase request cites only documents given (P3) | blocking | Pass | all cited documents given |

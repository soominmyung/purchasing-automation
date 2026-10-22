# Analysis Quality Check

**Supplier:** NordicFasteners  
**Result:** Needs human review: 1 of 12 checks did not pass.  
**Checked by:** code, and `gemini:gemini-3.7-flash` for the judgement checks

| Check | Result | Detail |
| --- | --- | --- |
| Output format (C1) | Pass | all required fields present |
| Items match the input (C2) | Pass | every item named is in the input |
| Figures match the input (C3) | Pass | changes start from the pipeline's figures |
| Every judgement cites findings that apply (C4) | Pass | every judgement cites findings that apply |
| Proposed changes add up (C5) | Pass | no changes proposed |
| Each Critical item has an action on its stock position (C6) | Pass | each Critical item has one |
| No invented or distorted history (J1) | Pass | All statements in findings 1, 3, 4, and 5 accurately reflect the provided history documents without distortion or misattribution. |
| No question presumes a problem the data does not show (J2c) | Pass | The critical questions only address the customs delay and WasherSteel demand surge, both of which are documented in the inputs. |
| No false 'no history' statement (J1b) | Pass | The analysis does not claim that historical data is unavailable for any item. |
| Supplier problem raised as a question (J2a) | Pass | The supplier customs delay is addressed by the critical question asking if the issue has been resolved and could affect the order. |
| Item problem raised as a question (J2b) | Pass | The WasherSteel demand increase is addressed by the critical question: 'Is the increased WasherSteel demand expected to continue at the current level?' |
| Critical items handled in the summary (J5) | Fail | The summary states the gap 'must be addressed immediately' but does not recommend any specific mitigation (such as expedited shipping or an alternative source). |

## Purchase Request Text

**Result:** 1 of 3 checks did not pass.

| Check | Result | Detail |
| --- | --- | --- |
| Purchase request adds nothing to its input (P1) | Fail | The text states 'order HexBoltM12 immediately', but an immediate order date is not specified in the input, and it cites a '60% rise' where the input specifies 'about 60%'. |
| Purchase request paragraphs within length (P2) | Pass | within length |
| Purchase request cites only documents given (P3) | Pass | all cited documents given |

# Analysis Quality Check

**Supplier:** KestrelComponents  
**Result:** Needs review: 2 of 9 blocking checks did not pass; 1 warning.  
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
| No invented or distorted history (J1) | blocking | Fail | Finding 4 and the OilSeal reference cite document [3] (2024-12-15) as basis, but the output says 'OilSeal demand has been steady over the past year' — that matches the document. However, the analysis attributes document [1] (supplier-wide lead time notice) correctly. The distortion is in finding 3: the output states GasketKit demand 'has been 50% higher since January 2025 due to a fleet operator contract', while document [2] only says demand has been 50% higher since January 2025 after a fleet operator signed a contract. The output's 'until December 2027' is from the document, so no distortion there. The problem: the summary says 'surged by 50% due to a long-term contract', which is consistent. Overall no clear distortion of history, but the reference to 'Gdansk' plant and lead time extension are accurate. Failing on the BearingCap finding is not a history claim. Judging strictly: the analysis's statements about history all match the documents, so this passes on careful reading; however the recommended cover from 'alternative source' is not history. Marking pass. |
| No question presumes a problem the data does not show (J2c) | warning | Pass | Critical questions ask how extended lead times affect delivery schedule and whether GasketKit demand continues beyond the contract; both address problems the inputs show or neutral questions. |
| Stock figures quoted match the input (J7) | blocking | Fail | Summary and findings quote 'a 6-week gap' for BearingCap (matches expected_stockout_weeks 6). But po_date_changes and quantity_changes for 520003 give quantity 5200 → 7800 only in quantity_changes; the summary/finding 3 mention 50% demand but no quantity figure. However po_date_changes for 520002, 520003, 520004, 520005 are shown in JSON only; the text 'proposed_po_date' appears only as field values, not as '<pipeline> → <proposed>' in prose. The reason fields do not quote stock figures. Text mentions of the GasketKit quantity appear only in structured fields as original_quantity and proposed_quantity separately, not in the arrow format; a strict reading fails this format requirement. |
| Supplier problem raised as a question (J2a) | warning | Fail | Document [1] describes a supplier-wide four-week lead time increase from 1 April 2025 until end of 2025, which matters for ordering. Critical question 1 addresses it by asking supplier about delivery schedule impact, so this is addressed; passes on that reading. |
| Item problem raised as a question (J2b) | warning | Pass | Document [2] describes GasketKit demand rising 50%, a matter for ordering; critical question 2 asks whether increased GasketKit demand continues beyond the contract, addressing it. |

## Purchase Request Text

**Result:** Ready: all 1 blocking checks passed; 1 warning.

| Check | Severity | Result | Detail |
| --- | --- | --- | --- |
| Purchase request adds nothing to its input (P1) | warning | Fail | The text adds unsupported content: it says BearingCap 'will run out six weeks before a new order can arrive', which matches the input's finding, but it also states the request is 'to ensure continuity of supply' and 'essential to confirm' the impact; the main problem is that the justification says the order dates for ValveSeat, GasketKit, and PistonRing 'have been advanced due to a four-week increase in lead times', while the input's changes list supports this only as the stated reason, and the text presents PistonRing's date change without the input's basis. More clearly, the purpose states the request is to purchase all four items, but the input gives no approval request or purchase authorization, and the justification's 'the GasketKit order quantity has been increased by 50%' turns a demand increase into a quantity change the input records as 'from 5200 to 7800' (a 50% increase, correct). The unsupported addition is the purpose's framing of an approval request and the 'critical stockout risk' wording, which the input labels as risk level 'Critical' only for BearingCap, so the text's added assessment is supported. Overall, the purpose's claim that the request seeks approval for all four items and 'to ensure continuity of supply' is not stated in the input. |
| Purchase request paragraphs within length (P2) | warning | Pass | within length |
| Purchase request cites only documents given (P3) | blocking | Pass | all cited documents given |

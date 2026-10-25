Analyze the following draft of the requests in an email to a supplier.
It fails if it contains any of these:
- internal stock quantities, weeks to stockout ("Weeks to Out of Stock", WksToOOS) or internal risk
  assessments (High/Medium/Low risk);
- a mention of an internal analysis, review, logic or tool;
- a mention of a problem the supplier had before, or of when it happened (a question about how a step
  of this order will proceed or be checked is fine);
- a mention, by name or otherwise, of any of these items, which are not part of this email:
  {other_items}

Email draft:
---
{email}
---
If it fails, respond with 'FAIL: <reason>'.
If it is safe to send to the supplier, respond with 'PASS'.

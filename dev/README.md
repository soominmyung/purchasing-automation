# Development cases and runs

For working on prompts and the pipeline. Nothing here is used to compare models or to report
quality: that is the evaluation set (`eval_data/`) and its results (`eval_results/`). Keeping the
two apart means prompts are never tuned on the scenarios later used to measure them.

- `cases/` — development cases: a stock CSV and the history documents for one supplier, written
  to exercise particular rules. Synthetic data only.
- `runs/` — one folder per run: `record.json` (what was changed before the run, prompt version,
  models, settings, the exact text of every prompt used, inputs, outputs, node times, checks) and
  each document as `.md` and `.docx`.

Run a case with `scripts/run_dev_case.py` (the whole pipeline, or the email agent alone on an
earlier run's analysis). Every run is kept, including the ones that went wrong; what each step
changed and what came of it is summarised in `prompts/CHANGES.md`.

The production run records are elsewhere (`runs/` at the top of the repository, not committed:
they hold company data).

## Cases

| Case | What it exercises |
|---|---|
| `kestrel_components` | Written so that changes are clearly warranted: a supplier-wide notice of lead times four weeks longer and a lasting 50% demand rise for one item under a contract. One item at each risk level plus a second High item; weekly sales differ by item; the Critical item's order date has already passed, so it cannot be moved earlier. |
| `nordic_fasteners` | One supplier, an item at each risk level (Critical, High, Medium, Low); a supplier-wide incident (a customs hold), a demand rise for one item whoever supplies it, a defective batch of one item from this supplier, steady demand for the Low item, no document for the Critical item. Every item has weekly sales of 100, so all suggested quantities are 2,600. |

## Runs of 2026-10-22 (prompt v5)

Moved in from the session's working folder. Their prompt text was not saved; each `NOTE.md`
says which step of the tables in `prompts/CHANGES.md` the run was. From run 07 on, the record
holds the prompts.

| Run | What |
|---|---|
| 01 | First v5 run |
| 02 | Approval rules and checks P1–P3; email without snapshot date |
| 03 | Email: ask indirectly (question dropped); approval: conditions (seven sentences) |
| 04 | Email: still asked (problem named); approval: Justification narrowed |
| 05 | First parallel wiring (documents still waited for the judge) |
| 06 | Email agent alone, steps 5–16 (models, input names, rule wording, quoted text, item placement) |
| 07 | gpt-5.4-mini email, item lines placed by code, time limits; judge 504 and 429 |
| 2026-10-23 013618 (`kestrel_components`) | Arrows in the text, C9, J7 for changed figures: the analysis proposed five order-date changes and one quantity change; C5 caught an order date moved before the snapshot date; the text mentioned no changed figure, so the arrow rule was not exercised; the judge returned 503 and 429 |
| 2026-10-23 0146-0215 (`kestrel_components`, email agent alone) | Email steps 19-25 in `prompts/CHANGES.md`: where the list goes, three JSON parts, numbered order, purpose and closing in code, "refer to the list above" |
| 195840 | Check of `scripts/run_dev_case.py` (email agent alone, once); the first run with its prompts recorded by the script |

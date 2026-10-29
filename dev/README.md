# Development cases

For working on prompts and the pipeline. Nothing here is used to compare models or to report
quality: that is the evaluation set (`eval_data/`) and its results (`eval_results/`). Keeping the
two apart means prompts are never tuned on the scenarios later used to measure them.

Each case is a stock CSV and the history documents for one supplier, written to exercise
particular rules. Synthetic data only.

Run a case with `scripts/run_dev_case.py` (the whole pipeline, or the email agent alone on an
earlier run's analysis). Each run is recorded in `dev/runs/`: what was changed before the run,
prompt version, models, settings, the exact text of every prompt used, inputs, outputs, node
times and checks, with each document as `.md` and `.docx`. One run is committed as a sample,
`kestrel_components` through the whole pipeline with the current prompts and checks
(`runs/2026-10-29_004531_judge-checks-renumbered-without-gaps-j3a-c-j4-j8/`); the others are not committed.

## Cases

| Case | What it exercises |
|---|---|
| `kestrel_components` | Changes clearly warranted: a supplier-wide notice of lead times four weeks longer and a lasting 50% demand rise for one item under a contract. One item at each risk level, plus two items whose risk level rises at the longer lead time, so a figure worked out at 18 weeks shows; weekly sales differ by item; the Critical item's order date has already passed. |
| `nordic_fasteners` | An item at each risk level and one history document of each kind: a supplier-wide incident (a customs hold the supplier says it has fixed), a demand rise for one item whoever supplies it, a defective batch of one item from this supplier, steady demand for the Low item; no document for the Critical item. Tests restraint: no lead-time change is warranted. |
| `delayvale_low_becomes_medium` | Every item Low at 18 weeks, but the supplier has announced lead times four weeks longer: at 22 weeks one item becomes Medium, so the run goes on to a purchase request and an email. |
| `hanbit_demand_market` | Whether a quantity change rests only on UK demand: three items with the same figures and a demand rise of unstated size, differing only in the market (not stated, the UK, Korea). |
| `lowfield_all_low` | Every item Low and no reason in the history to change that: the run ends with the analysis report, with no purchase request or email. |

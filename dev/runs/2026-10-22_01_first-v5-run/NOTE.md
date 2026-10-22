# 2026-10-22_01_first-v5-run

First v5 run. Email step 1 (gpt-4o-mini), approval step 1. The documents were re-assembled afterwards from the saved model outputs, once two assembly bugs were fixed (the internal question printed as a supplier one; ITEMS without braces left in the email); state.json holds the model outputs as returned. The judge failed J5 (mitigation not in the summary).

Moved from the working folder of the 2026-10-22 session. The prompt text used was not saved at the time (prompts were edited in place); the change made before each run is in prompts/CHANGES.md, v5, tables "Tuning the email prompt" and "Tuning the approval prompt". From now on scripts/run_dev_case.py keeps the prompt text with every run.

# Files

Every file in the repository, by folder. The overview of the system is in [`README.md`](README.md).

## Top level

| File | What it is |
|---|---|
| `README.md` | Overview: how a run works, the quality checks, the model choices, self-hosting, how to run it |
| `FILES.md` | This list |
| `main.py` | The FastAPI app: LangSmith settings at start-up, the routers, access control |
| `config.py` | Settings, read from environment variables (`.env`): models, rewrites, history window, prompt version, LangSmith, run records |
| `schemas.py` | Request and response models of the API, validated at the API boundary |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container image for Cloud Run |
| `.env.example` | The environment variables to set, with placeholders |
| `050425.csv` | Sample stock CSV (snapshot date 2025-04-05; five items from three suppliers) |
| `.gitignore` | Files kept out of the repository: environments, build files, the history store, generated documents, run records |
| `.gitattributes` | Git LFS patterns for large binary files |

## `.github/workflows/`

| File | What it is |
|---|---|
| `deploy.yml` | Builds the image and deploys the public demo to Cloud Run |

## `routers/`: the API

| File | What it is |
|---|---|
| `pipeline.py` | `POST /api/run` and `/api/run/stream`: stock CSV → grouping → analysis → documents; files returned for download or base64-encoded |
| `ingest.py` | History documents in: PDF uploads, a ZIP, or the shared-folder sync |
| `output.py` | Serves the generated Word files; temporary files are deleted after five minutes |
| `__init__.py` | Package marker |

## `services/`: the pipeline

| File | What it is |
|---|---|
| `agents.py` | The LangGraph graph and the agents: analysis, purchase request, supplier email and its validator, the judge node, rewrites |
| `quality_checks.py` | The checks: code checks C1–C10, the judge's checks J1a–J8, P1–P3 and E1–E7, their wording and severity |
| `documents.py` | The documents assembled in code from the analysis: report, purchase request, email; figures after a lead-time change; each agent's input |
| `item_grouping.py` | Stock rows grouped by supplier; risk level, gap, latest order and delivery dates, suggested quantity ([`docs/stock_figures.md`](docs/stock_figures.md)) |
| `vector_store.py` | The history store (Chroma): documents kept whole, retrieved by their supplier, item and date headers |
| `history_folder.py` | Keeps the history store in step with a shared folder (new, changed, deleted files) |
| `prompts.py` | Loads the prompts of the version in use and hashes them for the run record |
| `run_record.py` | One JSON record per run (inputs, history used, outputs, checks, times, tokens) and the run and check statistics in CSV |
| `security.py` | API key check and rate limiting |
| `__init__.py` | Package marker |

## `utils/`

| File | What it is |
|---|---|
| `csv_utils.py` | CSV parsing and validation; the snapshot date from the `SnapshotDate` line or the file name |
| `docx_utils.py` | Markdown to Word, with real tables and the red review notes |
| `pdf_utils.py` | Text extraction from PDF history documents |
| `__init__.py` | Package marker |

## `prompts/v5/`: the prompts in use

| File | Used by |
|---|---|
| `analysis.md` | The analysis agent (system prompt): input, stock rules, history rules, output fields and rules, one example |
| `analysis_revision_request.md` | The analysis rewrite: the failed checks sent back |
| `approval_request.md` | The purchase request writer |
| `email_draft.md` | The supplier email writer |
| `email_revision_request.md` | The email rewrite: the validator's reason sent back |
| `email_revision_system_suffix.md` | Added to the email writer's system prompt on a rewrite |
| `email_validator_system.md` | The email validator (system prompt) |
| `email_validator.md` | The email validator: what an email must not contain |

## `docs/`

| File | What it is |
|---|---|
| `Purchasing_AI_Suite_en.pdf` | Portfolio, English (7 pages) |
| `Purchasing_AI_Suite_ko.pdf` | Portfolio, Korean (7 pages) |
| `images/pipeline.png` | The pipeline diagram in the README |
| `eval_rubric.md` | Definitions of all 36 checks, what each needs to pass, severity, what happens on a failure |
| `stock_figures.md` | How code works out each stock figure from the snapshot date (English and Korean) |
| `analysis_model_selection.md` | How the analysis model and effort were chosen |
| `judge_selection.md` | How the judge was validated and chosen |
| `approval_model_selection.md` | How the purchase request writer was chosen |
| `email_model_selection.md` | How the email writer and the email validator were chosen |
| `serving_gemma4_vllm.md` | Gemma 4 12B served with vLLM: bf16 against FP8, memory, throughput, quality |
| `sft_gemma4.md` | Gemma 4 12B fine-tuned with QLoRA SFT: the training data, one example, the result (English and Korean) |

## `scripts/`

| File | What it is |
|---|---|
| `run_dev_case.py` | Runs a development case through the whole pipeline and records it in `dev/runs/` |
| `sync_history_folder.py` | Syncs the history store with the shared folder |
| `check_mcp_servers.py` | Starts each MCP server as a client would and calls every tool once |
| `eval_scenarios.py` | The 25 evaluation scenarios turned into the pipeline's input |
| `build_v5_base_outputs.py` | The analysis and purchase request on the 25 scenarios, the inputs of the later comparisons |
| `probe_order_dates.py` | Probe: how `gpt-4o` works out the latest order date after a lead-time change |
| `count_action_timing.py` | Counts how the analysis actions state their timing, per base-output file |
| `build_v5_judge_calibration.py` | Builds the judge's calibration set (seeded faults) |
| `run_judge_validation.py` | Judge candidates on the calibration set: detection and flip |
| `hand_scoring_v5.py` | Writes the hand-scoring sheet and reads the filled sheet back into the hand verdicts |
| `run_judge_false_alarm.py` | Judge candidates on the hand-scored outputs: false alarms |
| `rerun_p1_wording.py` | The purchase request checks again after P1a's wording changed |
| `compare_approval_models.py` | The purchase request written by every candidate model and effort |
| `judge_approval_models.py` | The judge's checks P1a–c on those purchase requests, three runs each |
| `compare_email_models.py` | The supplier email written by every candidate, through the production validator loop |
| `compare_email_validators.py` | The email validators compared on real and planted faults |
| `screen_base_models.py` | Open models, untuned, as the analysis agent on the 25 cases with the production checks |
| `screen_all_candidates.ps1` | Serves each open model in turn and screens it |
| `serve_local_model.ps1` | Serves one open model with vLLM on the local GPU |
| `generate_sft_scenarios.py` | Generates the SFT training scenarios |
| `build_sft_teacher_data.py` | The teacher (`gpt-6.1-sol`) on each scenario, scored by the production checks |
| `train_sft_v5.py` | QLoRA SFT of Gemma 4 12B with Unsloth |
| `train_learning_curve.ps1` | The learning curve: 25, 50 and 85 examples, two seeds each |
| `report_sft_results.py` | Tables of the SFT results |
| `serve_quant_compare.ps1` | Gemma 4 12B served bf16, then FP8: quality and throughput |
| `bench_vllm_v5.py` | Serving throughput on the production request at 1 to 16 concurrent requests |

## `mcp_servers/`

Built and tested, not used in the pipeline: code fixes the order of retrieval and calculation, so no history is skipped by a tool call left out.

| File | What it is |
|---|---|
| `history_server.py` | MCP server for the history retrieval |
| `replenishment_server.py` | MCP server for the stock figures, worked out by the pipeline's own code |
| `claude_desktop_config.example.json` | Example client configuration |

## `eval_data/`: evaluation inputs

| File | What it is |
|---|---|
| `eval_scenarios.jsonl` | 20 evaluation scenarios: stock rows and history documents, with labels |
| `holdout_scenarios.jsonl` | 5 more scenarios (H1–H5), carried over from the earlier evaluation set; with the 20, the 25 evaluation cases |
| `judge_calibration_v5.jsonl` | The judge's 51 calibration samples: 46 with a planted fault, 3 with a real fault, 2 without one |
| `hand_scoring_sheet_v5.md` | The sheet six outputs were scored on by hand |
| `hand_scores_v5.jsonl` | The 89 hand verdicts |

## `eval_results/`: raw results

Every output and verdict behind the choices in `docs/`; how the files are grouped is in
[`eval_results/README.md`](eval_results/README.md). `.jsonl` holds the records, `.md` the same
outputs side by side for reading, `*_settings.json` the models and check wording of a run.

| File | What it is |
|---|---|
| `README.md` | How the results are grouped |

**Analyses of the 25 evaluation scenarios** (`gpt-6.1-sol`), the inputs of the purchase request and email comparisons:

| File | Analysis prompt |
|---|---|
| `v5_base_outputs.jsonl` | The first v5 prompt |
| `v5_base_outputs_timing_v1.jsonl` | Actions say whether they come before, alongside or after the order (first wording) |
| `v5_base_outputs_timing.jsonl` | Action timing judged against the latest order date (kept) |
| `v5_base_outputs_delay.jsonl` | Quantity changes on UK demand only, and a delay without a size allowed for with an estimate (24 scenarios; the delay rule was dropped) |
| `v5_base_outputs_estimate.jsonl` | Quantity changes on UK demand only; an estimated size says what it rests on (the current prompt) |

**Judge** ([`docs/judge_selection.md`](docs/judge_selection.md)):

| File | What it is |
|---|---|
| `judge_validation_v5_round1_runs.jsonl`, `_settings.json` | Round 1: Haiku (forced tool call), Gemini Flash-Lite, `gpt-6-luna` on the 51 samples, three runs |
| `judge_validation_v5_round2_runs.jsonl`, `_settings.json` | Round 2: Haiku with thinking, medium and high effort |
| `judge_false_alarm_v5_runs.jsonl`, `_settings.json` | False alarms against the 89 hand verdicts |
| `judge_p1_rewording_v1_runs.jsonl`, `_settings.json` | P1a's first new wording |
| `judge_p1_rewording_runs.jsonl`, `_settings.json` | P1a's kept wording |

**Purchase request** ([`docs/approval_model_selection.md`](docs/approval_model_selection.md)). `comparison` holds the texts, `judged` the judge's verdicts and the issues found by reading, `issues` those issues alone:

| File | Round |
|---|---|
| `approval_model_comparison.jsonl`, `.md` | The first approval prompt: four models, then every model and effort |
| `approval_model_judged.jsonl`, `.md`, `_settings.json` | Judged |
| `approval_model_issues_effort.json` | Issues found by reading, at set efforts |
| `approval_model_comparison_revised_v1.jsonl` | The revised approval prompt, first wording |
| `approval_model_judged_revised_v1.jsonl`, `.md`, `_settings.json` | Judged |
| `approval_model_issues_revised_v1.json` | Issues found by reading |
| `approval_model_comparison_revised.jsonl` | The revised approval prompt, kept wording |
| `approval_model_judged_revised.jsonl`, `.md`, `_settings.json` | Judged |
| `approval_model_issues_revised.json` | Issues found by reading |
| `approval_model_comparison_timing_v1.jsonl` | After the analysis timing rule, first wording |
| `approval_model_judged_timing_v1.jsonl`, `_settings.json` | Judged |
| `approval_model_comparison_timing.jsonl` | After the analysis timing rule, kept wording: the final comparison |
| `approval_model_judged_timing.jsonl`, `.md`, `_settings.json` | Judged |
| `approval_model_comparison_estimate.jsonl` | After the analysis rule on estimated sizes |
| `approval_model_judged_estimate.jsonl`, `_settings.json` | Judged |

**Supplier email** ([`docs/email_model_selection.md`](docs/email_model_selection.md)):

| File | Round |
|---|---|
| `email_model_v1_comparison.jsonl`, `.md` | Round 1: emails written |
| `email_model_v1_judged.jsonl`, `_settings.json` | Round 1: judged |
| `email_model_v2_comparison.jsonl`, `.md` | Round 2: emails written |
| `email_model_v2_judged.jsonl`, `_settings.json` | Round 2: judged |
| `email_model_comparison.jsonl`, `.md` | Round 3: the final emails |
| `email_model_v3_comparison.md`, `email_model_v3_judged.jsonl`, `_settings.json` | Round 3, first scoring |
| `email_model_v3b_comparison.md`, `email_model_v3b_judged.jsonl`, `_settings.json` | Round 3, second scoring |
| `email_model_judged.jsonl`, `_settings.json` | Round 3, final scoring |
| `email_validator_comparison.jsonl`, `_settings.json` | The email validators on round 3's emails and planted faults |

**Self-hosting** ([`docs/serving_gemma4_vllm.md`](docs/serving_gemma4_vllm.md), [`docs/sft_gemma4.md`](docs/sft_gemma4.md)). `sft_screen_<model>_<tag>.jsonl`: one model as the analysis agent with the production checks. Tags: none, the first screening; `fac-rule`, the 25 cases with the current prompt; `in-dist`, the 13 validation scenarios from the training data's generator; `tie-in`, the 7 generalisation scenarios; `serve-bf16` and `serve-fp8`, the serving comparison.

| File | What it is |
|---|---|
| `sft_screen_gpt-6.1-sol.jsonl`, `_fac-rule.jsonl` | The production model, for reference |
| `sft_screen_gemma-4-12b-it.jsonl`, `_fac-rule.jsonl` | Gemma 4 12B, untuned |
| `sft_screen_gemma-4-12b-it_in-dist.jsonl`, `_tie-in.jsonl` | Gemma 4 12B, untuned, on the other scenario sets |
| `sft_screen_gemma-4-12b-it_serve-bf16.jsonl`, `_serve-fp8.jsonl` | Gemma 4 12B served bf16 and FP8 |
| `sft_screen_gemma-4-e4b-it.jsonl`, `_fac-rule.jsonl` | Gemma 4 E4B, untuned |
| `sft_screen_ministral-3-14b.jsonl`, `_fac-rule.jsonl` | Ministral 3 14B, untuned |
| `sft_screen_ministral-3-8b.jsonl`, `_fac-rule.jsonl` | Ministral 3 8B, untuned |
| `sft_screen_qwen3.5-9b.jsonl`, `_fac-rule.jsonl` | Qwen 3.5 9B, untuned |
| `sft_screen_sft-n25_s1_fac-rule.jsonl`, `_tie-in.jsonl` | Fine-tuned on 25 examples, seed 1 |
| `sft_screen_sft-n25_s2_fac-rule.jsonl`, `_tie-in.jsonl` | Fine-tuned on 25 examples, seed 2 |
| `sft_screen_sft-n50_s1_fac-rule.jsonl`, `_tie-in.jsonl` | Fine-tuned on 50 examples, seed 1 |
| `sft_screen_sft-n50_s2_fac-rule.jsonl`, `_tie-in.jsonl` | Fine-tuned on 50 examples, seed 2 |
| `sft_screen_sft-n85_s1_fac-rule.jsonl`, `_in-dist.jsonl`, `_tie-in.jsonl` | Fine-tuned on 85 examples, seed 1 |
| `sft_screen_sft-n85_s2_fac-rule.jsonl`, `_in-dist.jsonl`, `_tie-in.jsonl` | Fine-tuned on 85 examples, seed 2 |
| `serving_bench_gemma4-12b-bf16.json` | vLLM throughput and latency, bf16, 1 to 16 concurrent requests |
| `serving_bench_gemma4-12b-fp8.json` | The same, FP8 |

## `training_data/sft_v5/`: fine-tuning data

| File | What it is |
|---|---|
| `scenarios.jsonl` | 117 generated training scenarios: 13 types, nine each |
| `teacher_scenarios.jsonl` | The teacher's runs on them, each with the exact messages and the production checks' verdicts; the 98 that passed every check are the training and validation examples |
| `validation_scenarios.jsonl` | The 13 validation scenarios, held out of training |
| `generalisation_tie_in.jsonl` | 7 scenarios of a type no training scenario has (a supplier pushing a new product) |
| `teacher_generalisation_tie_in.jsonl` | The teacher's runs on those 7 |

## `dev/`: development cases

Kept apart from the evaluation set: used to work on the prompts and the pipeline, not to compare models.

| File | What it is |
|---|---|
| `README.md` | What the cases are for and how to run them |
| `cases/kestrel_components.json` | Changes clearly warranted: a supplier-wide four-week delay and a lasting demand rise; two items whose risk rises at 22 weeks |
| `cases/nordic_fasteners.json` | No lead-time change warranted: one item at each risk level and one history document of each kind (a customs hold since fixed, a demand rise, a defective batch, steady demand) |
| `cases/hanbit_demand_market.json` | Whether a quantity change rests only on UK demand: the same demand rise, in different markets |
| `cases/delayvale_low_becomes_medium.json` | A Low item that becomes Medium once a delay is allowed for |
| `cases/lowfield_all_low.json` | Every item Low: only the analysis report is written |

One recorded run of the Kestrel case, `runs/2026-10-29_004531_judge-checks-renumbered-without-gaps-j3a-c-j4-j8/`:

| File | What it is |
|---|---|
| `record.json` | The run: settings, models, efforts, the prompt text, the outputs and checks, and each node's time and tokens |
| `analysis_report.md`, `.docx` | The analysis report |
| `purchase_request.md`, `.docx` | The purchase request |
| `email_draft.md`, `.docx` | The supplier email draft |
| `evaluation.md`, `.docx` | The quality report: every check's verdict and reason |

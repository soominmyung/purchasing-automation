# Serving evaluation artifacts

Raw outputs behind the quality claims in [../docs/inference_serving_vllm.md](../docs/inference_serving_vllm.md).
All of it is generated from the synthetic scenarios in [../training_data/](../training_data/) —
no real supplier or company data.

| File | What it is |
|---|---|
| `serving_fp16_outputs.json` | The served model's analysis output for all 30 scenarios, **FP16** deployment. Each record: `{idx, reference, output}` where `reference` is the GPT-4o ground truth from the teacher dataset. |
| `serving_fp8_outputs.json` | Same 30 scenarios, **FP8** deployment. |
| `serving_quality_scored.json` | GPT-4o judge scoring each output 1–10 on data accuracy + reasoning. Means: **FP16 7.18, FP8 7.08** (−0.1). |
| `serving_quality_pairwise.json` | GPT-4o judge comparing FP16 vs FP8 head to head, each pair judged twice with the order swapped. Result: **FP8 15 / FP16 9 / 6 ties** — not significant vs a 50/50 split. |

Throughput and latency tables are in the write-up rather than here; the benchmark ran on an
ephemeral GPU VM that was deleted after the run.

## Regenerating

```bash
# 1. serve the model (see scripts/bench_vllm.py docstring for the docker command),
#    once as FP16 and once with --quantization fp8
python scripts/gen_serving_outputs.py training_data/teacher_dataset_20260302.jsonl \
    eval_results/serving_fp16_outputs.json 30

# 2. compare the two runs (needs OPENAI_API_KEY)
python scripts/eval_quantization_quality.py \
    eval_results/serving_fp16_outputs.json eval_results/serving_fp8_outputs.json --mode scored
python scripts/eval_quantization_quality.py \
    eval_results/serving_fp16_outputs.json eval_results/serving_fp8_outputs.json --mode pairwise
```

Outputs are sampled at temperature 0.1, so a rerun will not reproduce the files byte for byte.

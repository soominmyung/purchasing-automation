"""
Before/After Comparison: Base Llama vs SFT vs SFT+DPO vs GPT-4o

Runs the holdout test set (last 5 examples) through four models and
scores each with GPT-4o-as-judge using the same rubric as eval_sft.py:
  - data_accuracy (1-10): correct use of supplier/item data
  - reasoning_quality (1-10): logical replenishment analysis
  - avg = (data_accuracy + reasoning_quality) / 2

Results are saved to GCS and W&B for direct comparison with SFT baseline.

Usage (Vertex AI Custom Training Job):
    Override CMD with: ["python", "scripts/eval_dpo.py"]
"""

# IMPORTANT: Unsloth must be imported first
from unsloth import FastLanguageModel

import gc
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import torch
import wandb
from google.cloud import storage as gcs
from openai import OpenAI

# ── Configuration ───────────────────────────────────────────────────────────────

DATASET_PATH = os.environ.get(
    "DATASET_PATH", "training_data/teacher_dataset_20260302.jsonl"
)
HOLDOUT_N = 5

GCS_SFT_ADAPTER_URI = os.environ.get(
    "GCS_SFT_ADAPTER_URI", "gs://purchasing-automation-models/sft-runs/lora_adapter"
)
GCS_DPO_ADAPTER_URI = os.environ.get(
    "GCS_DPO_ADAPTER_URI", "gs://purchasing-automation-models/dpo-runs/lora_adapter"
)
GCS_EVAL_OUTPUT_URI = os.environ.get(
    "GCS_EVAL_OUTPUT_URI", "gs://purchasing-automation-models/eval-results"
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "purchasing-automation-dpo")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "msm1640-")

SFT_LOCAL_PATH = "/app/models/sft-adapter"
DPO_LOCAL_PATH = "/app/models/dpo-adapter"
MAX_SEQ_LENGTH = 2048


# ── Prompt Template (must match train_sft.py) ───────────────────────────────────

PROMPT_TEMPLATE = """### Instruction
{instruction}

### Input
Supplier: {supplier}
Items: {items}
Supplier History: {supplier_history}
Item History: {item_history}

### Response
"""


def format_prompt(example: dict) -> str:
    inp = example["input"]
    inventory = inp.get("inventory", [])
    supplier = inventory[0].get("SupplierName", "Unknown") if inventory else "Unknown"
    return PROMPT_TEMPLATE.format(
        instruction=example.get("instruction", "Analyze the purchasing data."),
        supplier=supplier,
        items=json.dumps(inventory, ensure_ascii=False),
        supplier_history=inp.get("supplier_history", "No supplier history available."),
        item_history=inp.get("item_history", "No item history available."),
    )


# ── GCS Helpers ─────────────────────────────────────────────────────────────────

def _download_gcs_prefix(gcs_uri: str, local_dir: str) -> None:
    uri_parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name, prefix = uri_parts[0], (uri_parts[1] if len(uri_parts) > 1 else "")
    blobs = list(gcs.Client().bucket(bucket_name).list_blobs(prefix=prefix))
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    for blob in blobs:
        rel = blob.name[len(prefix):].lstrip("/")
        if not rel:
            continue
        dest = Path(local_dir) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
    print(f"Downloaded {len(blobs)} files → {local_dir}")


def _upload_file_to_gcs(local_file: str, gcs_uri: str) -> None:
    uri_parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = uri_parts[0]
    blob_path = uri_parts[1] if len(uri_parts) > 1 else Path(local_file).name
    gcs.Client().bucket(bucket_name).blob(blob_path).upload_from_filename(local_file)
    print(f"Uploaded → gs://{bucket_name}/{blob_path}")


# ── Inference ────────────────────────────────────────────────────────────────────

def run_llama_inference(adapter_path: str, prompts: list) -> list:
    """Load a LoRA adapter (or base model) and run inference on all prompts."""
    print(f"\nLoading model from: {adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    outputs = []
    for i, prompt in enumerate(prompts):
        print(f"  Generating [{i+1}/{len(prompts)}]...")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_new_tokens=900,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        text = tokenizer.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(text.strip())

    # Free VRAM before loading next model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


# ── Scoring (same rubric as eval_sft.py) ─────────────────────────────────────────

def is_valid_json(text: str):
    """Extract first valid JSON object — mirrors eval_sft.py."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return True, json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        try:
            return True, json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return False, None


def score_with_gpt4o(client: OpenAI, prompt: str, reference: dict, candidate: str, model_name: str) -> dict:
    """
    GPT-4o-as-judge with the same rubric as eval_sft.py.
    Returns: {data_accuracy, reasoning_quality, comment}
    """
    judge_prompt = f"""You are evaluating a purchasing analysis AI response. Score it on two criteria (1-10 each).

=== Original Input ===
{prompt}

=== Reference (GPT-4o ground truth) ===
{json.dumps(reference, ensure_ascii=False, indent=2)[:3000]}

=== Candidate Response ({model_name}) ===
{candidate[:3000]}

Evaluate:
1. data_accuracy (1-10): Does the candidate correctly reference the supplier names, item codes, stock levels, and risk levels from the input?
2. reasoning_quality (1-10): Is the replenishment analysis and critical questions logically sound and relevant?

Respond in JSON only:
{{"data_accuracy": <int>, "reasoning_quality": <int>, "comment": "<one sentence>"}}"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=200,
        )
        raw = (resp.choices[0].message.content or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
        _, parsed = is_valid_json(cleaned)
        if parsed:
            return {
                "data_accuracy": parsed.get("data_accuracy", 0),
                "reasoning_quality": parsed.get("reasoning_quality", 0),
                "comment": parsed.get("comment", ""),
            }
        print(f"  Judge parse failed. Raw: {raw[:200]}")
        return {"data_accuracy": 0, "reasoning_quality": 0, "comment": f"parse_failed: {raw[:100]}"}
    except Exception as e:
        print(f"  Judge error: {e}")
        return {"data_accuracy": 0, "reasoning_quality": 0, "comment": str(e)}


# ── Main Comparison ───────────────────────────────────────────────────────────────

def run_comparison():
    print("=" * 60)
    print("  Before/After: Base / SFT / SFT+DPO / GPT-4o")
    print("=" * 60)

    # Load holdout set (last N examples)
    with open(DATASET_PATH, encoding="utf-8") as f:
        examples = [json.loads(l) for l in f if l.strip()]
    holdout = examples[-HOLDOUT_N:]
    prompts = [format_prompt(ex) for ex in holdout]
    print(f"\nHoldout examples: {len(holdout)}")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Download adapters from GCS
    _download_gcs_prefix(GCS_SFT_ADAPTER_URI, SFT_LOCAL_PATH)
    _download_gcs_prefix(GCS_DPO_ADAPTER_URI, DPO_LOCAL_PATH)

    # ── Inference ──────────────────────────────────────────────────────────────
    print("\n[1/4] Base Llama-3-8B (untuned)")
    base_outputs = run_llama_inference("unsloth/llama-3-8b-bnb-4bit", prompts)

    print("\n[2/4] SFT LoRA Adapter")
    sft_outputs = run_llama_inference(SFT_LOCAL_PATH, prompts)

    print("\n[3/4] SFT+DPO LoRA Adapter")
    dpo_outputs = run_llama_inference(DPO_LOCAL_PATH, prompts)

    # ── Scoring ────────────────────────────────────────────────────────────────
    print("\n[4/4] Scoring with GPT-4o-as-judge ...")
    results = []

    for i, ex in enumerate(holdout):
        sid = ex.get("scenario_id", f"scenario_{i}")
        supplier = ex["input"]["inventory"][0].get("SupplierName", "Unknown") if ex["input"]["inventory"] else "Unknown"
        print(f"\n  [{i+1}/{len(holdout)}] {sid} ({supplier})")

        reference = ex["output"]["analysis"]
        prompt = prompts[i]

        def _score(text, label):
            valid, parsed = is_valid_json(text)
            if not valid:
                print(f"    {label}: 0/10 (invalid JSON — skipping judge)")
                return {"data_accuracy": 0, "reasoning_quality": 0, "avg": 0.0,
                        "json_valid": False, "comment": "invalid JSON", "preview": text[:200]}
            candidate_text = json.dumps(parsed, ensure_ascii=False, indent=2)
            s = score_with_gpt4o(client, prompt, reference, candidate_text, label)
            avg = round((s["data_accuracy"] + s["reasoning_quality"]) / 2, 1)
            print(f"    {label}: data={s['data_accuracy']}/10 reasoning={s['reasoning_quality']}/10 avg={avg}")
            time.sleep(0.5)  # Rate limit buffer
            return {**s, "avg": avg, "json_valid": True, "preview": text[:200]}

        # GPT-4o ground truth scored against itself as upper bound
        gpt4o_text = json.dumps(reference, ensure_ascii=False, indent=2)
        gpt4o_s = score_with_gpt4o(client, prompt, reference, gpt4o_text, "GPT-4o")
        gpt4o_avg = round((gpt4o_s["data_accuracy"] + gpt4o_s["reasoning_quality"]) / 2, 1)
        print(f"    GPT-4o: data={gpt4o_s['data_accuracy']}/10 reasoning={gpt4o_s['reasoning_quality']}/10 avg={gpt4o_avg}")
        time.sleep(0.5)

        results.append({
            "scenario_id": sid,
            "supplier": supplier,
            "base_llama": _score(base_outputs[i], "Base"),
            "sft":        _score(sft_outputs[i],  "SFT"),
            "sft_dpo":    _score(dpo_outputs[i],  "SFT+DPO"),
            "gpt4o":      {**gpt4o_s, "avg": gpt4o_avg, "json_valid": True, "preview": gpt4o_text[:200]},
        })

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    model_keys = ["base_llama", "sft", "sft_dpo", "gpt4o"]
    labels     = ["Base Llama", "SFT",      "SFT+DPO", "GPT-4o"]
    summary = {}

    for key, label in zip(model_keys, labels):
        avgs       = [r[key]["avg"] for r in results]
        valid_cnt  = sum(1 for r in results if r[key]["json_valid"])
        mean       = round(sum(avgs) / len(avgs), 2) if avgs else 0.0
        summary[key] = {"mean_avg_score": mean, "json_valid_count": valid_cnt}
        print(f"  {label:12s} | avg={mean:.1f}/10 | json_valid={valid_cnt}/{len(holdout)}")

    # ── W&B ───────────────────────────────────────────────────────────────────
    run_name = f"eval-dpo-comparison-{datetime.now().strftime('%Y%m%d-%H%M')}"
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config={"holdout_size": HOLDOUT_N, "dataset": DATASET_PATH},
    )
    wandb.log({f"{k}_mean_score": v["mean_avg_score"] for k, v in summary.items()})
    wandb.log({f"{k}_json_valid":  v["json_valid_count"] for k, v in summary.items()})

    cols = (["scenario_id", "supplier"] +
            [f"{k}_{m}" for k in model_keys for m in ["avg", "data_accuracy", "reasoning_quality", "json_valid"]])
    rows = []
    for r in results:
        row = [r["scenario_id"], r["supplier"]]
        for k in model_keys:
            row += [r[k]["avg"], r[k].get("data_accuracy", 0), r[k].get("reasoning_quality", 0), r[k]["json_valid"]]
        rows.append(row)
    wandb.log({"comparison_table": wandb.Table(columns=cols, data=rows)})
    wandb.finish()

    # ── Save to GCS ────────────────────────────────────────────────────────────
    local_file = "/tmp/eval_dpo_results.json"
    with open(local_file, "w", encoding="utf-8") as f:
        json.dump({"run_name": run_name, "summary": summary, "results": results},
                  f, ensure_ascii=False, indent=2)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    _upload_file_to_gcs(local_file, f"{GCS_EVAL_OUTPUT_URI}/eval_dpo_{timestamp}.json")

    print("\nDone!")
    return results


if __name__ == "__main__":
    run_comparison()

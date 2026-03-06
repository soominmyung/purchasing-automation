"""
Evaluation: Fine-tuned Llama-3-8B vs GPT-4o baseline.

Loads the last 5 examples from the teacher dataset as a holdout set,
runs both models on identical inputs, scores outputs, and logs results to W&B.

Usage (Vertex AI Custom Training Job):
    Override CMD with: ["python", "scripts/eval_sft.py"]
"""

import json
import os
from datetime import datetime
from pathlib import Path

import torch
import wandb
from google.cloud import storage as gcs

# ── Config ─────────────────────────────────────────────────────────────────────

GCS_ADAPTER_URI = os.environ.get(
    "GCS_ADAPTER_URI",
    "gs://purchasing-automation-models/sft-runs/lora_adapter",
)
GCS_EVAL_OUTPUT_URI = os.environ.get(
    "GCS_EVAL_OUTPUT_URI",
    "gs://purchasing-automation-models/eval-results",
)
DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    "training_data/teacher_dataset_20260302.jsonl",
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
WANDB_PROJECT  = os.environ.get("WANDB_PROJECT", "purchasing-automation-sft")
WANDB_ENTITY   = os.environ.get("WANDB_ENTITY", "msm1640-")

LOCAL_ADAPTER_PATH = "/app/lora_adapter"
HOLDOUT_SIZE = 5  # Last N examples used for evaluation


# ── Prompt template (must match train_sft.py) ──────────────────────────────────

PROMPT_TEMPLATE = """### Instruction
{instruction}

### Input
Supplier: {supplier}
Items: {items}
Supplier History: {supplier_history}
Item History: {item_history}

### Response
"""  # No {response} — model generates from here


def build_prompt(example: dict) -> str:
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


# ── GCS helper ─────────────────────────────────────────────────────────────────

def download_gcs_prefix(gcs_uri: str, local_dir: str) -> None:
    uri_parts   = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = uri_parts[0]
    prefix      = uri_parts[1] if len(uri_parts) > 1 else ""

    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    blobs  = list(bucket.list_blobs(prefix=prefix))

    Path(local_dir).mkdir(parents=True, exist_ok=True)
    for blob in blobs:
        rel = blob.name[len(prefix):].lstrip("/")
        if not rel:
            continue
        dest = Path(local_dir) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
        print(f"  ↓ {blob.name}")
    print(f"Downloaded {len(blobs)} files → {local_dir}")


def upload_file_to_gcs(local_file: str, gcs_uri: str) -> None:
    uri_parts   = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = uri_parts[0]
    blob_path   = uri_parts[1] if len(uri_parts) > 1 else Path(local_file).name

    client = gcs.Client()
    client.bucket(bucket_name).blob(blob_path).upload_from_filename(local_file)
    print(f"Uploaded → gs://{bucket_name}/{blob_path}")


# ── Llama inference ─────────────────────────────────────────────────────────────

def load_finetuned_model(adapter_path: str):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def run_llama_inference(model, tokenizer, prompt: str, max_new_tokens: int = 900) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the generated tokens (not the prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── GPT-4o baseline ────────────────────────────────────────────────────────────

def run_gpt4o_inference(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    # For GPT-4o we send the prompt as a user message
    system_msg = (
        "You are a purchasing analysis AI. "
        "Given the instruction and input data, generate the analysis JSON as described. "
        "Output ONLY the JSON object for the analysis field."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# ── Scoring ────────────────────────────────────────────────────────────────────

def is_valid_json(text: str) -> tuple[bool, dict | None]:
    """Try to parse JSON, handling code fences and trailing text (mirrors production _extract_json_from_text)."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    # Direct parse first
    try:
        return True, json.loads(text)
    except json.JSONDecodeError:
        pass
    # Extract first JSON object (handles over-generation / trailing text)
    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        try:
            return True, json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return False, None


def score_with_gpt4o(prompt: str, reference: dict, candidate: str, model_name: str) -> dict:
    """Use GPT-4o as judge to score a candidate response."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

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

    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    raw = resp.choices[0].message.content.strip()
    _, parsed = is_valid_json(raw)
    if parsed:
        return parsed
    return {"data_accuracy": 0, "reasoning_quality": 0, "comment": "parse error"}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Evaluation: Fine-tuned Llama-3-8B vs GPT-4o")
    print("=" * 60)

    # 0. Load dataset
    with open(DATASET_PATH, encoding="utf-8") as f:
        examples = [json.loads(l) for l in f if l.strip()]
    holdout = examples[-HOLDOUT_SIZE:]
    print(f"\nLoaded {len(examples)} examples, using last {HOLDOUT_SIZE} as holdout.")

    # 1. Download LoRA adapter
    print(f"\n[1/4] Downloading LoRA adapter from {GCS_ADAPTER_URI} ...")
    download_gcs_prefix(GCS_ADAPTER_URI, LOCAL_ADAPTER_PATH)

    # 2. Load fine-tuned model
    print("\n[2/4] Loading fine-tuned model ...")
    model, tokenizer = load_finetuned_model(LOCAL_ADAPTER_PATH)

    # 3. W&B init
    run_name = f"eval-llama3-vs-gpt4o-{datetime.now().strftime('%Y%m%d-%H%M')}"
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config={"holdout_size": HOLDOUT_SIZE, "dataset": DATASET_PATH},
    )

    # 4. Run evaluation
    print(f"\n[3/4] Running inference on {HOLDOUT_SIZE} examples ...")
    results = []

    for i, example in enumerate(holdout):
        print(f"\n--- Example {i+1}/{HOLDOUT_SIZE} ---")
        prompt   = build_prompt(example)
        reference = example["output"]["analysis"]

        # Fine-tuned Llama
        print("  Running Llama-3-8B (fine-tuned) ...")
        llama_output = run_llama_inference(model, tokenizer, prompt)
        llama_valid, llama_parsed = is_valid_json(llama_output)
        llama_scores = score_with_gpt4o(prompt, reference, llama_output, "Llama-3-8B SFT") if llama_valid else {"data_accuracy": 0, "reasoning_quality": 0, "comment": "invalid JSON"}

        # GPT-4o baseline
        print("  Running GPT-4o baseline ...")
        gpt4o_output = run_gpt4o_inference(prompt)
        gpt4o_valid, gpt4o_parsed = is_valid_json(gpt4o_output)
        gpt4o_scores = score_with_gpt4o(prompt, reference, gpt4o_output, "GPT-4o") if gpt4o_valid else {"data_accuracy": 0, "reasoning_quality": 0, "comment": "invalid JSON"}

        row = {
            "example_idx":           len(examples) - HOLDOUT_SIZE + i,
            "supplier":              example["input"]["inventory"][0]["SupplierName"] if example["input"]["inventory"] else "Unknown",
            "llama_json_valid":      llama_valid,
            "llama_data_accuracy":   llama_scores["data_accuracy"],
            "llama_reasoning":       llama_scores["reasoning_quality"],
            "llama_avg_score":       round((llama_scores["data_accuracy"] + llama_scores["reasoning_quality"]) / 2, 1),
            "gpt4o_json_valid":      gpt4o_valid,
            "gpt4o_data_accuracy":   gpt4o_scores["data_accuracy"],
            "gpt4o_reasoning":       gpt4o_scores["reasoning_quality"],
            "gpt4o_avg_score":       round((gpt4o_scores["data_accuracy"] + gpt4o_scores["reasoning_quality"]) / 2, 1),
            "llama_comment":         llama_scores.get("comment", ""),
            "llama_output_preview":  llama_output[:300],
            "gpt4o_output_preview":  gpt4o_output[:300],
        }
        results.append(row)

        print(f"  Llama: valid={llama_valid}, avg={row['llama_avg_score']}")
        print(f"  GPT-4o: valid={gpt4o_valid}, avg={row['gpt4o_avg_score']}")

    # 5. Log to W&B
    print("\n[4/4] Logging results to W&B ...")
    table = wandb.Table(columns=list(results[0].keys()), data=[list(r.values()) for r in results])
    wandb.log({"comparison_table": table})

    # Summary metrics
    llama_avg  = sum(r["llama_avg_score"]  for r in results) / len(results)
    gpt4o_avg  = sum(r["gpt4o_avg_score"]  for r in results) / len(results)
    llama_valid_pct = sum(r["llama_json_valid"] for r in results) / len(results) * 100
    wandb.log({
        "llama_mean_score":      round(llama_avg, 2),
        "gpt4o_mean_score":      round(gpt4o_avg, 2),
        "llama_json_valid_pct":  round(llama_valid_pct, 1),
        "score_gap":             round(gpt4o_avg - llama_avg, 2),
    })

    print(f"\nResults summary:")
    print(f"  Llama-3-8B SFT avg score : {llama_avg:.2f} / 10 (JSON valid: {llama_valid_pct:.0f}%)")
    print(f"  GPT-4o baseline avg score: {gpt4o_avg:.2f} / 10")
    print(f"  Gap: {gpt4o_avg - llama_avg:+.2f}")

    # 6. Save results JSON to GCS
    local_result_file = "/tmp/eval_results.json"
    with open(local_result_file, "w", encoding="utf-8") as f:
        json.dump({
            "run_name": run_name,
            "llama_mean_score": round(llama_avg, 2),
            "gpt4o_mean_score": round(gpt4o_avg, 2),
            "llama_json_valid_pct": round(llama_valid_pct, 1),
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    gcs_dest = f"{GCS_EVAL_OUTPUT_URI}/eval_{timestamp}.json"
    upload_file_to_gcs(local_result_file, gcs_dest)

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()

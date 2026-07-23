"""
DPO Preference Pair Generation.

Runs SFT model inference on all training examples to build preference pairs:
  chosen:   GPT-4o teacher output (ground truth)
  rejected: judge-verified worst-scoring SFT sample

For each example, N candidate completions are sampled from the SFT model
(temperature > 0 for diversity) and each is scored by a GPT-4o judge on
content only (data accuracy + reasoning quality — explicitly not style/tone).
The lowest-scoring candidate becomes `rejected`, so the preference signal
reflects a verified quality gap rather than a fixed "teacher vs. student" split.

Output: training_data/dpo_preference_pairs.jsonl
        gs://purchasing-automation-models/dpo-data/dpo_preference_pairs.jsonl

Usage (Vertex AI Custom Training Job):
    Override CMD with: ["python", "scripts/generate_dpo_pairs.py"]
"""

import json
import os
import re
from pathlib import Path

import torch
from google.cloud import storage as gcs

# ── Config ─────────────────────────────────────────────────────────────────────

GCS_ADAPTER_URI = os.environ.get(
    "GCS_ADAPTER_URI",
    "gs://purchasing-automation-models/sft-runs/lora_adapter",
)
GCS_OUTPUT_URI = os.environ.get(
    "GCS_OUTPUT_URI",
    "gs://purchasing-automation-models/dpo-data",
)
DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    "training_data/teacher_dataset_20260302.jsonl",
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OUTPUT_PATH = "training_data/dpo_preference_pairs.jsonl"
LOCAL_ADAPTER_PATH = "/app/lora_adapter"
HOLDOUT_SIZE = 5   # Last 5 examples are holdout — exclude from DPO training
N_CANDIDATES = int(os.environ.get("N_CANDIDATES", "4"))   # SFT samples per example
SAMPLE_TEMPERATURE = float(os.environ.get("SAMPLE_TEMPERATURE", "0.8"))


# ── Prompt template (must match train_sft.py) ──────────────────────────────────

PROMPT_TEMPLATE = """### Instruction
{instruction}

### Input
Supplier: {supplier}
Items: {items}
Supplier History: {supplier_history}
Item History: {item_history}

### Response
"""


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


# ── GCS helpers ────────────────────────────────────────────────────────────────

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
    print(f"Downloaded {len(blobs)} files → {local_dir}")


def upload_file_to_gcs(local_file: str, gcs_uri: str) -> None:
    uri_parts   = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = uri_parts[0]
    blob_path   = uri_parts[1] if len(uri_parts) > 1 else Path(local_file).name
    client = gcs.Client()
    client.bucket(bucket_name).blob(blob_path).upload_from_filename(local_file)
    print(f"Uploaded → gs://{bucket_name}/{blob_path}")


# ── Inference ──────────────────────────────────────────────────────────────────

def load_sft_model(adapter_path: str):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 900,
                   temperature: float = SAMPLE_TEMPERATURE) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── Judge scoring (content only — mirrors eval_sft.py's rubric) ───────────────

def is_valid_json(text: str) -> tuple[bool, dict | None]:
    """Try to parse JSON, handling code fences and trailing text."""
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


def score_with_gpt4o(prompt: str, reference: dict, candidate: str) -> dict:
    """Judge a candidate SFT completion on content only (not style/tone/verbosity)."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    judge_prompt = f"""You are evaluating a purchasing analysis AI response. Score it on two criteria (1-10 each).
Judge CONTENT ONLY — do not penalize or reward differences in writing style, tone, phrasing, or verbosity.

=== Original Input ===
{prompt}

=== Reference (GPT-4o ground truth) ===
{json.dumps(reference, ensure_ascii=False, indent=2)[:3000]}

=== Candidate Response ===
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


def avg_score(scores: dict) -> float:
    return (scores["data_accuracy"] + scores["reasoning_quality"]) / 2


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DPO Preference Pair Generation (judge-verified rejects)")
    print("=" * 60)

    # 1. Load dataset — exclude holdout
    with open(DATASET_PATH, encoding="utf-8") as f:
        examples = [json.loads(l) for l in f if l.strip()]
    train_examples = examples[:-HOLDOUT_SIZE]
    print(f"\nUsing {len(train_examples)} training examples (excluding last {HOLDOUT_SIZE} holdout)")

    # 2. Download SFT adapter
    print(f"\n[1/3] Downloading SFT adapter from {GCS_ADAPTER_URI} ...")
    download_gcs_prefix(GCS_ADAPTER_URI, LOCAL_ADAPTER_PATH)

    # 3. Load model
    print("\n[2/3] Loading SFT model ...")
    model, tokenizer = load_sft_model(LOCAL_ADAPTER_PATH)

    # 4. Generate pairs
    print(f"\n[3/3] Generating preference pairs (N={N_CANDIDATES} candidates/example) ...")
    pairs = []
    for i, example in enumerate(train_examples):
        supplier = example["input"]["inventory"][0].get("SupplierName", "?")
        print(f"\n  Example {i+1}/{len(train_examples)}: {supplier}")
        prompt    = build_prompt(example)
        reference = example["output"]["analysis"]
        chosen    = json.dumps(reference, ensure_ascii=False, indent=2)

        candidates = []
        for c in range(N_CANDIDATES):
            text = run_inference(model, tokenizer, prompt)
            valid, _ = is_valid_json(text)
            scores = (
                score_with_gpt4o(prompt, reference, text)
                if valid else {"data_accuracy": 0, "reasoning_quality": 0, "comment": "invalid JSON"}
            )
            candidates.append({"text": text, "valid": valid, "scores": scores, "avg": avg_score(scores)})
            print(f"    candidate {c+1}/{N_CANDIDATES}: valid={valid} avg={candidates[-1]['avg']:.1f}")

        worst = min(candidates, key=lambda c: c["avg"])
        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": worst["text"],
            "rejected_avg_score": worst["avg"],
            "rejected_comment": worst["scores"].get("comment", ""),
            "candidate_scores": [c["avg"] for c in candidates],
        })
        print(f"    → rejected: candidate with avg={worst['avg']:.1f} ({worst['scores'].get('comment', '')})")

    # 5. Save locally
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(pairs)} pairs → {OUTPUT_PATH}")

    # 6. Upload to GCS
    gcs_dest = f"{GCS_OUTPUT_URI}/dpo_preference_pairs.jsonl"
    upload_file_to_gcs(OUTPUT_PATH, gcs_dest)
    print("\nDone!")


if __name__ == "__main__":
    main()

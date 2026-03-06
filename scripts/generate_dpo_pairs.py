"""
DPO Preference Pair Generation.

Runs SFT model inference on all training examples to build preference pairs:
  chosen:   GPT-4o teacher output (ground truth)
  rejected: Llama SFT output (lower quality / invalid)

Output: training_data/dpo_preference_pairs.jsonl
        gs://purchasing-automation-models/dpo-data/dpo_preference_pairs.jsonl

Usage (Vertex AI Custom Training Job):
    Override CMD with: ["python", "scripts/generate_dpo_pairs.py"]
"""

import json
import os
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
OUTPUT_PATH = "training_data/dpo_preference_pairs.jsonl"
LOCAL_ADAPTER_PATH = "/app/lora_adapter"
HOLDOUT_SIZE = 5   # Last 5 examples are holdout — exclude from DPO training


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


def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 900) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DPO Preference Pair Generation")
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
    print(f"\n[3/3] Generating preference pairs ...")
    pairs = []
    for i, example in enumerate(train_examples):
        print(f"\n  Example {i+1}/{len(train_examples)}: {example['input']['inventory'][0].get('SupplierName', '?')}")
        prompt  = build_prompt(example)
        chosen  = json.dumps(example["output"]["analysis"], ensure_ascii=False, indent=2)
        rejected = run_inference(model, tokenizer, prompt)
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        print(f"    chosen:   {len(chosen)} chars")
        print(f"    rejected: {len(rejected)} chars")

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

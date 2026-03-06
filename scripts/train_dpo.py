"""
DPO (Direct Preference Optimization) training script for Llama-3-8B.
Runs AFTER SFT to further reduce hallucinations using preference pairs.

Preference pairs are built from:
- Chosen: High-scoring teacher outputs (8.5+ from Evaluation Agent)
- Rejected: Low-scoring outputs (re-generated with intentionally weaker prompts)

Package strategy: Use Unsloth + PatchDPOTrainer().
PatchDPOTrainer() must be called BEFORE importing DPOTrainer from trl.
This patches DPOTrainer to be compatible with Unsloth's CUDA kernels.

Usage:
    python scripts/train_dpo.py
"""

# IMPORTANT: Import Unsloth first and patch DPOTrainer BEFORE trl import
from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()  # Must be called before `from trl import DPOTrainer`

import json
import os
from datetime import datetime
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from google.cloud import storage as gcs
from trl import DPOTrainer, DPOConfig


# ── Configuration ──────────────────────────────────────────────────────────────

# Start from the SFT checkpoint (not base Llama)
SFT_MODEL_PATH = os.environ.get("SFT_MODEL_PATH", "models/llama3-purchasing-sft/lora_adapter")
# Reduce from 4096 to 2048: DPO does a dual forward pass (policy + reference)
# which roughly doubles VRAM usage vs SFT. 2048 fits comfortably on a T4 (16GB).
MAX_SEQ_LENGTH = 2048

# DPO-specific hyperparameters
BETA = 0.1           # KL penalty coefficient (lower = more aggressive preference learning)
EPOCHS = 3
BATCH_SIZE = 1       # Reduced from 2 → 1 (dual forward pass doubles VRAM per step)
GRADIENT_ACCUMULATION = 8  # Increased from 4 → 8 to keep effective batch = 1×8 = 8
LEARNING_RATE = 5e-5  # Lower LR than SFT for stability

# Paths
PREFERENCE_DATA_PATH = os.environ.get(
    "PREFERENCE_DATA_PATH",
    "training_data/dpo_preference_pairs.jsonl"
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "models/llama3-purchasing-dpo")

# W&B
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "purchasing-automation-dpo")
WANDB_RUN_NAME = f"dpo-llama3-8b-{datetime.now().strftime('%Y%m%d-%H%M')}"


# ── GCS Helpers ────────────────────────────────────────────────────────────────

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


def _download_gcs_file(gcs_uri: str, local_path: str) -> None:
    uri_parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name, blob_path = uri_parts[0], (uri_parts[1] if len(uri_parts) > 1 else "")
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    gcs.Client().bucket(bucket_name).blob(blob_path).download_to_filename(local_path)
    print(f"Downloaded → {local_path}")


def _upload_dir_to_gcs(local_dir: str, gcs_uri: str) -> None:
    uri_parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name, prefix = uri_parts[0], (uri_parts[1] if len(uri_parts) > 1 else "")
    bucket = gcs.Client().bucket(bucket_name)
    for local_file in Path(local_dir).rglob("*"):
        if local_file.is_file():
            rel = local_file.relative_to(local_dir)
            blob_path = f"{prefix}/{rel}" if prefix else str(rel)
            bucket.blob(blob_path).upload_from_filename(str(local_file))
    print(f"Uploaded → gs://{bucket_name}/{prefix}")


# ── Prompt Template (must match SFT) ──────────────────────────────────────────

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
    """Format the prompt (without response) for DPO pairs."""
    inp = example["input"]
    inventory = inp.get("inventory", [])
    # Use "SupplierName" key — matches generate_dpo_pairs.py, eval_sft.py, eval_dpo.py
    supplier = inventory[0].get("SupplierName", "Unknown") if inventory else "Unknown"

    return PROMPT_TEMPLATE.format(
        instruction=example.get("instruction", "Analyze the purchasing data."),
        supplier=supplier,
        items=json.dumps(inventory, ensure_ascii=False),
        supplier_history=inp.get("supplier_history", "No supplier history available."),
        item_history=inp.get("item_history", "No item history available."),
    )


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_preference_data(path: str) -> Dataset:
    """
    Load DPO preference pairs.
    Each entry has: prompt, chosen (high-quality response), rejected (low-quality response).
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                records.append({
                    "prompt": entry["prompt"],
                    "chosen": entry["chosen"],
                    "rejected": entry["rejected"],
                })

    dataset = Dataset.from_list(records)
    print(f"Loaded {len(dataset)} preference pairs from {path}")
    return dataset


# ── Training ───────────────────────────────────────────────────────────────────

def train_dpo():
    """Run DPO training starting from the SFT checkpoint."""

    # Load SFT model via Unsloth (memory-efficient 4-bit, patched for DPO)
    print(f"Loading SFT model from: {SFT_MODEL_PATH}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,       # auto-detect bf16/fp16
        load_in_4bit=True,
    )
    # SFT LoRA adapter is already embedded — no get_peft_model() needed

    # Load preference data
    dataset = load_preference_data(PREFERENCE_DATA_PATH)

    # Initialize W&B
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "sft_model": SFT_MODEL_PATH,
            "beta": BETA,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "dataset_size": len(dataset),
        },
    )

    # DPO training config
    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        # Explicitly cap sequence lengths to prevent OOM on T4 (16GB)
        # DPO concatenates prompt+chosen and prompt+rejected — these can get long
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=MAX_SEQ_LENGTH // 2,  # Leave room for response
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_strategy="epoch",
        optim="adamw_8bit",
        seed=42,
        report_to="wandb",
    )

    # Train (DPOTrainer is patched by PatchDPOTrainer() at module top)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Use implicit reference (peft model with adapter disabled)
        processing_class=tokenizer,  # TRL>=0.12: renamed from 'tokenizer'
        train_dataset=dataset,
        args=dpo_config,
    )

    print("Starting DPO training...")
    trainer.train()
    print("DPO training complete!")

    # Save
    adapter_path = os.path.join(OUTPUT_DIR, "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"DPO adapter saved to {adapter_path}")
    # Note: TRL auto-calls wandb.finish() after trainer.train() — do not call again


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Download SFT adapter from GCS (container storage is ephemeral)
    gcs_adapter_uri = os.environ.get("GCS_ADAPTER_URI", "")
    if gcs_adapter_uri:
        print(f"Downloading SFT adapter from {gcs_adapter_uri} ...")
        _download_gcs_prefix(gcs_adapter_uri, SFT_MODEL_PATH)

    # 2. Download DPO preference pairs from GCS
    gcs_pairs_uri = os.environ.get("GCS_PAIRS_URI", "")
    if gcs_pairs_uri:
        print(f"Downloading DPO pairs from {gcs_pairs_uri} ...")
        _download_gcs_file(gcs_pairs_uri, PREFERENCE_DATA_PATH)

    # 3. Run DPO training
    train_dpo()

    # 4. Upload DPO adapter to GCS
    gcs_output_uri = os.environ.get("GCS_OUTPUT_URI", "")
    if gcs_output_uri:
        adapter_path = os.path.join(OUTPUT_DIR, "lora_adapter")
        print(f"Uploading DPO adapter to {gcs_output_uri}/lora_adapter ...")
        _upload_dir_to_gcs(adapter_path, f"{gcs_output_uri}/lora_adapter")

    print("Done!")

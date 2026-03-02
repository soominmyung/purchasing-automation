"""
DPO (Direct Preference Optimization) training script for Llama-3-8B.
Runs AFTER SFT to further reduce hallucinations using preference pairs.

Preference pairs are built from:
- Chosen: High-scoring teacher outputs (8.5+ from Evaluation Agent)
- Rejected: Low-scoring outputs (re-generated with intentionally weaker prompts)

Usage:
    python scripts/train_dpo.py
"""

import json
import os
import sys
from datetime import datetime

import torch
import wandb
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel

# ── Configuration ──────────────────────────────────────────────────────────────

# Start from the SFT checkpoint (not base Llama)
SFT_MODEL_PATH = os.environ.get("SFT_MODEL_PATH", "models/llama3-purchasing-sft/lora_adapter")
MAX_SEQ_LENGTH = 4096

# DPO-specific hyperparameters
BETA = 0.1           # KL penalty coefficient (lower = more aggressive preference learning)
EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
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
    items_str = json.dumps(inp.get("inventory", []), ensure_ascii=False)
    supplier = inp.get("inventory", [{}])[0].get("supplier", "Unknown") if inp.get("inventory") else "Unknown"

    return PROMPT_TEMPLATE.format(
        instruction=example.get("instruction", "Analyze the purchasing data."),
        supplier=supplier,
        items=items_str,
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


# ── Build Preference Pairs ─────────────────────────────────────────────────────

def build_preference_pairs_from_teacher_data(
    high_score_path: str,
    low_score_path: str,
    output_path: str,
):
    """
    Build DPO preference pairs from two rounds of teacher data.
    High-score entries become 'chosen', low-score entries become 'rejected'.
    Pairs are matched by scenario index.
    """
    with open(high_score_path, "r", encoding="utf-8") as f:
        high_data = [json.loads(l) for l in f if l.strip()]

    with open(low_score_path, "r", encoding="utf-8") as f:
        low_data = [json.loads(l) for l in f if l.strip()]

    min_len = min(len(high_data), len(low_data))
    pairs = []

    for i in range(min_len):
        prompt = format_prompt(high_data[i])
        chosen = json.dumps(high_data[i]["output"].get("analysis", {}), ensure_ascii=False, indent=2)
        rejected = json.dumps(low_data[i]["output"].get("analysis", {}), ensure_ascii=False, indent=2)

        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Built {len(pairs)} preference pairs → {output_path}")
    return output_path


# ── Training ───────────────────────────────────────────────────────────────────

def train_dpo():
    """Run DPO training starting from the SFT checkpoint."""

    # Load SFT model
    print(f"Loading SFT model from: {SFT_MODEL_PATH}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

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
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_strategy="epoch",
        optim="adamw_8bit",
        seed=42,
        report_to="wandb",
    )

    # Train
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Use implicit reference (peft model)
        tokenizer=tokenizer,
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

    # Log to W&B
    artifact = wandb.Artifact(
        name="llama3-purchasing-dpo-lora",
        type="model",
        description="DPO-refined LoRA adapter for Llama-3-8B purchasing analysis",
    )
    artifact.add_dir(adapter_path)
    wandb.log_artifact(artifact)

    wandb.finish()
    print("Done!")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_dpo()

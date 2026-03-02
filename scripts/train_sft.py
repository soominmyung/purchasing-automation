"""
SFT (Supervised Fine-Tuning) script for Llama-3-8B.
Uses QLoRA (4-bit quantization + LoRA) via Unsloth for efficient training.
Integrates W&B for experiment tracking.

Usage (local):
    python scripts/train_sft.py

Usage (Vertex AI Custom Training):
    Packaged as a Docker container and submitted as a Vertex AI Custom Training Job.
    See scripts/vertex_ai_submit.py for job submission.
"""

import json
import os
import sys
from datetime import datetime

import torch
import wandb
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"  # Pre-quantized 4-bit Llama-3
MAX_SEQ_LENGTH = 4096
LORA_R = 16          # LoRA rank (higher = more capacity, more VRAM)
LORA_ALPHA = 32      # LoRA scaling factor
LORA_DROPOUT = 0.05

# Training hyperparameters
EPOCHS = 5
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4  # Effective batch = 2 * 4 = 8
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

# Paths
DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    "training_data/teacher_dataset_20260302.jsonl"
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "models/llama3-purchasing-sft")

# W&B
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "purchasing-automation-sft")
WANDB_RUN_NAME = f"sft-llama3-8b-{datetime.now().strftime('%Y%m%d-%H%M')}"


# ── Prompt Template ────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """### Instruction
{instruction}

### Input
Supplier: {supplier}
Items: {items}
Supplier History: {supplier_history}
Item History: {item_history}

### Response
{response}"""


def format_training_example(example: dict) -> str:
    """
    Convert a teacher dataset entry into the prompt template format.
    The model learns to generate the Response given Instruction + Input.
    """
    inp = example["input"]
    out = example["output"]

    # Build the response: Analysis JSON output (the core task)
    analysis = out.get("analysis", {})
    response = json.dumps(analysis, ensure_ascii=False, indent=2)

    items_str = json.dumps(inp.get("inventory", []), ensure_ascii=False)

    return PROMPT_TEMPLATE.format(
        instruction=example.get("instruction", "Analyze the purchasing data."),
        supplier=inp.get("inventory", [{}])[0].get("supplier", "Unknown") if inp.get("inventory") else "Unknown",
        items=items_str,
        supplier_history=inp.get("supplier_history", "No supplier history available."),
        item_history=inp.get("item_history", "No item history available."),
        response=response,
    )


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_dataset_from_jsonl(path: str) -> Dataset:
    """Load teacher dataset JSONL and convert to HuggingFace Dataset."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    formatted = [{"text": format_training_example(ex)} for ex in examples]
    dataset = Dataset.from_list(formatted)

    print(f"Loaded {len(dataset)} training examples from {path}")
    print(f"Sample (first 300 chars):\n{formatted[0]['text'][:300]}...")
    return dataset


# ── Model Setup ────────────────────────────────────────────────────────────────

def setup_model():
    """Load Llama-3-8B with 4-bit quantization and attach LoRA adapters."""
    print(f"Loading model: {MODEL_NAME}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect (float16 for T4, bfloat16 for A100)
        load_in_4bit=True,
    )

    # Attach LoRA adapters to key attention layers
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Memory-efficient
        random_state=42,
    )

    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


# ── Training ───────────────────────────────────────────────────────────────────

def train(model, tokenizer, dataset):
    """Run SFT training with W&B logging."""

    # Initialize W&B
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "model": MODEL_NAME,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "dataset_size": len(dataset),
        },
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_strategy="epoch",
        optim="adamw_8bit",
        seed=42,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        packing=True,  # Pack short examples together for efficiency
    )

    print("Starting SFT training...")
    trainer.train()
    print("Training complete!")

    return trainer


# ── Save & Upload ──────────────────────────────────────────────────────────────

def save_model(model, tokenizer):
    """Save the LoRA adapter (not the full model) for efficient storage."""
    adapter_path = os.path.join(OUTPUT_DIR, "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"LoRA adapter saved to {adapter_path}")

    # Log adapter to W&B as artifact
    artifact = wandb.Artifact(
        name="llama3-purchasing-sft-lora",
        type="model",
        description="QLoRA adapter for Llama-3-8B fine-tuned on purchasing analysis",
    )
    artifact.add_dir(adapter_path)
    wandb.log_artifact(artifact)
    print("LoRA adapter logged to W&B as artifact.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SFT Training: Llama-3-8B for Purchasing Analysis")
    print("=" * 60)

    # 1. Load data
    dataset = load_dataset_from_jsonl(DATASET_PATH)

    # 2. Setup model with QLoRA
    model, tokenizer = setup_model()

    # 3. Train
    trainer = train(model, tokenizer, dataset)

    # 4. Save LoRA adapter
    save_model(model, tokenizer)

    # 5. Cleanup
    wandb.finish()
    print("Done! LoRA adapter ready for deployment.")


if __name__ == "__main__":
    main()

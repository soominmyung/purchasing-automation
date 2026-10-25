"""
SFT of the v5 analysis agent on Gemma 4 12B (docs/sft_plan.md, section 4), QLoRA with Unsloth on
the RTX 5090, run inside the Unsloth container (scripts/train_learning_curve.ps1).

The training examples are the teacher runs that passed every blocking check
(training_data/sft_v5/teacher_scenarios.jsonl, "train": true), as the exact system and user
messages the pipeline sends and the raw answer, in Gemma 4's chat template with thinking off. The
loss is on the answer only. A tenth of the passing runs is held out for validation loss, which
chooses the checkpoint; the 25 evaluation cases are never used here.

Learning curve: --suppliers N trains on the first N of a fixed, type-balanced order of the
training runs (25, 50, then all), so each size contains the smaller ones.

Run (in the container): python scripts/train_sft_v5.py --suppliers 100 --seed 1
Out: /runs/<name>/adapter (LoRA), /runs/<name>/train_log.json
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from unsloth import FastModel  # first, before transformers

import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parent.parent
TEACHER = ROOT / "training_data" / "sft_v5" / "teacher_scenarios.jsonl"
BASE = "google/gemma-4-12B-it"
MAX_LEN = 12288


def ordered_runs(seed_for_split: int = 0) -> tuple[list[dict], list[dict]]:
    """Passing runs split into validation (a tenth, type-balanced) and a fixed training order that
    interleaves the types, so any prefix is balanced."""
    rows = [json.loads(l) for l in open(TEACHER, encoding="utf-8")]
    rows = [r for r in rows if r.get("train")]
    by_type = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["scenario_id"]):
        by_type[r["category"]].append(r)
    rnd = random.Random(seed_for_split)
    val, train_by_type = [], {}
    for t, rs in by_type.items():
        rnd.shuffle(rs)
        k = max(1, round(len(rs) / 10)) if len(rs) >= 4 else 0
        val += rs[:k]
        train_by_type[t] = rs[k:]
    order, i = [], 0
    while any(train_by_type.values()):
        for t in sorted(train_by_type):
            if train_by_type[t]:
                order.append(train_by_type[t].pop(0))
        i += 1
    if not val:  # too few runs per type (a smoke test): hold out the last two
        order, val = order[:-2], order[-2:]
    return order, val


def encode(tokenizer, run: dict) -> dict:
    msgs = run["messages"]
    prompt = tokenizer.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    if not full.startswith(prompt):  # the template renders the answer turn differently: fall back
        full = prompt + msgs[2]["content"] + "<turn|>\n"
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suppliers", type=int, required=True, help="training runs to use (0 = all)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", default="/runs")
    a = ap.parse_args()

    order, val = ordered_runs()
    train_runs = order if a.suppliers in (0, None) else order[: a.suppliers]
    name = f"gemma4-12b_n{len(train_runs)}_s{a.seed}"
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastModel.from_pretrained(BASE, max_seq_length=MAX_LEN, load_in_4bit=True, dtype=None)
    tok = getattr(tokenizer, "tokenizer", tokenizer)  # a processor wraps the text tokenizer
    model = FastModel.get_peft_model(
        model, r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0, bias="none",
        finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        use_gradient_checkpointing="unsloth", random_state=a.seed)

    train_ds = Dataset.from_list([encode(tok, r) for r in train_runs])
    val_ds = Dataset.from_list([encode(tok, r) for r in val])
    lengths = [len(x) for x in train_ds["input_ids"]]
    too_long = sum(l > MAX_LEN for l in lengths)
    print(f"{name}: {len(train_ds)} train, {len(val_ds)} validation; tokens max {max(lengths)}, "
          f"mean {sum(lengths) // len(lengths)}; over {MAX_LEN}: {too_long}", flush=True)
    assert too_long == 0, "an example would be cut"

    steps_per_epoch = max(1, len(train_ds) // 8)
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=a.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8, per_device_eval_batch_size=1,
        learning_rate=a.lr, warmup_steps=max(1, round(0.1 * steps_per_epoch * a.epochs)), weight_decay=0.01, lr_scheduler_type="cosine",
        bf16=True, logging_steps=1, eval_strategy="steps", eval_steps=steps_per_epoch,
        save_strategy="steps", save_steps=steps_per_epoch, save_total_limit=4,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        optim="adamw_8bit", seed=a.seed, report_to="none", remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    started = time.time()
    trainer.train()
    minutes = round((time.time() - started) / 60, 1)
    model.save_pretrained(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))
    log = {"name": name, "base": BASE, "train_runs": [r["scenario_id"] for r in train_runs],
           "validation_runs": [r["scenario_id"] for r in val], "seed": a.seed, "epochs": a.epochs,
           "lr": a.lr, "rank": a.rank, "alpha": 2 * a.rank, "max_len": MAX_LEN, "minutes": minutes,
           "best_checkpoint": trainer.state.best_model_checkpoint, "best_eval_loss": trainer.state.best_metric,
           "history": trainer.state.log_history, "gpu": torch.cuda.get_device_name(0)}
    (out / "train_log.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    print(f"done {name}: {minutes} min, best eval loss {trainer.state.best_metric}", flush=True)


if __name__ == "__main__":
    main()

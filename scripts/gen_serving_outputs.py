"""
Generate analysis outputs from the served model, for the quantization quality check.

Builds the same prompt as eval_sft.py so results stay comparable, then fires all
prompts concurrently (vLLM batches them, so 30 scenarios take ~a minute instead of ~20).
Run it once against the FP16 deployment and once against the FP8 one, then compare the
two files with eval_quantization_quality.py.

Usage:
    python scripts/gen_serving_outputs.py training_data/teacher_dataset_20260302.jsonl \
        fp16_all.json 30
"""

import concurrent.futures as cf
import json
import sys
import urllib.request

# ── Config ─────────────────────────────────────────────────────────────────────

ENDPOINT = "http://localhost:8000/v1/completions"
MODEL = "ft"  # LoRA module name registered with --lora-modules

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


def generate(prompt: str, max_tokens: int = 900) -> str:
    body = json.dumps({"model": MODEL, "prompt": prompt,
                       "max_tokens": max_tokens, "temperature": 0.1}).encode()
    req = urllib.request.Request(ENDPOINT, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["text"]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    dataset_path, out_path = sys.argv[1], sys.argv[2]
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 16

    examples = [json.loads(l) for l in open(dataset_path, encoding="utf-8") if l.strip()]
    selected = examples[-count:]
    base_idx = len(examples) - count

    def work(i, e):
        return i, e["output"]["analysis"], generate(build_prompt(e))

    results = [None] * len(selected)
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, i, e) for i, e in enumerate(selected)]
        done = 0
        for f in cf.as_completed(futures):
            i, ref, out = f.result()
            results[i] = {"idx": base_idx + i, "reference": ref, "output": out}
            done += 1
            print(f"done {done}/{len(selected)}", flush=True)

    json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved", out_path, flush=True)


if __name__ == "__main__":
    main()

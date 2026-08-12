"""
Baseline: naive HuggingFace transformers generate() loop (no batching engine).

This is the "before" number the vLLM benchmark is compared against — the same
inference path eval_sft.py uses: load FP16 base + LoRA adapter with peft, then
process requests ONE AT A TIME. Run it with the vLLM container stopped so the
GPU is free.

Usage:
    python scripts/bench_hf_baseline.py --base unsloth/llama-3-8b \
        --adapter ~/lora_adapter --requests 8 --max-tokens 128
"""

import argparse
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ─────────────────────────────────────────────────────────────────────

# Identical prompt to bench_vllm.py so the two numbers are comparable.
PROMPT = """### Instruction
Analyze the purchasing data and produce a JSON analysis report.

### Input
Supplier: Acme Components Ltd
Items: [{"ItemCode":"AB-100","ItemName":"M4 bolt","CurrentStock":120,"WksToOOS":2.5,"RiskLevel":"High"}]
Supplier History: Two late deliveries in the last quarter; renegotiated lead time to 3 weeks.
Item History: Demand steady; no quality issues recorded.

### Response
"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/llama-3-8b")
    ap.add_argument("--adapter", default=None, help="path to the LoRA adapter dir (optional)")
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    print(f"Loading base {args.base} (fp16) ...")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="cuda",
    )
    if args.adapter:
        print(f"Attaching LoRA adapter {args.adapter} ...")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    inputs = tok(PROMPT, return_tensors="pt").to("cuda")

    # Warm up so the first call's kernel/graph setup isn't counted.
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=8, do_sample=False,
                       pad_token_id=tok.eos_token_id)

    print(f"Running {args.requests} SEQUENTIAL requests (no batching) ...")
    total_out = 0
    t0 = time.perf_counter()
    for _ in range(args.requests):
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_tokens,
                                 temperature=0.1, do_sample=True,
                                 pad_token_id=tok.eos_token_id)
        total_out += out[0][inputs["input_ids"].shape[1]:].shape[0]
    wall = time.perf_counter() - t0

    print("\n=== HF baseline (sequential generate) ===")
    print(f"requests        : {args.requests}")
    print(f"wall_sec        : {wall:.2f}")
    print(f"throughput_tok_s: {total_out / wall:.1f}")
    print(f"avg_latency_s   : {wall / args.requests:.2f}")


if __name__ == "__main__":
    main()

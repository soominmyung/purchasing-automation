"""
Benchmark: vLLM serving throughput / latency across concurrency levels.

Fires N requests at each concurrency level against an OpenAI-compatible endpoint
(the vLLM container) and reports aggregate output tokens/sec plus p50/p95 latency.
Throughput rising while latency stays flat is the signature of continuous batching.

Serve the model first (FP16; drop --quantization for the FP16 run):

    docker run -d --gpus all -p 8000:8000 --ipc=host \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      -v ~/lora_adapter:/lora_adapter \
      vllm/vllm-openai:latest \
      --model unsloth/llama-3-8b --enable-lora --lora-modules ft=/lora_adapter \
      --max-lora-rank 16 --max-model-len 4096 --gpu-memory-utilization 0.90 \
      --quantization fp8

Usage:
    python scripts/bench_vllm.py --model ft --concurrency 1,2,4,8,16,32 \
        --requests-per-level 16 --max-tokens 128
"""

import argparse
import concurrent.futures as cf
import json
import statistics
import time

from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────────

# A representative purchasing-analysis prompt (mirrors the production workload shape).
PROMPT = """### Instruction
Analyze the purchasing data and produce a JSON analysis report.

### Input
Supplier: Acme Components Ltd
Items: [{"ItemCode":"AB-100","ItemName":"M4 bolt","CurrentStock":120,"WksToOOS":2.5,"RiskLevel":"High"}]
Supplier History: Two late deliveries in the last quarter; renegotiated lead time to 3 weeks.
Item History: Demand steady; no quality issues recorded.

### Response
"""


# ── Benchmark ──────────────────────────────────────────────────────────────────

def one_request(client: OpenAI, model: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    resp = client.completions.create(
        model=model, prompt=PROMPT, max_tokens=max_tokens, temperature=0.1,
    )
    dt = time.perf_counter() - t0
    out_tokens = resp.usage.completion_tokens if resp.usage else max_tokens
    return {"latency": dt, "out_tokens": out_tokens}


def run_level(client, model, concurrency, n_requests, max_tokens) -> dict:
    """Fire n_requests with `concurrency` in flight; measure aggregate throughput."""
    results = []
    wall_start = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one_request, client, model, max_tokens) for _ in range(n_requests)]
        for f in cf.as_completed(futures):
            results.append(f.result())
    wall = time.perf_counter() - wall_start

    latencies = sorted(r["latency"] for r in results)
    total_out = sum(r["out_tokens"] for r in results)
    return {
        "concurrency": concurrency,
        "requests": n_requests,
        "wall_sec": round(wall, 2),
        "throughput_tok_s": round(total_out / wall, 1),
        "p50_latency_s": round(statistics.median(latencies), 2),
        "p95_latency_s": round(latencies[int(len(latencies) * 0.95) - 1], 2),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")  # vLLM ignores the key by default
    ap.add_argument("--model", required=True, help="model id as registered by the server (e.g. ft)")
    ap.add_argument("--concurrency", default="1,2,4,8,16,32")
    ap.add_argument("--requests-per-level", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", default="bench_results.json")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    levels = [int(x) for x in args.concurrency.split(",")]

    print(f"\nBenchmarking {args.model} @ {args.base_url}")
    print(f"{'conc':>5} {'reqs':>5} {'wall(s)':>8} {'tok/s':>8} {'p50(s)':>7} {'p95(s)':>7}")
    print("-" * 46)

    rows = []
    for c in levels:
        row = run_level(client, args.model, c, args.requests_per_level, args.max_tokens)
        rows.append(row)
        print(f"{row['concurrency']:>5} {row['requests']:>5} {row['wall_sec']:>8} "
              f"{row['throughput_tok_s']:>8} {row['p50_latency_s']:>7} {row['p95_latency_s']:>7}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "base_url": args.base_url, "rows": rows}, f, indent=2)
    print(f"\nSaved -> {args.out}")

    peak = max(rows, key=lambda r: r["throughput_tok_s"])
    print(f"Peak throughput: {peak['throughput_tok_s']} tok/s at concurrency {peak['concurrency']}")


if __name__ == "__main__":
    main()

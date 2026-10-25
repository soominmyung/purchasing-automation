"""
Serving throughput on the production workload: the v5 analysis request (the production system
prompt and the pipeline's user JSON for the evaluation scenarios), sent at several concurrency
levels to a vLLM server. Reports output tokens/s across all requests and p50/p95 latency per
level. Rising throughput with flat latency is continuous batching at work.

scripts/bench_vllm.py measured the Llama-3-8B study with a short placeholder prompt and 128 output
tokens. This one uses the real request (about 4,800 prompt tokens, answers up to max_tokens).

Run (server on :8000): python scripts/bench_vllm_v5.py --model gemma-4-12b-it --label bf16
Out: eval_results/serving_bench_<label>.json
"""
import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402

from scripts.gen_eval_outputs_gpt4o import SNAPSHOT_DATE, load_scenarios, preprocess  # noqa: E402
from services.agents import analysis_history, history_for_model  # noqa: E402
from services.prompts import ANALYSIS_AGENT_SYSTEM  # noqa: E402


def requests() -> list[list[dict]]:
    out = []
    for s in load_scenarios():
        g = preprocess(s)
        inp = {"snapshot_date": SNAPSHOT_DATE, "supplier": g["supplier"], "items": g["items"]}
        docs, _ = analysis_history(inp, s["history"])
        user = json.dumps({**inp, "history": history_for_model(docs)}, ensure_ascii=False)
        out.append([{"role": "system", "content": ANALYSIS_AGENT_SYSTEM}, {"role": "user", "content": user}])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--concurrency", default="1,2,4,8,16")
    ap.add_argument("--max-tokens", type=int, default=1500)
    a = ap.parse_args()
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY", timeout=900)
    msgs = requests()

    def one(m):
        t = time.time()
        r = client.chat.completions.create(model=a.model, messages=m, temperature=0, max_tokens=a.max_tokens,
                                           extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return time.time() - t, r.usage.completion_tokens, r.usage.prompt_tokens

    one(msgs[0])  # warm-up
    levels = []
    for c in [int(x) for x in a.concurrency.split(",")]:
        batch = (msgs * 4)[: max(c * 2, 8)]
        t0 = time.time()
        with ThreadPoolExecutor(c) as ex:
            res = list(ex.map(one, batch))
        wall = time.time() - t0
        lat = sorted(x[0] for x in res)
        out_tok = sum(x[1] for x in res)
        levels.append({"concurrency": c, "requests": len(batch), "output_tokens": out_tok,
                       "prompt_tokens_mean": round(statistics.mean(x[2] for x in res)),
                       "throughput_tok_s": round(out_tok / wall, 1), "p50_s": round(lat[len(lat) // 2], 1),
                       "p95_s": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 1), "wall_s": round(wall, 1)})
        print(json.dumps(levels[-1]), flush=True)
    out = ROOT / "eval_results" / f"serving_bench_{a.label}.json"
    out.write_text(json.dumps({"model": a.model, "label": a.label, "max_tokens": a.max_tokens,
                               "levels": levels}, indent=2), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()

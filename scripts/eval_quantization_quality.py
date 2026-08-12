"""
Quality check: does FP8 quantization degrade the served model's output?

Compares two files produced by gen_serving_outputs.py (one from the FP16 deployment,
one from FP8) with GPT-4o as judge, in two complementary ways:

  scored   — each output scored 1-10 on data accuracy + reasoning against the GPT-4o
             ground truth, the same rubric as eval_sft.py
  pairwise — the two outputs compared head to head for the same scenario. Each pair is
             judged twice with the order swapped, and a side only wins the item if it
             wins both orderings, which cancels the judge's position bias.

All examples can be used (no unseen holdout needed): FP16 and FP8 are the same weights,
so any train-set familiarity applies equally to both and cancels out.

Requires OPENAI_API_KEY in the environment.

Usage:
    python scripts/eval_quantization_quality.py fp16_all.json fp8_all.json --mode scored
    python scripts/eval_quantization_quality.py fp16_all.json fp8_all.json --mode pairwise
"""

import argparse
import json
import os
import re

from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o")

client = OpenAI(api_key=OPENAI_API_KEY)


def is_valid_json(text: str) -> bool:
    """Mirrors production _extract_json_from_text: tolerate code fences / trailing text."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        m = re.search(r"(\{[\s\S]*\})", text)
        if m:
            try:
                json.loads(m.group(1))
                return True
            except json.JSONDecodeError:
                return False
    return False


def ask_judge(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=JUDGE_MODEL, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


# ── Mode 1: absolute scoring against the ground truth ──────────────────────────

def score_one(reference: dict, candidate: str) -> dict:
    raw = ask_judge(f"""You are evaluating a purchasing analysis AI response. Score it on two criteria (1-10 each).

=== Reference (GPT-4o ground truth) ===
{json.dumps(reference, ensure_ascii=False, indent=2)[:3000]}

=== Candidate Response ===
{candidate[:3000]}

Evaluate:
1. data_accuracy (1-10): Does the candidate correctly reference the supplier names, item codes, stock levels, and risk levels?
2. reasoning_quality (1-10): Is the replenishment analysis and are the critical questions logically sound and relevant?

Respond in JSON only:
{{"data_accuracy": <int>, "reasoning_quality": <int>, "comment": "<one sentence>"}}""")
    m = re.search(r"(\{[\s\S]*\})", raw)
    try:
        return json.loads(m.group(1))
    except Exception:
        return {"data_accuracy": 0, "reasoning_quality": 0, "comment": "parse error"}


def run_scored(path: str, label: str) -> dict:
    data = json.load(open(path, encoding="utf-8"))
    rows = []
    for r in data:
        valid = is_valid_json(r["output"])
        s = score_one(r["reference"], r["output"]) if valid else \
            {"data_accuracy": 0, "reasoning_quality": 0, "comment": "invalid JSON"}
        avg = (s["data_accuracy"] + s["reasoning_quality"]) / 2
        rows.append({"idx": r["idx"], "valid_json": valid, "avg": avg, **s})
        print(f"  [{label}] idx {r['idx']}: valid={valid} avg={avg}")
    return {
        "label": label,
        "mean_score": round(sum(x["avg"] for x in rows) / len(rows), 2),
        "valid_json_pct": round(100 * sum(x["valid_json"] for x in rows) / len(rows), 1),
        "rows": rows,
    }


# ── Mode 2: order-swapped pairwise comparison ──────────────────────────────────

def judge_pair(a_text: str, b_text: str) -> str:
    """Return 'A', 'B', or 'tie'."""
    raw = ask_judge(f"""Two AI systems produced a purchasing-analysis response to the SAME input.
Judge which is better on: data accuracy, reasoning quality, and well-formed structured (JSON) output.

=== Response A ===
{a_text[:3500]}

=== Response B ===
{b_text[:3500]}

Answer with JSON only: {{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}}""")
    m = re.search(r'"winner"\s*:\s*"(A|B|tie)"', raw)
    return m.group(1) if m else "tie"


def run_pairwise(fp16_path: str, fp8_path: str) -> dict:
    fp16 = {r["idx"]: r["output"] for r in json.load(open(fp16_path, encoding="utf-8"))}
    fp8 = {r["idx"]: r["output"] for r in json.load(open(fp8_path, encoding="utf-8"))}

    fp16_wins = fp8_wins = ties = 0
    detail = []
    for idx in sorted(set(fp16) & set(fp8)):
        v1 = judge_pair(fp16[idx], fp8[idx])   # order 1: A=fp16
        v2 = judge_pair(fp8[idx], fp16[idx])   # order 2: A=fp8
        if v1 == "A" and v2 == "B":
            winner = "FP16"; fp16_wins += 1
        elif v1 == "B" and v2 == "A":
            winner = "FP8"; fp8_wins += 1
        else:
            winner = "tie"; ties += 1          # inconsistent across orderings => tie
        detail.append({"idx": idx, "v1": v1, "v2": v2, "winner": winner})
        print(f"  idx {idx}: order1={v1} order2={v2} -> {winner}", flush=True)

    return {"n": len(detail), "fp16_wins": fp16_wins, "fp8_wins": fp8_wins,
            "ties": ties, "detail": detail}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fp16_outputs")
    ap.add_argument("fp8_outputs")
    ap.add_argument("--mode", choices=["scored", "pairwise"], default="scored")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY is not set.")

    if args.mode == "scored":
        print("=== Scoring FP16 ===")
        fp16 = run_scored(args.fp16_outputs, "FP16")
        print("=== Scoring FP8 ===")
        fp8 = run_scored(args.fp8_outputs, "FP8")

        print("\n================ QUALITY COMPARISON ================")
        print(f"{'model':<8} {'mean/10':>8} {'valid_json%':>12}")
        for r in (fp16, fp8):
            print(f"{r['label']:<8} {r['mean_score']:>8} {r['valid_json_pct']:>11}%")
        print(f"\nQuality delta (FP8 - FP16): {fp8['mean_score'] - fp16['mean_score']:+.2f} / 10")
        result = {"fp16": fp16, "fp8": fp8}
    else:
        result = run_pairwise(args.fp16_outputs, args.fp8_outputs)
        print(f"\n============ PAIRWISE RESULT (n={result['n']}) ============")
        print(f"FP16 wins : {result['fp16_wins']}")
        print(f"FP8  wins : {result['fp8_wins']}")
        print(f"ties      : {result['ties']}")
        print("\nBalanced wins/ties => no systematic quality regression from FP8.")

    out = args.out or f"quality_{args.mode}.json"
    json.dump(result, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

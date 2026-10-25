"""
SFT base-model screening (docs/sft_plan.md, section 5): every candidate, untuned, as the analysis
agent on the 25 evaluation scenarios, with the production prompt (v5) and the production checks
(code checks C1-C10 and the judge, as in a run), against gpt-6.1-sol's outputs on the same prompt.

Each candidate is served by vLLM on the RTX 5090 (scripts/serve_local_model.ps1) and called as
"local:<served name>" through services/agents._llm, so the analysis goes through exactly the
production code path (run_analysis_agent). No rewrites: the first answer is what is screened.

Run:
  python scripts/screen_base_models.py --model local:qwen3.5-9b      one candidate (resumes)
  python scripts/screen_base_models.py --sol                         score Sol's outputs the same way
  python scripts/screen_base_models.py --report                      the table of every file so far
  python scripts/screen_base_models.py --model gpt-6.1-sol --tag fac-rule   a fresh run (Sol at its
      production effort) after a prompt revision, kept apart from the earlier files
Out: eval_results/sft_screen_<name>.jsonl, one line per scenario
"""
import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from scripts.gen_eval_outputs_gpt4o import SNAPSHOT_DATE, load_scenarios, preprocess  # noqa: E402
from services import agents  # noqa: E402
from services.prompts import ANALYSIS_AGENT_SYSTEM, PROMPT_VERSION  # noqa: E402
from services.quality_checks import evaluate_analysis  # noqa: E402

RESULTS = ROOT / "eval_results"
# Sol on the same prompt (2026-10-25), the outputs every other agent's comparison was built on
SOL_OUTPUTS = RESULTS / "v5_base_outputs_estimate.jsonl"


TAG = ""  # --tag: a suffix for a run on a revised prompt, e.g. "fac-rule"


def out_file(name: str) -> Path:
    return RESULTS / f"sft_screen_{name.removeprefix('local:').replace('/', '_')}{'_' + TAG if TAG else ''}.jsonl"


def _done(path: Path) -> set[str]:
    return {json.loads(l)["scenario_id"] for l in open(path, encoding="utf-8") if l.strip()} if path.exists() else set()


def _score(group: dict, analysis: dict, docs: list[dict]) -> dict:
    if not isinstance(analysis, dict) or "unparsed_output" in analysis:
        return {"verdict": "unparsed", "failed": ["C1"], "warnings": [], "checks": {}, "unparsed": True}
    ev = evaluate_analysis(group["supplier"], group["items"], analysis, docs, settings.evaluator_model,
                           snapshot_date=SNAPSHOT_DATE)
    return {**ev, "unparsed": False}


def run_model(model: str) -> None:
    agents.AGENT_MODELS["analysis"] = model
    if model.startswith("local:"):  # an API model keeps its production effort (Sol: medium)
        agents.AGENT_EFFORT["analysis"] = None
    path = out_file(model)
    done = _done(path)
    with open(path, "a", encoding="utf-8") as f:
        for s in load_scenarios():
            if s["scenario_id"] in done:
                continue
            group = preprocess(s)
            input_json = {"snapshot_date": SNAPSHOT_DATE, "supplier": group["supplier"], "items": group["items"]}
            started = time.time()
            try:
                analysis, docs = agents.run_analysis_agent(input_json, history_override=s["history"])
                error = None
            except Exception as e:  # a timeout or a server error is a result too
                analysis, docs, error = {"unparsed_output": ""}, s["history"], f"{type(e).__name__}: {e}"[:300]
            seconds = round(time.time() - started, 1)
            score = _score(group, analysis, docs)
            f.write(json.dumps({
                "scenario_id": s["scenario_id"], "category": s.get("category"), "labels": s.get("labels"),
                "model": model, "prompt_version": PROMPT_VERSION, "prompt": ANALYSIS_AGENT_SYSTEM,
                "generated_at": datetime.now().isoformat(timespec="seconds"), "analysis_seconds": seconds,
                "error": error, "input": input_json, "history_docs": docs, "analysis": analysis, "evaluation": score,
            }, ensure_ascii=False, default=str) + "\n")
            f.flush()
            print(f"  {s['scenario_id']:<8} {seconds:>6}s  {score['verdict']}  {', '.join(score['failed'])}", flush=True)
    print(f"-> {path}")


def run_sol() -> None:
    path = out_file("gpt-6.1-sol")
    done = _done(path)
    by_id = {s["scenario_id"]: s for s in load_scenarios()}
    with open(path, "a", encoding="utf-8") as f:
        for line in open(SOL_OUTPUTS, encoding="utf-8"):
            r = json.loads(line)
            if r["scenario_id"] in done:
                continue
            group = preprocess(by_id[r["scenario_id"]])
            score = _score(group, r["analysis"], r["history_docs"])
            f.write(json.dumps({
                "scenario_id": r["scenario_id"], "category": r.get("category"), "labels": r.get("labels"),
                "model": "gpt-6.1-sol", "prompt_version": r["prompt_version"], "prompt": r["prompts"]["analysis"],
                "generated_at": r["generated_at"], "analysis_seconds": r["analysis_seconds"], "error": None,
                "input": r["input"], "history_docs": r["history_docs"], "analysis": r["analysis"], "evaluation": score,
            }, ensure_ascii=False, default=str) + "\n")
            f.flush()
            print(f"  {r['scenario_id']:<8} {score['verdict']}  {', '.join(score['failed'])}", flush=True)
    print(f"-> {path}")


def report() -> None:
    print(f"| Model | Parsed | Ready (all blocking passed) | Blocking failures by check | Time per analysis |")
    print("|---|---|---|---|---|")
    for path in sorted(RESULTS.glob("sft_screen_*.jsonl")):
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        ev = [r["evaluation"] for r in rows]
        fails = Counter(c for e in ev for c in e["failed"])
        secs = sorted(r["analysis_seconds"] for r in rows)
        print(f"| {rows[0]['model']} | {sum(not e['unparsed'] for e in ev)}/{len(rows)} | "
              f"{sum(e['verdict'] == 'ready' for e in ev)}/{len(rows)} | "
              f"{', '.join(f'{k} {v}' for k, v in fails.most_common()) or '-'} | "
              f"{secs[len(secs) // 2]} s median |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--sol", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    TAG = a.tag
    if a.model:
        run_model(a.model)
    if a.sol:
        run_sol()
    if a.report:
        report()

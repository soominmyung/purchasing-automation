"""
Teacher data for the SFT retraining: gpt-6.1-sol at medium effort
as the production analysis agent on each generated scenario, scored by the production checks
(code checks C1-C10 and the judge). Every run is kept with the exact messages sent and the raw
text the model returned, so the training example is exactly what the pipeline sends and receives.
Only first answers that pass every blocking check become training examples; the rest are kept
for DPO pairs. Runs in parallel.

Run:
  python scripts/build_sft_teacher_data.py                         the training scenarios
  python scripts/build_sft_teacher_data.py --scenarios training_data/sft_v5/generalisation_tie_in.jsonl
Out: training_data/sft_v5/teacher_<scenario file stem>.jsonl (resumes)
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from config import settings  # noqa: E402
from scripts.eval_scenarios import SNAPSHOT_DATE, preprocess  # noqa: E402
from services.agents import (_extract_json_from_text, _llm, analysis_history, document_references,  # noqa: E402
                             from_model, history_for_model)
from services.prompts import ANALYSIS_AGENT_SYSTEM, PROMPT_VERSION  # noqa: E402
from services.quality_checks import evaluate_analysis  # noqa: E402

TEACHER, EFFORT = "gpt-6.1-sol", "medium"
WORKERS = 8
_lock = threading.Lock()


def run_one(s: dict) -> dict:
    group = preprocess(s)
    input_json = {"snapshot_date": SNAPSHOT_DATE, "supplier": group["supplier"], "items": group["items"]}
    docs, window = analysis_history(input_json, s["history"])
    user_text = json.dumps({**input_json, "history": history_for_model(docs)}, ensure_ascii=False)
    started = time.time()
    answer = _llm(TEACHER, EFFORT).invoke([SystemMessage(content=ANALYSIS_AGENT_SYSTEM), HumanMessage(content=user_text)])
    seconds = round(time.time() - started, 1)
    raw = answer.text if hasattr(answer, "text") else str(answer.content)
    usage = getattr(answer, "usage_metadata", None) or {}
    try:
        out = _extract_json_from_text(raw)
        analysis = from_model(out) if isinstance(out, dict) else {"unparsed_output": raw}
    except Exception:
        analysis = {"unparsed_output": raw}
    if "unparsed_output" in analysis:
        evaluation = {"verdict": "unparsed", "failed": ["C1"], "warnings": [], "checks": {}}
    else:
        analysis["references"] = {"documents": document_references(docs), "window": window}
        evaluation = evaluate_analysis(group["supplier"], group["items"], analysis, docs, settings.evaluator_model,
                                       snapshot_date=SNAPSHOT_DATE)
    return {"scenario_id": s["scenario_id"], "category": s["category"], "labels": s["labels"],
            "teacher": TEACHER, "effort": EFFORT, "prompt_version": PROMPT_VERSION,
            "generated_at": datetime.now().isoformat(timespec="seconds"), "seconds": seconds,
            "usage": {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")},
            "messages": [{"role": "system", "content": ANALYSIS_AGENT_SYSTEM},
                         {"role": "user", "content": user_text},
                         {"role": "assistant", "content": raw}],
            "analysis": analysis, "evaluation": evaluation,
            "train": evaluation["verdict"] == "ready"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=Path, default=ROOT / "training_data" / "sft_v5" / "scenarios.jsonl")
    a = ap.parse_args()
    out = a.scenarios.parent / f"teacher_{a.scenarios.stem}.jsonl"
    done = {json.loads(l)["scenario_id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    todo = [json.loads(l) for l in open(a.scenarios, encoding="utf-8") if json.loads(l)["scenario_id"] not in done]
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(run_one, s): s["scenario_id"] for s in todo}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                print("ERROR", futs[fut], type(e).__name__, str(e)[:200], flush=True)
                continue
            with _lock:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                f.flush()
            print(r["scenario_id"], r["seconds"], "s", r["evaluation"]["verdict"], ",".join(r["evaluation"]["failed"]), flush=True)
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    print(f"-> {out}: {len(rows)} runs, {sum(r['train'] for r in rows)} pass every blocking check")


if __name__ == "__main__":
    main()

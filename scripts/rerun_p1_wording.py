"""
Judge validation: the purchase request checks (P1a-c) again after P1a's wording was changed
(2026-10-24: drawing a conclusion the input does not draw from a finding, such as that a
supplier's record supports buying from it, adds an assessment). Only P1 changed, so only P1 is
measured again, with the provisional judge (Haiku, high effort, scoring-sheet format), three runs:
  - the 8 seeded P1 faults of eval_data/judge_calibration_v5.jsonl (detection);
  - the purchase requests of the 6 hand-scored outputs (eval_data/hand_scores_v5.jsonl: false alarms,
    and eval_14's added supplier assessment, the one P1a fail).

Out: eval_results/judge_p1_rewording_runs.jsonl, with the settings in
eval_results/judge_p1_rewording_settings.json.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.run_judge_validation as jv  # noqa: E402
from config import settings  # noqa: E402
from services import quality_checks as q  # noqa: E402

SPEC = "anthropic:claude-haiku-5-5?effort=high&format=schema"
OUT = ROOT / "eval_results" / "judge_p1_rewording_runs.jsonl"
SETTINGS = ROOT / "eval_results" / "judge_p1_rewording_settings.json"


def judge(kind: str, sid: str, payload: dict, approval: dict, rep: int) -> dict:
    jv._usage.calls = []
    t = time.perf_counter()
    res = q.approval_checks(payload, approval, [], judge=SPEC, retries=settings.judge_retries)
    tin = sum(c[0] for c in jv._usage.calls); tout = sum(c[1] for c in jv._usage.calls)
    return {"kind": kind, "sample": sid, "rep": rep, "seconds": round(time.perf_counter() - t, 2),
            "cost": round((tin * 0.10 + tout * 0.50) / 1e6, 6),
            "verdicts": {k: {"pass": v["pass"], "detail": v.get("detail", "")} for k, v in res["checks"].items() if k.startswith("P1")}}


def main() -> None:
    SETTINGS.write_text(json.dumps({"judge": SPEC, "p1_wording": q._P1_TEXT, "judge_system": q.JUDGE_SYSTEM,
                                    "quality_checks_sha256": jv.sha256(ROOT / "services" / "quality_checks.py"),
                                    "started": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False, indent=2), encoding="utf-8")
    seeded = [s for s in (json.loads(l) for l in open(ROOT / "eval_data" / "judge_calibration_v5.jsonl", encoding="utf-8"))
              if s["target_check"].startswith("P")]
    scored = {json.loads(l)["scenario_id"] for l in open(ROOT / "eval_data" / "hand_scores_v5.jsonl", encoding="utf-8")}
    bases = [b for b in (json.loads(l) for l in open(ROOT / "eval_results" / "v5_base_outputs.jsonl", encoding="utf-8"))
             if b["scenario_id"] in scored and b.get("approval")]
    jobs = [("seeded", s["id"] + ":" + s["target_check"], s["approval_input"], s["approval"], r) for s in seeded for r in (1, 2, 3)]
    jobs += [("hand", b["scenario_id"], b["approval_input"], b["approval"], r) for b in bases for r in (1, 2, 3)]
    with open(OUT, "w", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=4) as ex:
        for row in ex.map(lambda j: judge(*j), jobs):
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            print(row["kind"], row["sample"], row["rep"], {k: v["pass"] for k, v in row["verdicts"].items()}, row["seconds"], flush=True)


if __name__ == "__main__":
    main()

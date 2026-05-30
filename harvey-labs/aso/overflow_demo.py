"""overflow_demo.py — the long-context headline as a controlled A/B.

Tool-result clearing keeps the inner agent alive on matters whose documents
overflow the context window. For each oversized matter we run the SAME agent
twice: the baseline scaffold (no clearing) vs a clearing scaffold. Baseline is
expected to hit context_overflow=True (the agent dies mid-task); clearing should
stay False and actually score. Runs are dispatched to the deployed Modal
function (run_eval_job), so the heavy 200K-900K-token reads happen remotely.

Run:
    uv run modal deploy aso/modal_app.py     # once (deploys run_eval_job)
    uv run python -m aso.overflow_demo
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from harness.run import _load_env

_load_env()  # harmless locally; this script only needs Modal creds to dispatch

import modal

from aso.scaffold import Scaffold

RESULTS = Path(__file__).resolve().parent.parent / "results" / "aso"

# Oversized matters (real cl100k token counts via the harness parsers), none in
# SCREEN/DEV/HOLDOUT. Hero = 917K (4.5x window); the others overflow too. The
# 200K-ish one (indenture) is the safest clean clearing win within the Modal
# per-run timeout; the bigger ones make the baseline-overflow dramatic.
DEMO_TASKS = [
    "funds-asset-management/respond-to-comment-memo",                     # ~917K tokens
    "tax/draft-cross-border-acquisition-tax-memo",                        # ~467K tokens
    "capital-markets/draft-indenture-for-senior-secured-notes-offering",  # ~225K tokens
]

MODEL = "anthropic/claude-haiku-4-5"
JUDGE = "claude-sonnet-4-6"
MAX_TURNS = 200
CLEARING = {"trigger": 100000, "keep": 3}


def main():
    base = Scaffold.baseline()
    clearing = base.copy_with(module_config={"clearing": dict(CLEARING)})
    variants = {"baseline": base, "clearing": clearing}

    jobs = []
    for task in DEMO_TASKS:
        for vid, sc in variants.items():
            jobs.append({
                "variant_id": vid, "task": task, "scaffold": sc.model_dump(),
                "model": MODEL, "judge_model": JUDGE, "max_turns": MAX_TURNS,
            })

    fn = modal.Function.from_name("aso", "run_eval_job")
    print(f"dispatching {len(jobs)} runs ({len(DEMO_TASKS)} tasks x baseline/clearing) ...", flush=True)
    results = list(fn.map(jobs, return_exceptions=True))

    rows = []
    for job, res in zip(jobs, results):
        if isinstance(res, Exception):
            rows.append({"task": job["task"], "variant": job["variant_id"],
                         "status": "failed", "error": f"{type(res).__name__}: {res}"})
        else:
            rows.append({"task": job["task"], "variant": job["variant_id"],
                         "status": res.get("status"),
                         "context_overflow": res.get("context_overflow"),
                         "pass_rate": res.get("pass_rate"),
                         "n_passed": res.get("n_passed"), "n_criteria": res.get("n_criteria"),
                         "turn_count": res.get("turn_count"),
                         "input_tokens": res.get("input_tokens")})

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "overflow_demo.json").write_text(json.dumps({
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "judge_model": JUDGE, "clearing": CLEARING, "results": rows,
    }, indent=2))

    print("\n" + "=" * 80)
    print(f"  OVERFLOW DEMO — clearing vs baseline   (inner model: {MODEL})")
    print("=" * 80)
    by_task: dict[str, dict] = {}
    for r in rows:
        by_task.setdefault(r["task"], {})[r["variant"]] = r
    for task, d in by_task.items():
        print(f"\n{task}")
        for vid in ("baseline", "clearing"):
            r = d.get(vid, {})
            if r.get("status") != "ok":
                print(f"  {vid:9s}: {r.get('status')}  {r.get('error', '')}")
            else:
                pr = r.get("pass_rate")
                print(f"  {vid:9s}: overflow={str(r.get('context_overflow')):5s}  "
                      f"pass_rate={pr:.3f}  ({r.get('n_passed')}/{r.get('n_criteria')})  "
                      f"turns={r.get('turn_count')}  in_tok={r.get('input_tokens')}")
    print(f"\nsaved -> {RESULTS / 'overflow_demo.json'}")


if __name__ == "__main__":
    main()

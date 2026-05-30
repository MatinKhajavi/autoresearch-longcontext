"""overflow_force.py — decisive test of whether overflow is reachable at all.

The A/B (overflow_demo.py) showed the inner agent reads SELECTIVELY and never
overflows, even on a ~917K-token matter. This forces the opposite: a coverage
scaffold that orders the agent to read EVERY document in full before drafting,
so the context actually balloons. Run baseline-coverage vs clearing-coverage on
the heaviest matter.

  - If baseline-coverage hits context_overflow=True -> the window is < the
    ingested size; clearing should then survive (the real long-context win).
  - If it still doesn't overflow -> the effective window is huge (>=~1M) and the
    overflow framing is dead for this model; the story is coverage/focus instead.

Run:  uv run python -m aso.overflow_force
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from harness.run import _load_env

_load_env()

import modal

from aso.scaffold import Scaffold

RESULTS = Path(__file__).resolve().parent.parent / "results" / "aso"
TASK = "funds-asset-management/respond-to-comment-memo"   # ~917K doc tokens
MODEL = "anthropic/claude-haiku-4-5"
JUDGE = "claude-sonnet-4-6"
MAX_TURNS = 300

COVERAGE = (
    "\n\n## MANDATORY FULL COVERAGE\n"
    "Before producing ANY deliverable you MUST read EVERY file in the documents/ "
    "directory IN FULL with the read tool — the entire file, not just the head, and "
    "do not skip or merely sample any document. First list all documents, then read "
    "each one completely, then begin drafting. Comprehensive reading is required."
)


def main():
    base = Scaffold.baseline()
    cover = base.copy_with(system_prompt=base.system_prompt + COVERAGE)
    cover_clear = cover.copy_with(module_config={"clearing": {"trigger": 100000, "keep": 3}})
    variants = {"coverage_baseline": cover, "coverage_clearing": cover_clear}

    jobs = [{
        "variant_id": vid, "task": TASK, "scaffold": sc.model_dump(),
        "model": MODEL, "judge_model": JUDGE, "max_turns": MAX_TURNS,
    } for vid, sc in variants.items()]

    fn = modal.Function.from_name("aso", "run_eval_job")
    print(f"forcing full-coverage reads on {TASK} ({len(jobs)} runs) ...", flush=True)
    results = list(fn.map(jobs, return_exceptions=True))

    rows = []
    for job, res in zip(jobs, results):
        if isinstance(res, Exception):
            rows.append({"variant": job["variant_id"], "status": "failed", "error": str(res)})
        else:
            rows.append({"variant": job["variant_id"], "status": res.get("status"),
                         "context_overflow": res.get("context_overflow"),
                         "pass_rate": res.get("pass_rate"),
                         "n_passed": res.get("n_passed"), "n_criteria": res.get("n_criteria"),
                         "turn_count": res.get("turn_count"), "input_tokens": res.get("input_tokens")})

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "overflow_force.json").write_text(json.dumps(
        {"saved_at": datetime.now(timezone.utc).isoformat(), "task": TASK,
         "model": MODEL, "max_turns": MAX_TURNS, "results": rows}, indent=2))

    print("\n" + "=" * 78)
    print(f"  FORCED-COVERAGE OVERFLOW TEST — {TASK}")
    print("=" * 78)
    for r in rows:
        if r.get("status") != "ok":
            print(f"  {r['variant']:18s}: {r.get('status')} {r.get('error','')}")
        else:
            print(f"  {r['variant']:18s}: overflow={str(r.get('context_overflow')):5s}  "
                  f"pass={r.get('pass_rate'):.3f} ({r.get('n_passed')}/{r.get('n_criteria')})  "
                  f"turns={r.get('turn_count')}  in_tok={r.get('input_tokens'):,}")
    print(f"\nsaved -> {RESULTS / 'overflow_force.json'}")


if __name__ == "__main__":
    main()

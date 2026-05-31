"""overflow_seeded.py — clearing vs baseline A/B, with SEEDS for error bars.

Same controlled A/B as overflow_demo (baseline scaffold vs a clearing scaffold,
same inner model, same matters), but each (matter, variant) is run SEEDS times so
we report mean + spread. Run-to-run variance is large on this metric, so a single
seed is not trustworthy; the paired seeded A/B is the honest version of "did the
change help?" — both arms measured under identical conditions, not vs a stale or
noise-low recorded baseline.

All jobs go out in ONE parallel Modal map (run_eval_job), so wall-clock is the
slowest single run, not the sum.

Run:
    uv run modal deploy aso/modal_app.py     # once (deploys run_eval_job)
    uv run python -m aso.overflow_seeded
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

from harness.run import _load_env

_load_env()  # harmless locally; this script only needs Modal creds to dispatch

import modal

from aso.scaffold import Scaffold

RESULTS = Path(__file__).resolve().parent.parent / "results" / "aso"

# Oversized matters (real cl100k token counts), none in SCREEN/DEV/HOLDOUT.
# 917K = 4.5x window (clearing shines); 467K still overflows-ish; 225K is the
# small control where we EXPECT clearing to be neutral/slightly negative — kept
# in deliberately so the story is honest ("helps where context is big").
DEMO_TASKS = [
    "funds-asset-management/respond-to-comment-memo",                     # ~917K tokens
    "tax/draft-cross-border-acquisition-tax-memo",                        # ~467K tokens
    "capital-markets/draft-indenture-for-senior-secured-notes-offering",  # ~225K tokens
]

MODEL = "anthropic/claude-haiku-4-5"
JUDGE = "claude-sonnet-4-6"
MAX_TURNS = 200
CLEARING = {"trigger": 100000, "keep": 3}
SEEDS = 3


def main():
    base = Scaffold.baseline()
    clearing = base.copy_with(module_config={"clearing": dict(CLEARING)})
    variants = {"baseline": base, "clearing": clearing}

    jobs, meta = [], []
    for task in DEMO_TASKS:
        for vid, sc in variants.items():
            for s in range(SEEDS):
                jobs.append({
                    "variant_id": vid, "task": task, "scaffold": sc.model_dump(),
                    "model": MODEL, "judge_model": JUDGE, "max_turns": MAX_TURNS,
                })
                meta.append({"task": task, "variant": vid, "seed": s})

    fn = modal.Function.from_name("aso", "run_eval_job")
    print(f"dispatching {len(jobs)} runs "
          f"({len(DEMO_TASKS)} tasks x 2 variants x {SEEDS} seeds) ...", flush=True)
    results = list(fn.map(jobs, return_exceptions=True))

    rows = []
    for m, res in zip(meta, results):
        if isinstance(res, Exception):
            rows.append({**m, "status": "failed", "error": f"{type(res).__name__}: {res}"})
        else:
            rows.append({**m, "status": res.get("status"),
                         "context_overflow": res.get("context_overflow"),
                         "pass_rate": res.get("pass_rate"),
                         "n_passed": res.get("n_passed"), "n_criteria": res.get("n_criteria"),
                         "turn_count": res.get("turn_count"),
                         "input_tokens": res.get("input_tokens")})

    agg: dict[str, dict] = {}
    for task in DEMO_TASKS:
        agg[task] = {}
        for vid in ("baseline", "clearing"):
            prs = [r["pass_rate"] for r in rows
                   if r["task"] == task and r["variant"] == vid
                   and r.get("status") == "ok" and r.get("pass_rate") is not None]
            if prs:
                agg[task][vid] = {"mean": mean(prs), "min": min(prs), "max": max(prs),
                                  "std": pstdev(prs) if len(prs) > 1 else 0.0,
                                  "n": len(prs), "all": prs}
            else:
                agg[task][vid] = {"mean": None, "n": 0, "all": []}

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "overflow_seeded.json"
    out.write_text(json.dumps({
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "judge_model": JUDGE, "clearing": CLEARING, "seeds": SEEDS,
        "rows": rows, "aggregate": agg,
    }, indent=2))

    print("\n" + "=" * 86)
    print(f"  CLEARING vs BASELINE — seeded A/B ({SEEDS} seeds)   inner: {MODEL}  judge: {JUDGE}")
    print("=" * 86)
    for task in DEMO_TASKS:
        b, c = agg[task]["baseline"], agg[task]["clearing"]
        print(f"\n{task}")
        for vid, a in (("baseline", b), ("clearing", c)):
            if a["mean"] is None:
                print(f"  {vid:9s}: no successful runs")
            else:
                seeds_str = ", ".join(f"{x:.3f}" for x in a["all"])
                print(f"  {vid:9s}: mean {a['mean']:.3f}  [min {a['min']:.3f}, max {a['max']:.3f}]"
                      f"  n={a['n']}  ({seeds_str})")
        if b["mean"] is not None and c["mean"] is not None:
            d = c["mean"] - b["mean"]
            pct = (d / b["mean"] * 100) if b["mean"] else float("nan")
            print(f"  delta    : {d:+.3f}  ({pct:+.0f}%)")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()

"""optimize.py — run the autoresearch loop for one tier.

Wires together: baseline measurement, the researcher agent, the Modal fan-out
(via the successive-halving controller), a live progress view, and a final
HOLDOUT evaluation of the discovered champion (the honest headline number).

Usage:
    uv run modal deploy aso/modal_app.py        # once
    uv run python -m aso.optimize --rounds 3
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from agents import Runner
from harness.run import _load_env

# Load harvey-labs/.env so the LOCAL researcher (GPT-5.5) sees OPENAI_API_KEY.
# (Remote Modal runs get their keys from the llm-keys secret instead.)
_load_env()

from aso.controller import build_jobs, make_modal_eval_fn, mean_by_variant
from aso.datasets import DEV, HOLDOUT, SCREEN, TASK_SETS
from aso.researcher import ResearchState, build_researcher
from aso.scaffold import Scaffold
from aso.tracking import RunLedger, live

RESULTS = Path(__file__).resolve().parent.parent / "results" / "aso"


async def evaluate_scaffold(scaffold, tasks, eval_fn, model, judge_model, max_turns, variant_id, seeds=1):
    """Run one scaffold across `tasks` (× `seeds`) via the fan-out; return (mean, results)."""
    jobs = build_jobs({variant_id: scaffold}, tasks, model, judge_model, max_turns, seeds)
    results = await eval_fn(jobs)
    return mean_by_variant(results).get(variant_id, 0.0), results


async def run_optimization(rounds, model, judge_model, researcher_model, inner_max_turns,
                           screen=SCREEN, dev=DEV, holdout=HOLDOUT, seeds=1, headline_seeds=3,
                           tier=1, reuse_baseline=True):
    RESULTS.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger(round_label=f"tier{tier}-baseline")
    jsonl = RESULTS / f"runs_tier{tier}.jsonl"   # per-tier so T1/T2/T3 don't overwrite

    with live(ledger) as refresh:
        eval_fn = make_modal_eval_fn(ledger=ledger, jsonl_path=jsonl, refresh=refresh)
        baseline = Scaffold.baseline()

        # ── Baseline on dev + holdout (the number to beat) ───────────────
        # The unchanged scaffold is measured ONCE, ever. By default we REUSE the
        # recorded baseline_tier{tier}.json and skip this phase — re-running the
        # baseline here (plus the per-round "keep" control, now removed) was the
        # bulk of the wasted compute. Pass --fresh-baseline to force a re-measure.
        recorded = RESULTS / f"baseline_tier{tier}.json"
        if reuse_baseline and recorded.exists():
            rec = json.loads(recorded.read_text())
            base_dev_mean = rec.get("dev_mean_pass_rate", 0.0)
            base_hold_mean = rec.get("holdout_mean_pass_rate", 0.0)
            base_overflows = rec.get("dev_overflows", 0)
            sample_fails = []
            print(f"[baseline] REUSING {recorded.name}: dev={base_dev_mean:.3f} "
                  f"holdout={base_hold_mean:.3f}  (baseline phase skipped; "
                  f"--fresh-baseline to re-measure)")
        else:
            base_dev_mean, base_dev_results = await evaluate_scaffold(
                baseline, dev, eval_fn, model, judge_model, inner_max_turns, "baseline",
                seeds=headline_seeds)
            base_hold_mean, _ = await evaluate_scaffold(
                baseline, holdout, eval_fn, model, judge_model, inner_max_turns, "baseline_holdout",
                seeds=headline_seeds)
            base_overflows = sum(1 for r in base_dev_results if r.get("context_overflow"))
            sample_fails = [t for r in base_dev_results for t in r.get("failed_criteria", [])[:2]][:10]
            (RESULTS / f"baseline_tier{tier}.json").write_text(json.dumps({
                "dev_mean_pass_rate": base_dev_mean, "holdout_mean_pass_rate": base_hold_mean,
                "dev_overflows": base_overflows, "model": model, "judge_model": judge_model,
                "n_dev": len(dev), "n_holdout": len(holdout), "tier": tier,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2))

        # ── Researcher loop (agent proposes; controller allocates) ───────
        ledger.round_label = f"tier{tier}-optimizing"
        state = ResearchState(
            champion=baseline, screen=screen, dev=dev, model=model, judge_model=judge_model,
            eval_fn=eval_fn, inner_max_turns=inner_max_turns, seeds=seeds, tier=tier,
            rounds_jsonl_path=str(RESULTS / f"rounds_tier{tier}.jsonl"),
            baseline_dev_mean=round(base_dev_mean, 3),
        )
        researcher = build_researcher(model=researcher_model, tier=tier)
        seed = (
            f"Baseline scaffold scores {base_dev_mean:.3f} mean criterion-pass-rate on the "
            f"dev set ({len(dev)} tasks); {base_overflows} dev run(s) overflowed context. "
            f"Sample failing criteria: {sample_fails}. "
            f"Run up to {rounds} rounds to improve it. Begin round 1."
        )
        # max_turns budget: a few agent turns per round (propose/evaluate/inspect/set/decide)
        await Runner.run(researcher, seed, context=state, max_turns=rounds * 6 + 4)

        # ── Champion on dev + holdout (honest headline) ──────────────────
        # If the search never cleared the bar, the champion IS the baseline — don't
        # pay to re-measure the unchanged scaffold a third time; reuse its numbers.
        ledger.round_label = "champion-eval"
        if state.champion.model_dump() == baseline.model_dump():
            champ_dev_mean, champ_hold_mean, champ_overflows = base_dev_mean, base_hold_mean, 0
            print("[champion] unchanged from baseline — reusing baseline headline "
                  "(no extra champion runs)")
        else:
            champ_dev_mean, _ = await evaluate_scaffold(
                state.champion, dev, eval_fn, model, judge_model, inner_max_turns, "champion",
                seeds=headline_seeds)
            champ_hold_mean, champ_hold_results = await evaluate_scaffold(
                state.champion, holdout, eval_fn, model, judge_model, inner_max_turns, "champion_holdout",
                seeds=headline_seeds)
            champ_overflows = sum(1 for r in champ_hold_results if r.get("context_overflow"))

    (RESULTS / f"champion_tier{tier}.json").write_text(json.dumps({
        "tier": tier,
        "baseline_dev_mean": base_dev_mean, "champion_dev_mean": champ_dev_mean,
        "baseline_holdout_mean": base_hold_mean, "champion_holdout_mean": champ_hold_mean,
        "champion_holdout_overflows": champ_overflows,
        "rounds_done": state.rounds_done, "history": state.history,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    (RESULTS / f"champion_scaffold_tier{tier}.json").write_text(
        json.dumps(state.champion.model_dump(), indent=2))

    print("\n" + "=" * 64)
    print(f"  TIER {tier}  rounds: {state.rounds_done}")
    print(f"  DEV     baseline {base_dev_mean:.3f}  ->  champion {champ_dev_mean:.3f}")
    print(f"  HOLDOUT baseline {base_hold_mean:.3f}  ->  champion {champ_hold_mean:.3f}")
    print(f"  saved -> {RESULTS}/champion_tier{tier}.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--model", default="anthropic/claude-haiku-4-5", help="inner agent model")
    p.add_argument("--judge-model", default="claude-sonnet-4-6",
                   help="stricter judge than Haiku; more discriminating signal (measured)")
    p.add_argument("--researcher-model", default="gpt-5.5",
                   help="OpenAI model for the researcher orchestrator (Agents SDK, native)")
    p.add_argument("--inner-max-turns", type=int, default=120)
    p.add_argument("--max-screen", type=int, default=None, help="cap screen tasks (fast validation)")
    p.add_argument("--max-dev", type=int, default=None, help="cap dev tasks (fast validation)")
    p.add_argument("--max-holdout", type=int, default=None, help="cap holdout tasks (fast validation)")
    p.add_argument("--seeds", type=int, default=3,
                   help="runs per (variant,task) in the loop; >=3 lets the controller's median "
                        "de-noise the ~1/3 spurious-0.0 runs (at <=2 it can't)")
    p.add_argument("--headline-seeds", type=int, default=3,
                   help="runs per task for the baseline/champion headline numbers, averaged")
    p.add_argument("--tier", type=int, default=1, choices=[1, 2, 3],
                   help="mutation surface: 1=text only, 2=+long-context modules, 3=+code")
    p.add_argument("--task-set", default="default", choices=list(TASK_SETS),
                   help="default = small analysis tasks; heavy = big matters where memory mgmt helps")
    p.add_argument("--fresh-baseline", action="store_true",
                   help="re-measure the baseline even if results/aso/baseline_tier{tier}.json exists. "
                        "Default REUSES the recorded baseline and skips the baseline phase entirely "
                        "(the unchanged scaffold is measured once, ever — not re-run every invocation).")
    a = p.parse_args()
    base_screen, base_dev, base_holdout = TASK_SETS[a.task_set]
    screen = base_screen[: a.max_screen] if a.max_screen else base_screen
    dev = base_dev[: a.max_dev] if a.max_dev else base_dev
    holdout = base_holdout[: a.max_holdout] if a.max_holdout else base_holdout
    asyncio.run(run_optimization(
        a.rounds, a.model, a.judge_model, a.researcher_model, a.inner_max_turns,
        screen=screen, dev=dev, holdout=holdout, seeds=a.seeds, headline_seeds=a.headline_seeds,
        tier=a.tier, reuse_baseline=not a.fresh_baseline))


if __name__ == "__main__":
    main()

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

from aso.controller import build_jobs, make_modal_eval_fn, mean_by_variant
from aso.datasets import DEV, HOLDOUT, SCREEN
from aso.researcher import ResearchState, build_researcher
from aso.scaffold import Scaffold
from aso.tracking import RunLedger, live

RESULTS = Path(__file__).resolve().parent.parent / "results" / "aso"


def evaluate_scaffold(scaffold, tasks, eval_fn, model, judge_model, max_turns, variant_id):
    """Run one scaffold across `tasks` via the fan-out; return (mean, results)."""
    jobs = build_jobs({variant_id: scaffold}, tasks, model, judge_model, max_turns)
    results = eval_fn(jobs)
    return mean_by_variant(results).get(variant_id, 0.0), results


async def run_optimization(rounds, model, judge_model, researcher_model, inner_max_turns):
    RESULTS.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger(round_label="baseline")
    jsonl = RESULTS / "runs.jsonl"

    with live(ledger) as refresh:
        eval_fn = make_modal_eval_fn(ledger=ledger, jsonl_path=jsonl, refresh=refresh)
        baseline = Scaffold.baseline()

        # ── Baseline on dev + holdout (the number to beat) ───────────────
        base_dev_mean, base_dev_results = evaluate_scaffold(
            baseline, DEV, eval_fn, model, judge_model, inner_max_turns, "baseline")
        base_hold_mean, _ = evaluate_scaffold(
            baseline, HOLDOUT, eval_fn, model, judge_model, inner_max_turns, "baseline_holdout")
        base_overflows = sum(1 for r in base_dev_results if r.get("context_overflow"))
        sample_fails = [t for r in base_dev_results for t in r.get("failed_criteria", [])[:2]][:10]
        (RESULTS / "baseline.json").write_text(json.dumps({
            "dev_mean_pass_rate": base_dev_mean, "holdout_mean_pass_rate": base_hold_mean,
            "dev_overflows": base_overflows, "model": model, "judge_model": judge_model,
            "n_dev": len(DEV), "n_holdout": len(HOLDOUT),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))

        # ── Researcher loop (agent proposes; controller allocates) ───────
        ledger.round_label = "optimizing"
        state = ResearchState(
            champion=baseline, screen=SCREEN, dev=DEV, model=model, judge_model=judge_model,
            eval_fn=eval_fn, inner_max_turns=inner_max_turns, baseline_dev_mean=round(base_dev_mean, 3),
        )
        researcher = build_researcher(model=researcher_model)
        seed = (
            f"Baseline scaffold scores {base_dev_mean:.3f} mean criterion-pass-rate on the "
            f"dev set ({len(DEV)} tasks). Sample failing criteria: {sample_fails}. "
            f"Run up to {rounds} rounds to improve it. Begin round 1."
        )
        # max_turns budget: a few agent turns per round (propose/evaluate/inspect/set/decide)
        await Runner.run(researcher, seed, context=state, max_turns=rounds * 6 + 4)

        # ── Champion on dev + holdout (honest headline) ──────────────────
        ledger.round_label = "champion-eval"
        champ_dev_mean, _ = evaluate_scaffold(
            state.champion, DEV, eval_fn, model, judge_model, inner_max_turns, "champion")
        champ_hold_mean, champ_hold_results = evaluate_scaffold(
            state.champion, HOLDOUT, eval_fn, model, judge_model, inner_max_turns, "champion_holdout")
        champ_overflows = sum(1 for r in champ_hold_results if r.get("context_overflow"))

    (RESULTS / "champion.json").write_text(json.dumps({
        "baseline_dev_mean": base_dev_mean, "champion_dev_mean": champ_dev_mean,
        "baseline_holdout_mean": base_hold_mean, "champion_holdout_mean": champ_hold_mean,
        "champion_holdout_overflows": champ_overflows,
        "rounds_done": state.rounds_done, "history": state.history,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    (RESULTS / "champion_scaffold.json").write_text(json.dumps(state.champion.model_dump(), indent=2))

    print("\n" + "=" * 64)
    print(f"  rounds: {state.rounds_done}")
    print(f"  DEV     baseline {base_dev_mean:.3f}  ->  champion {champ_dev_mean:.3f}")
    print(f"  HOLDOUT baseline {base_hold_mean:.3f}  ->  champion {champ_hold_mean:.3f}")
    print(f"  saved -> {RESULTS}/champion.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--model", default="anthropic/claude-haiku-4-5", help="inner agent model")
    p.add_argument("--judge-model", default="claude-haiku-4-5")
    p.add_argument("--researcher-model", default="gpt-5.5",
                   help="OpenAI model for the researcher orchestrator (Agents SDK, native)")
    p.add_argument("--inner-max-turns", type=int, default=120)
    a = p.parse_args()
    asyncio.run(run_optimization(
        a.rounds, a.model, a.judge_model, a.researcher_model, a.inner_max_turns))


if __name__ == "__main__":
    main()

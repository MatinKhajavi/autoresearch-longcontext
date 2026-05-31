"""Controller — successive-halving allocation over the Modal fan-out.

Per round the researcher proposes K variants. We do NOT run them all on the
full dev set. Instead:
  1. SCREEN  — run every variant on the tiny screen set.
  2. PRUNE   — keep the top-`keep_m` by mean pass-rate; the rest are killed here.
  3. PROMOTE — run survivors on the full dev set; best dev mean = round champion.

`eval_fn(jobs) -> list[dict]` is injected: a fake in tests, the Modal fan-out in
production. Each job is a plain dict (crosses the Modal boundary); each result
carries `variant_id`, `task`, `pass_rate`, `status`, ...
"""

from statistics import median

from aso.scaffold import Scaffold


def build_jobs(variants: dict[str, Scaffold], tasks: list[str], model: str,
               judge_model: str, max_turns: int, seeds: int = 1) -> list[dict]:
    """One job per (variant, task, seed). `seeds`>1 replicates each (variant,task)
    to average out the agent's run-to-run nondeterminism (mean_by_variant then
    averages over tasks AND seeds)."""
    return [
        {
            "variant_id": vid,
            "task": task,
            "seed": s,
            "scaffold": scaf.model_dump(),
            "model": model,
            "judge_model": judge_model,
            "max_turns": max_turns,
        }
        for vid, scaf in variants.items()
        for task in tasks
        for s in range(seeds)
    ]


def mean_by_variant(results: list[dict]) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for r in results:
        by.setdefault(r["variant_id"], []).append(r.get("pass_rate", 0.0))
    return {vid: (sum(v) / len(v) if v else 0.0) for vid, v in by.items()}


def median_by_variant(results: list[dict]) -> dict[str, float]:
    """Robust per-variant score: MEDIAN pass-rate over all runs (seeds x tasks).

    The metric is noisy — on this harness a single run can crater to 0.0 (an
    empty / ungradeable deliverable) on an otherwise-strong scaffold (~1/3 of
    runs, scaffold-independent). The mean lets one such spike dominate; the median
    outvotes a MINORITY of them, so selection isn't hijacked by one unlucky seed.
    Needs seeds>=3 to bite (at seeds<=2 median degenerates to the mean). Crashed
    runs (status=failed -> pass_rate 0.0) are kept in, so a consistently-broken
    variant is still correctly punished."""
    by: dict[str, list[float]] = {}
    for r in results:
        by.setdefault(r["variant_id"], []).append(float(r.get("pass_rate", 0.0) or 0.0))
    return {vid: (median(v) if v else 0.0) for vid, v in by.items()}


def zero_rate_by_variant(results: list[dict]) -> dict[str, float]:
    """Per-variant fraction of runs scoring exactly 0.0 (ungradeable / crashed).

    A RELIABILITY signal kept separate from the quality score: two variants can
    share a median yet differ sharply in how often they produce nothing usable."""
    by: dict[str, list[float]] = {}
    for r in results:
        by.setdefault(r["variant_id"], []).append(
            1.0 if (float(r.get("pass_rate", 0.0) or 0.0) == 0.0) else 0.0)
    return {vid: (sum(v) / len(v) if v else 0.0) for vid, v in by.items()}


async def successive_halving(
    variants: dict[str, Scaffold],
    screen: list[str],
    dev: list[str],
    eval_fn,
    keep_m: int,
    model: str,
    judge_model: str,
    max_turns: int = 120,
    seeds: int = 1,
) -> tuple[str | None, dict]:
    # 1. SCREEN
    screen_results = await eval_fn(build_jobs(variants, screen, model, judge_model, max_turns, seeds))
    screen_means = mean_by_variant(screen_results)
    screen_medians = median_by_variant(screen_results)

    # 2. PRUNE — keep top-m by ROBUST score (median; mean breaks ties). Median so
    #    one unlucky 0.0 seed doesn't prune an otherwise-strong variant.
    survivors = sorted(
        screen_medians,
        key=lambda v: (screen_medians[v], screen_means.get(v, 0.0)),
        reverse=True,
    )[:keep_m]

    # 3. PROMOTE — survivors on the full dev set
    survivor_variants = {v: variants[v] for v in survivors}
    dev_results = await eval_fn(build_jobs(survivor_variants, dev, model, judge_model, max_turns, seeds))
    dev_means = mean_by_variant(dev_results)
    dev_medians = median_by_variant(dev_results)

    # 4. champion = best dev MEDIAN (mean breaks ties); fall back to best screener.
    if dev_medians:
        champion = max(dev_medians, key=lambda v: (dev_medians[v], dev_means.get(v, 0.0)))
    else:
        champion = survivors[0] if survivors else None

    table = {
        "screen_means": screen_means,
        "screen_medians": screen_medians,
        "survivors": survivors,
        "dev_means": dev_means,
        "dev_medians": dev_medians,
        "dev_zero_rate": zero_rate_by_variant(dev_results),
        "champion": champion,
        "screen_results": screen_results,
        "dev_results": dev_results,
    }
    return champion, table


def make_modal_eval_fn(ledger=None, jsonl_path=None, refresh=None):
    """Production eval_fn (async): fan jobs out to the Modal `run_eval_job` function.

    Uses Modal's async `fn.map.aio()` because the optimizer runs inside an event
    loop (the Agents SDK researcher) — the sync `.map()` can't be iterated there.
    ~20 concurrent (bounded by the function's max_containers). Exceptions are
    normalized to failed-result dicts so one bad run never crashes the round.
    `order_outputs=True` keeps results aligned with `jobs` for index mapping.
    """
    import modal

    from aso.tracking import append_jsonl

    fn = modal.Function.from_name("aso", "run_eval_job")

    async def eval_fn(jobs: list[dict]) -> list[dict]:
        if ledger is not None:
            ledger.start(len(jobs))
        results: list[dict] = []
        idx = 0
        async for res in fn.map.aio(jobs, return_exceptions=True, order_outputs=True):
            job = jobs[idx]
            idx += 1
            if isinstance(res, Exception):
                res = {
                    "variant_id": job["variant_id"], "task": job["task"],
                    "pass_rate": 0.0, "status": "failed", "error": str(res),
                    "context_overflow": False, "input_tokens": 0,
                    "output_tokens": 0, "wall_clock_seconds": 0.0,
                }
            if ledger is not None:
                ledger.record(res)
            if jsonl_path is not None:
                append_jsonl(jsonl_path, res)
            if refresh is not None:
                refresh()
            results.append(res)
        return results

    return eval_fn

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

from aso.scaffold import Scaffold


def build_jobs(variants: dict[str, Scaffold], tasks: list[str], model: str,
               judge_model: str, max_turns: int) -> list[dict]:
    return [
        {
            "variant_id": vid,
            "task": task,
            "scaffold": scaf.model_dump(),
            "model": model,
            "judge_model": judge_model,
            "max_turns": max_turns,
        }
        for vid, scaf in variants.items()
        for task in tasks
    ]


def mean_by_variant(results: list[dict]) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for r in results:
        by.setdefault(r["variant_id"], []).append(r.get("pass_rate", 0.0))
    return {vid: (sum(v) / len(v) if v else 0.0) for vid, v in by.items()}


def successive_halving(
    variants: dict[str, Scaffold],
    screen: list[str],
    dev: list[str],
    eval_fn,
    keep_m: int,
    model: str,
    judge_model: str,
    max_turns: int = 120,
) -> tuple[str | None, dict]:
    # 1. SCREEN
    screen_results = eval_fn(build_jobs(variants, screen, model, judge_model, max_turns))
    screen_means = mean_by_variant(screen_results)

    # 2. PRUNE — keep top-m by screen mean (the rest never touch the dev set)
    survivors = sorted(screen_means, key=lambda v: screen_means[v], reverse=True)[:keep_m]

    # 3. PROMOTE — survivors on the full dev set
    survivor_variants = {v: variants[v] for v in survivors}
    dev_results = eval_fn(build_jobs(survivor_variants, dev, model, judge_model, max_turns))
    dev_means = mean_by_variant(dev_results)

    # 4. champion = best dev mean (fallback to best screener if dev empty)
    if dev_means:
        champion = max(dev_means, key=lambda v: dev_means[v])
    else:
        champion = survivors[0] if survivors else None

    table = {
        "screen_means": screen_means,
        "survivors": survivors,
        "dev_means": dev_means,
        "champion": champion,
        "screen_results": screen_results,
        "dev_results": dev_results,
    }
    return champion, table


def make_modal_eval_fn(ledger=None, jsonl_path=None, refresh=None):
    """Production eval_fn: fan jobs out to the Modal `run_eval_job` function.

    ~20 concurrent (bounded by the function's max_containers). Exceptions are
    normalized to failed-result dicts so one bad run never crashes the round.
    """
    import modal

    from aso.tracking import append_jsonl

    fn = modal.Function.from_name("aso", "run_eval_job")

    def eval_fn(jobs: list[dict]) -> list[dict]:
        if ledger is not None:
            ledger.start(len(jobs))
        results: list[dict] = []
        for job, res in zip(jobs, fn.map(jobs, return_exceptions=True)):
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

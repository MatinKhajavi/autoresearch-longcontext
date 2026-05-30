"""report.py — build the demo artifacts from the loop's outputs.

Auto-discovers per-tier results (champion_tier{N}.json, runs_tier{N}.jsonl) and emits:
  - report.md       : headline + freedom-vs-payoff table across tiers, overflow
                      note, and each tier's champion scaffold diff
  - pareto.png      : pass-rate vs cost (tokens) across all runs, champions marked
  - improvement.png : best dev mean per round, per tier

Matplotlib is optional — the markdown report is always produced; PNGs are
skipped with a note if matplotlib isn't installed.

Usage: uv run python -m aso.report
"""

import difflib
import json
from pathlib import Path

from aso.scaffold import Scaffold

RESULTS = Path(__file__).resolve().parent.parent / "results" / "aso"


def _load(path: Path):
    if not path.exists():
        return None
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def _discover_tiers() -> list[int]:
    tiers = sorted(int(p.stem.split("tier")[-1]) for p in RESULTS.glob("champion_tier*.json"))
    return tiers


def _all_runs() -> list[dict]:
    runs = []
    for p in RESULTS.glob("runs_tier*.jsonl"):
        runs += _load(p) or []
    if not runs:  # back-compat
        runs = _load(RESULTS / "runs.jsonl") or []
    return runs


def _plots(champions: dict[int, dict], runs: list[dict]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ["_(matplotlib not installed — PNGs skipped)_"]

    notes = []
    pts = [(r.get("input_tokens", 0) + r.get("output_tokens", 0), r.get("pass_rate", 0.0),
            str(r.get("variant_id", ""))) for r in runs if r.get("status") == "ok"]
    if pts:
        fig, ax = plt.subplots(figsize=(6, 4))
        for x, y, vid in pts:
            champ = vid.startswith("champion")
            ax.scatter(x, y, c="crimson" if champ else "steelblue",
                       s=44 if champ else 20, alpha=0.8,
                       edgecolors="black" if champ else "none", zorder=3 if champ else 2)
        ax.set_xlabel("tokens per run (cost)")
        ax.set_ylabel("criterion pass-rate")
        ax.set_title("Pass-rate vs cost (champion runs in red)")
        fig.tight_layout(); fig.savefig(RESULTS / "pareto.png", dpi=120); plt.close(fig)
        notes.append("![Pareto](pareto.png)")

    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for tier, champ in sorted(champions.items()):
        hist = champ.get("history", [])
        if not hist:
            continue
        xs = [h.get("round", i + 1) for i, h in enumerate(hist)]
        ys = [h.get("round_best_dev_mean", 0.0) for h in hist]
        ax.plot(xs, ys, "o-", label=f"tier {tier}")
        plotted = True
    if plotted:
        any_base = next(iter(champions.values())).get("baseline_dev_mean")
        if any_base is not None:
            ax.axhline(any_base, ls="--", c="gray", label="baseline")
        ax.set_xlabel("round"); ax.set_ylabel("dev mean pass-rate")
        ax.set_title("Scaffold optimization progress"); ax.legend()
        fig.tight_layout(); fig.savefig(RESULTS / "improvement.png", dpi=120); plt.close(fig)
        notes.append("![Improvement](improvement.png)")
    else:
        plt.close(fig)
    return notes


def build_report():
    tiers = _discover_tiers()
    champions = {t: _load(RESULTS / f"champion_tier{t}.json") for t in tiers}
    champions = {t: c for t, c in champions.items() if c}
    runs = _all_runs()

    lines = ["# ASO — Autoresearch Scaffold Optimizer: results\n"]

    # Freedom-vs-payoff: one row per tier, holdout is the honest number.
    if champions:
        lines += [
            "## Freedom vs. payoff (holdout = honest number)\n",
            "| tier | surface | base dev | champ dev | base holdout | champ holdout | Δ holdout |",
            "|---|---|---|---|---|---|---|",
        ]
        surface = {1: "text", 2: "text+modules", 3: "text+modules+code"}
        for t, c in sorted(champions.items()):
            dh = c.get("champion_holdout_mean", 0) - c.get("baseline_holdout_mean", 0)
            lines.append(
                f"| {t} | {surface.get(t, '?')} | {c.get('baseline_dev_mean', 0):.3f} | "
                f"{c.get('champion_dev_mean', 0):.3f} | {c.get('baseline_holdout_mean', 0):.3f} | "
                f"{c.get('champion_holdout_mean', 0):.3f} | {dh:+.3f} |")
        lines.append("")
        for t, c in sorted(champions.items()):
            ov = c.get("champion_holdout_overflows")
            if ov is not None:
                lines.append(f"- tier {t}: champion holdout context-overflows = {ov}, "
                             f"rounds = {c.get('rounds_done', 0)}")
        lines.append("")

    lines += ["## Charts\n", *_plots(champions, runs), ""]

    # Per-tier champion scaffold diff (the discovered behavior).
    base_sp = Scaffold.baseline().system_prompt.splitlines()
    for t in sorted(champions):
        sc = _load(RESULTS / f"champion_scaffold_tier{t}.json")
        if not sc:
            continue
        new_sp = Scaffold(**sc).system_prompt.splitlines()
        diff = list(difflib.unified_diff(base_sp, new_sp, "baseline", f"tier{t}", lineterm=""))
        lines += [f"## Tier {t} champion diff (discovered behavior)\n", "```diff",
                  *(diff or ["(no system-prompt change)"]), "```\n"]
        mc = Scaffold(**sc).module_config
        if mc:
            lines.append(f"**module_config:** `{json.dumps(mc)}`\n")

    out = RESULTS / "report.md"
    out.write_text("\n".join(lines))
    print(f"report -> {out}")
    return out


if __name__ == "__main__":
    build_report()

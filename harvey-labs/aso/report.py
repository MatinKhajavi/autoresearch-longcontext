"""report.py — build the demo artifacts from the loop's outputs.

Reads results/aso/{champion.json, runs.jsonl, champion_scaffold.json} and emits:
  - report.md         : headline table (baseline -> champion, dev + holdout),
                        per-round improvement, overflow note, champion diff
  - pareto.png        : pass-rate vs cost (tokens) scatter, champions highlighted
  - improvement.png   : best dev mean per round

Matplotlib is optional — the markdown report is always produced; PNGs are
skipped with a note if matplotlib isn't installed.

Usage: uv run python -m aso.report
"""

import difflib
import json
from pathlib import Path

from aso.scaffold import Scaffold

RESULTS = Path(__file__).resolve().parent.parent / "results" / "aso"


def _load(name, default=None):
    p = RESULTS / name
    if not p.exists():
        return default
    if name.endswith(".jsonl"):
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return json.loads(p.read_text())


def _maybe_plots(champ, runs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ["_(matplotlib not installed — PNGs skipped)_"]

    notes = []
    # Pareto: pass_rate vs total tokens
    pts = [(r.get("input_tokens", 0) + r.get("output_tokens", 0), r.get("pass_rate", 0.0),
            r.get("variant_id", "")) for r in runs if r.get("status") == "ok"]
    if pts:
        fig, ax = plt.subplots(figsize=(6, 4))
        for x, y, vid in pts:
            champ_pt = vid and vid.startswith("champion")
            ax.scatter(x, y, c="crimson" if champ_pt else "steelblue",
                       s=42 if champ_pt else 22, alpha=0.8,
                       edgecolors="black" if champ_pt else "none", zorder=3 if champ_pt else 2)
        ax.set_xlabel("tokens per run (cost)")
        ax.set_ylabel("criterion pass-rate")
        ax.set_title("Pass-rate vs cost (champion runs in red)")
        fig.tight_layout()
        fig.savefig(RESULTS / "pareto.png", dpi=120)
        plt.close(fig)
        notes.append("![Pareto](pareto.png)")

    # Improvement curve from per-round history
    hist = champ.get("history", []) if champ else []
    if hist:
        xs = [h.get("round", i + 1) for i, h in enumerate(hist)]
        ys = [h.get("round_best_dev_mean", 0.0) for h in hist]
        fig, ax = plt.subplots(figsize=(6, 4))
        if champ and champ.get("baseline_dev_mean") is not None:
            ax.axhline(champ["baseline_dev_mean"], ls="--", c="gray", label="baseline")
        ax.plot(xs, ys, "o-", c="crimson", label="round best")
        ax.set_xlabel("round")
        ax.set_ylabel("dev mean pass-rate")
        ax.set_title("Scaffold optimization progress")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / "improvement.png", dpi=120)
        plt.close(fig)
        notes.append("![Improvement](improvement.png)")
    return notes


def build_report():
    champ = _load("champion.json", {})
    runs = _load("runs.jsonl", [])
    champ_scaffold = _load("champion_scaffold.json")

    lines = ["# ASO — Autoresearch Scaffold Optimizer: results\n"]

    # Headline table
    if champ:
        lines += [
            "## Headline\n",
            "| split | baseline | champion | delta |",
            "|---|---|---|---|",
            f"| dev | {champ.get('baseline_dev_mean', 0):.3f} | {champ.get('champion_dev_mean', 0):.3f} | "
            f"{champ.get('champion_dev_mean', 0) - champ.get('baseline_dev_mean', 0):+.3f} |",
            f"| **holdout** | {champ.get('baseline_holdout_mean', 0):.3f} | "
            f"{champ.get('champion_holdout_mean', 0):.3f} | "
            f"{champ.get('champion_holdout_mean', 0) - champ.get('baseline_holdout_mean', 0):+.3f} |",
            f"\nRounds run: {champ.get('rounds_done', 0)}. "
            f"Holdout overflows (champion): {champ.get('champion_holdout_overflows', 'n/a')}.\n",
        ]

    # Plots
    lines += ["## Charts\n", *(_maybe_plots(champ, runs)), ""]

    # Champion scaffold diff (the discovered behavior)
    if champ_scaffold:
        base = Scaffold.baseline().system_prompt.splitlines()
        new = Scaffold(**champ_scaffold).system_prompt.splitlines()
        diff = list(difflib.unified_diff(base, new, "baseline", "champion", lineterm=""))
        lines += ["## Champion scaffold diff (discovered behavior)\n", "```diff",
                  *(diff or ["(no system-prompt change — check module_config / skills)"]), "```\n"]
        mc = Scaffold(**champ_scaffold).module_config
        if mc:
            lines += [f"**module_config:** `{json.dumps(mc)}`\n"]

    out = RESULTS / "report.md"
    out.write_text("\n".join(lines))
    print(f"report -> {out}")
    return out


if __name__ == "__main__":
    build_report()

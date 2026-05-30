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

# Canonical doc-size hints (real cl100k token counts the overflow A/B was built
# around — see aso/overflow_demo.py). Keyed by task so the table can annotate
# each matter with how heavy it is; unknown tasks fall back to a dash.
DOC_SIZE_HINTS = {
    "funds-asset-management/respond-to-comment-memo": "~917K tok",
    "tax/draft-cross-border-acquisition-tax-memo": "~467K tok",
    "capital-markets/draft-indenture-for-senior-secured-notes-offering": "~225K tok",
}


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


def _short_task(task: str) -> str:
    """Last path segment of a task id (e.g. .../tax-memo -> tax-memo)."""
    return str(task).rsplit("/", 1)[-1]


def _overflow_pairs(demo: dict) -> dict[str, dict[str, dict]]:
    """Group overflow_demo rows by task -> {variant: row}."""
    by_task: dict[str, dict[str, dict]] = {}
    for r in demo.get("results", []):
        by_task.setdefault(r.get("task", "?"), {})[r.get("variant", "?")] = r
    return by_task


def _overflow_plot(demo: dict) -> list[str]:
    """Grouped bar chart: baseline vs clearing pass-rate per matter."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    by_task = _overflow_pairs(demo)
    tasks = [t for t, d in by_task.items()
             if d.get("baseline", {}).get("status") == "ok"
             or d.get("clearing", {}).get("status") == "ok"]
    if not tasks:
        return []

    labels = [_short_task(t) for t in tasks]
    base = [by_task[t].get("baseline", {}).get("pass_rate", 0.0) or 0.0 for t in tasks]
    clear = [by_task[t].get("clearing", {}).get("pass_rate", 0.0) or 0.0 for t in tasks]

    import numpy as np
    x = np.arange(len(tasks))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, base, w, label="baseline", color="steelblue")
    ax.bar(x + w / 2, clear, w, label="clearing", color="crimson")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("criterion pass-rate")
    ax.set_title("Tool-result clearing vs baseline pass-rate per matter")
    ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS / "overflow_bars.png", dpi=120); plt.close(fig)
    return ["![Overflow bars](overflow_bars.png)"]


def _doc_size_tok_from_hint(hint: str) -> int:
    """Token count parsed from a doc-size hint string (0 when unparseable).

    "~917K tok" -> 917000, "~1.2M tok" -> 1200000.
    """
    num = "".join(ch for ch in (hint or "") if ch.isdigit() or ch == ".")
    if not num:
        return 0
    mult = 1_000_000 if "M" in hint else (1_000 if "K" in hint else 1)
    try:
        return int(float(num) * mult)
    except ValueError:
        return 0


def _doc_size_tok(task: str) -> int | None:
    """Heaviness of `task` from its canonical doc-size hint, or None if unknown."""
    hint = DOC_SIZE_HINTS.get(task)
    return _doc_size_tok_from_hint(hint) if hint else None


def _demo_table(demo: dict) -> tuple[list[str], bool]:
    """Render the natural A/B table; return (lines, any_overflow).

    One row per (matter, variant) with pass_rate, n_passed/n_criteria,
    context_overflow and input_tokens; the clearing row also carries the
    per-task Δ (clearing − baseline pass-rate).
    """
    cfg = demo.get("clearing", {})
    cfg_note = (f" Clearing config: trigger={cfg.get('trigger')} input tokens, "
                f"keep last {cfg.get('keep')} tool results." if cfg else "")
    lines = [
        f"A/B of baseline vs clearing on oversized matters. Inner model "
        f"`{demo.get('model', '?')}`, judge `{demo.get('judge_model', '?')}`, n=1 per "
        f"cell.{cfg_note}\n",
        "| matter | doc-size hint | variant | pass_rate | n_passed/n_criteria | "
        "context_overflow | input_tokens | Δ pass (clearing − baseline) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    any_overflow = False
    for task, d in _overflow_pairs(demo).items():
        b, c = d.get("baseline", {}), d.get("clearing", {})
        hint = DOC_SIZE_HINTS.get(task, "—")
        bp, cp = b.get("pass_rate"), c.get("pass_rate")
        delta = (cp - bp) if (bp is not None and cp is not None) else None
        any_overflow = any_overflow or bool(b.get("context_overflow")) or bool(c.get("context_overflow"))
        short = _short_task(task)
        lines.append(
            f"| {short} | {hint} | baseline | {_fmt(bp, '.3f')} | "
            f"{_fmt(b.get('n_passed'), 'd')}/{_fmt(b.get('n_criteria'), 'd')} | "
            f"{_fmt_bool(b.get('context_overflow'))} | {_fmt(b.get('input_tokens'), ',')} | — |")
        lines.append(
            f"| {short} | {hint} | clearing | {_fmt(cp, '.3f')} | "
            f"{_fmt(c.get('n_passed'), 'd')}/{_fmt(c.get('n_criteria'), 'd')} | "
            f"{_fmt_bool(c.get('context_overflow'))} | {_fmt(c.get('input_tokens'), ',')} | "
            f"{('%+.3f' % delta) if delta is not None else '—'} |")
    lines.append("")
    return lines, any_overflow


def _force_table(force: dict) -> list[str]:
    """Render the forced full-coverage table (coverage_baseline vs _clearing)."""
    fb = _force_variant(force, "coverage_baseline")
    fc = _force_variant(force, "coverage_clearing")
    delta = None
    if fb.get("pass_rate") is not None and fc.get("pass_rate") is not None:
        delta = fc["pass_rate"] - fb["pass_rate"]
    lines = [
        f"Forced full-coverage on the heaviest matter "
        f"(`{_short_task(force.get('task', '?'))}`, model `{force.get('model', '?')}`, "
        f"max_turns={force.get('max_turns', '?')}, n=1 per cell): every document is read "
        f"in full before drafting.\n",
        "| variant | pass_rate | n_passed/n_criteria | context_overflow | "
        "input_tokens | Δ pass (clearing − baseline) |",
        "|---|---|---|---|---|---|",
    ]
    lines.append(
        f"| coverage_baseline | {_fmt(fb.get('pass_rate'), '.3f')} | "
        f"{_fmt(fb.get('n_passed'), 'd')}/{_fmt(fb.get('n_criteria'), 'd')} | "
        f"{_fmt_bool(fb.get('context_overflow'))} | {_fmt(fb.get('input_tokens'), ',')} | — |")
    lines.append(
        f"| coverage_clearing | {_fmt(fc.get('pass_rate'), '.3f')} | "
        f"{_fmt(fc.get('n_passed'), 'd')}/{_fmt(fc.get('n_criteria'), 'd')} | "
        f"{_fmt_bool(fc.get('context_overflow'))} | {_fmt(fc.get('input_tokens'), ',')} | "
        f"{('%+.3f' % delta) if delta is not None else '—'} |")
    lines.append("")
    return lines


def _coverage_takeaway(demo: dict | None, force: dict | None, any_overflow: bool) -> str:
    """Build the data-driven takeaway paragraph entirely from the numbers."""
    parts: list[str] = ["**Takeaway:** "]

    # (a) overflow never happened -> the failure mode is under-coverage.
    if not any_overflow:
        heavy = max(DOC_SIZE_HINTS.values(), key=lambda h: _doc_size_tok_from_hint(h)) \
            if DOC_SIZE_HINTS else None
        heavy_note = f" — not even on the {heavy} matter" if heavy else ""
        parts.append(
            f"No `context_overflow` occurred on any run{heavy_note}, "
            "so the failure mode here is **under-coverage**, not overflow-and-die. ")
    else:
        parts.append("`context_overflow` was observed on at least one run (see tables). ")

    # (b) clearing's value = coverage/throughput: more input tokens + higher pass
    # on the heaviest matters, quoting the forced-coverage doubling.
    if force:
        fb = _force_variant(force, "coverage_baseline")
        fc = _force_variant(force, "coverage_clearing")
        bt, ct = fb.get("input_tokens"), fc.get("input_tokens")
        bp, cp = fb.get("pass_rate"), fc.get("pass_rate")
        if bt and ct and bp is not None and cp is not None:
            tok_ratio = ct / bt if bt else 0.0
            pass_ratio = cp / bp if bp else 0.0
            parts.append(
                f"Clearing's value is **coverage/throughput**: under forced full coverage it "
                f"processed **{tok_ratio:.1f}x** more input tokens ({ct:,} vs {bt:,}) and "
                f"raised pass-rate from **{bp:.3f}** to **{cp:.3f}** "
                f"(**{pass_ratio:.1f}x**, a {cp - bp:+.3f} gain). ")

    # (c) clearing helped most as matters got larger (rank by doc-size hint).
    if demo:
        ranked = []
        for task, d in _overflow_pairs(demo).items():
            bp = d.get("baseline", {}).get("pass_rate")
            cp = d.get("clearing", {}).get("pass_rate")
            sz = _doc_size_tok(task)
            if bp is not None and cp is not None and sz is not None:
                ranked.append((sz, task, cp - bp))
        if len(ranked) >= 2:
            ranked.sort()  # smallest matter first
            smallest_sz, smallest_task, smallest_delta = ranked[0]
            largest_sz, largest_task, largest_delta = ranked[-1]
            parts.append(
                f"In the natural A/B the benefit tracked matter size: the largest "
                f"(`{_short_task(largest_task)}`, {DOC_SIZE_HINTS.get(largest_task, '?')}) "
                f"gained {largest_delta:+.3f} while the smallest "
                f"(`{_short_task(smallest_task)}`, {DOC_SIZE_HINTS.get(smallest_task, '?')}) "
                f"moved {smallest_delta:+.3f} — clearing helped most as matters got larger. ")

    parts.append("Caveat: **n=1 per cell** (no seeds yet) — directional, needs replication.")
    return "".join(parts)


def _overflow_section() -> list[str]:
    """Long-context section: clearing as a coverage lever on oversized matters.

    Renders the natural A/B (overflow_demo.json) and forced full-coverage
    (overflow_force.json) tables plus a takeaway computed from the JSON. Each
    file is guarded independently; the whole section is skipped if both absent.
    """
    demo = _load(RESULTS / "overflow_demo.json")
    force = _load(RESULTS / "overflow_force.json")
    if not demo and not force:
        return []

    lines = ["## Long-context: clearing & coverage\n"]
    any_overflow = False

    if demo:
        demo_lines, demo_overflow = _demo_table(demo)
        any_overflow = any_overflow or demo_overflow
        lines += demo_lines

    if force:
        lines += ["### Forced full-coverage (917K-token matter)\n"]
        lines += _force_table(force)
        for r in force.get("results", []):
            any_overflow = any_overflow or bool(r.get("context_overflow"))

    lines += [_coverage_takeaway(demo, force, any_overflow), ""]
    lines += [*(_overflow_plot(demo) if demo else []), ""]
    return [ln for ln in lines if ln is not None]


def _fmt(v, spec: str) -> str:
    """Format a numeric value with `spec`, or '—' if it is None."""
    return format(v, spec) if v is not None else "—"


def _fmt_bool(v) -> str:
    return "—" if v is None else str(bool(v))


def _force_variant(force: dict, variant: str) -> dict:
    for r in force.get("results", []):
        if r.get("variant") == variant:
            return r
    return {}


def _champion_modules(sc: dict) -> list[str]:
    """Surface a champion's module_config + harness code edits from its raw JSON.

    Reads straight off the loaded scaffold dict (not via the Scaffold model) so
    it tolerates the skill_scripts -> code_overrides rename either way. Flags the
    long-context modules (clearing / validate_revise) and, for Tier-3 champions
    carrying harness code_overrides, lists the edited repo-relative file paths.
    """
    lines: list[str] = []
    mc = sc.get("module_config") or {}
    if mc:
        flags = [k for k in ("clearing", "validate_revise") if mc.get(k)]
        flag_note = f" (active: {', '.join(flags)})" if flags else ""
        lines.append(f"**module_config:** `{json.dumps(mc)}`{flag_note}\n")
    # Tier-3 may carry harness code edits: repo-relative path under harness/ ->
    # new source. Accept the new key and the legacy `skill_scripts` name.
    overrides = sc.get("code_overrides") or sc.get("skill_scripts") or {}
    if overrides:
        lines.append(f"**code_overrides (edited harness files):** {len(overrides)} file(s)\n")
        for path in sorted(overrides):
            lines.append(f"- `{path}`")
        lines.append("")
    return lines


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

    # Long-context: clearing & coverage; skipped if the overflow artifacts are
    # absent so the core report still builds.
    lines += _overflow_section()

    # Per-tier champion scaffold diff (the discovered behavior).
    base_sp = Scaffold.baseline().system_prompt.splitlines()
    for t in sorted(champions):
        sc = _load(RESULTS / f"champion_scaffold_tier{t}.json")
        if not sc:
            continue
        new_sp = (sc.get("system_prompt") or "").splitlines()
        diff = list(difflib.unified_diff(base_sp, new_sp, "baseline", f"tier{t}", lineterm=""))
        lines += [f"## Tier {t} champion diff (discovered behavior)\n", "```diff",
                  *(diff or ["(no system-prompt change)"]), "```\n"]
        lines += _champion_modules(sc)

    out = RESULTS / "report.md"
    out.write_text("\n".join(lines))
    print(f"report -> {out}")
    return out


if __name__ == "__main__":
    build_report()

"""The researcher — an OpenAI Agents SDK agent that optimizes the scaffold.

It runs an autonomous loop: propose variants (prompt/skill/module edits, each
with a behavioral hypothesis) -> call `evaluate` (screen->prune->promote on
Modal) -> optionally `inspect_trace` a failure -> `set_champion` the winner ->
repeat until gains plateau. Search budget is allocated by the controller; the
agent only forms hypotheses and reads results, so its own context stays small.

The researcher runs natively on an OpenAI model (GPT-5.5) via the Agents SDK
with high reasoning effort. The inner legal agent and the judge stay on Claude
(via harvey-labs' own anthropic adapter). State is carried in a typed
RunContext, not globals. Tracing is left ON: native OpenAI now, so it works with
OPENAI_API_KEY and gives a free researcher-trace view in the OpenAI dashboard
for the demo (call set_tracing_disabled(True) to opt out).
"""

import asyncio
from dataclasses import dataclass, field

from agents import Agent, ModelSettings, RunContextWrapper, function_tool
from openai.types.shared import Reasoning
from pydantic import BaseModel

from aso.controller import successive_halving
from aso.scaffold import BENCH_ROOT, Scaffold
from aso.tracking import append_jsonl


# ── researcher state (carried via RunContext) ────────────────────────────
@dataclass
class ResearchState:
    champion: Scaffold
    screen: list[str]
    dev: list[str]
    model: str
    judge_model: str
    eval_fn: object                       # callable(jobs)->list[dict]
    keep_m: int = 2
    inner_max_turns: int = 120
    seeds: int = 1                        # runs per (variant, task) — averaged (noise control)
    tier: int = 1                         # 1=text only; 2=+long-context modules; 3=+code
    rounds_jsonl_path: str | None = None  # per-round summaries appended here (live progress)
    rounds_done: int = 0
    materialized: dict[str, Scaffold] = field(default_factory=dict)
    traces: dict[str, str] = field(default_factory=dict)   # run_id -> transcript tail
    history: list[dict] = field(default_factory=list)      # per-round lessons ledger
    baseline_dev_mean: float | None = None


class ClearingSpec(BaseModel):
    """Typed config for the tool-result-clearing long-context module (Phase 5).

    Typed (not a bare dict) so the Agents SDK strict tool schema accepts it.
    """
    trigger: int = 100000
    keep: int = 3


class CodeEdit(BaseModel):
    """Tier-3: a full-file rewrite of one harness source file.

    Typed (not a bare dict) so the Agents SDK strict tool schema accepts it.
    """
    path: str       # repo-root-relative, UNDER harness/ (e.g. "harness/agent_loop.py")
    content: str    # COMPLETE new file source (must parse as Python — compile-gated)


class VariantSpec(BaseModel):
    """One proposed scaffold mutation. Behavioral, not numeric-only."""
    id: str
    hypothesis: str
    system_prompt_append: str | None = None        # text appended to the system prompt
    system_prompt_replace: str | None = None        # full replacement (rare)
    enable_clearing: ClearingSpec | None = None      # Tier 2 module: tool-result clearing
    validate_revise_passes: int | None = None        # Tier 2 module: forced self-review+fix passes (1-2)
    code_edits: list[CodeEdit] | None = None   # Tier 3: rewrite harness source files (free-form)


def apply_patch(champion: Scaffold, spec: VariantSpec, tier: int = 2) -> Scaffold:
    """Build a new Scaffold from the champion + a variant's edits.

    `tier` gates the mutation surface so each tier is a clean experiment:
      tier 1 = text only (system prompt); module/code edits ignored.
      tier 2 = text + long-context modules (clearing, validate-revise).
      tier 3 = text + modules + free-form harness CODE rewrites (under harness/, compile-gated).
    """
    system_prompt = champion.system_prompt
    if spec.system_prompt_replace is not None:
        system_prompt = spec.system_prompt_replace
    elif spec.system_prompt_append:
        system_prompt = champion.system_prompt + "\n\n" + spec.system_prompt_append
    module_config = dict(champion.module_config)
    if tier >= 2 and spec.enable_clearing is not None:
        module_config["clearing"] = {
            "trigger": spec.enable_clearing.trigger,
            "keep": spec.enable_clearing.keep,
        }
    if tier >= 2 and spec.validate_revise_passes:
        module_config["validate_revise"] = int(spec.validate_revise_passes)
    code_overrides = dict(champion.code_overrides)
    if tier >= 3 and spec.code_edits:
        for edit in spec.code_edits:
            code_overrides[edit.path] = edit.content
    return champion.copy_with(
        system_prompt=system_prompt, module_config=module_config, code_overrides=code_overrides,
    )


def _summarize(table: dict, specs: dict[str, VariantSpec], st: ResearchState) -> dict:
    """Compact, bounded result for the agent (keeps its context small)."""
    champ = table["champion"]
    dev_results = table.get("dev_results", [])
    # sample a few failed criteria from the champion's dev runs for hypothesis-forming,
    # and capture every run's transcript tail so inspect_trace can retrieve it
    sample_fails: list[str] = []
    run_errors: dict[str, str] = {}
    for r in dev_results + table.get("screen_results", []):
        if r.get("variant_id") == champ:
            sample_fails.extend(r.get("failed_criteria", [])[:3])
        # P0: surface WHY a variant failed (compile-gate / crash / timeout / KeyError)
        # so the researcher can fix its code edits instead of seeing an unexplained 0.
        if r.get("status") == "failed" and r.get("error"):
            run_errors.setdefault(r.get("variant_id", "?"), str(r.get("error"))[:400])
        if r.get("run_id") and r.get("transcript_tail"):
            st.traces[r["run_id"]] = r["transcript_tail"]

    # Per-variant overflow + cost on the dev set, so the agent can weigh pass-rate
    # against context-overflow and token cost (not pass-rate in isolation).
    def _mean_by_variant(key: str) -> dict[str, float]:
        by: dict[str, list[float]] = {}
        for r in dev_results:
            by.setdefault(r.get("variant_id"), []).append(float(r.get(key, 0) or 0))
        return {v: (sum(x) / len(x) if x else 0.0) for v, x in by.items()}

    return {
        "round": st.rounds_done,
        "screen_means": {k: round(v, 3) for k, v in table["screen_means"].items()},
        "dev_means": {k: round(v, 3) for k, v in table["dev_means"].items()},
        # dev_medians is the DECISION metric (robust to the ~1/3 spurious-0.0 runs);
        # dev_zero_rate is the reliability signal to weigh against it.
        "dev_medians": {k: round(v, 3) for k, v in table.get("dev_medians", {}).items()},
        "dev_zero_rate": {k: round(v, 2) for k, v in table.get("dev_zero_rate", {}).items()},
        "dev_overflow_rate": {v: round(x, 2) for v, x in _mean_by_variant("context_overflow").items()},
        "dev_mean_input_tokens": {v: int(x) for v, x in _mean_by_variant("input_tokens").items()},
        "dev_mean_turns": {v: round(x, 1) for v, x in _mean_by_variant("turn_count").items()},
        "survivors": table["survivors"],
        "round_best_variant": champ,
        "round_best_dev_median": round(table.get("dev_medians", {}).get(champ, 0.0), 3),
        "round_best_dev_mean": round(table["dev_means"].get(champ, 0.0), 3),
        "baseline_dev_mean": st.baseline_dev_mean,
        "sample_failed_criteria": sample_fails[:8],
        "hypotheses_tried": {s.id: s.hypothesis for s in specs.values()},
        "changes_tried": {
            s.id: {
                "prompt": bool(s.system_prompt_append or s.system_prompt_replace),
                "clearing": s.enable_clearing is not None,
                "validate_revise": s.validate_revise_passes,
                "code_files": [e.path for e in (s.code_edits or [])],
            }
            for s in specs.values()
        },
        "run_errors": run_errors,
    }


# ── tools ────────────────────────────────────────────────────────────────
@function_tool
async def evaluate(ctx: RunContextWrapper[ResearchState], variants: list[VariantSpec]) -> dict:
    """Screen -> prune -> promote the proposed variants on the dev set.

    Returns per-variant mean pass-rates, the round's best variant, and a sample
    of the best variant's still-failing criteria to inform the next round.
    """
    st = ctx.context
    built: dict[str, Scaffold] = {}
    specs: dict[str, VariantSpec] = {}
    for v in variants:
        built[v.id] = apply_patch(st.champion, v, tier=st.tier)
        specs[v.id] = v
        st.materialized[v.id] = built[v.id]
    _champ, table = await successive_halving(
        built, st.screen, st.dev, st.eval_fn,
        keep_m=st.keep_m, model=st.model, judge_model=st.judge_model,
        max_turns=st.inner_max_turns, seeds=st.seeds,
    )
    st.rounds_done += 1
    summary = _summarize(table, specs, st)
    st.history.append(summary)
    # Persist the round summary immediately so progress is visible mid-run
    # (e.g. `tail -f rounds_tier{N}.jsonl` shows round 1 while round 2 runs).
    if st.rounds_jsonl_path:
        append_jsonl(st.rounds_jsonl_path, summary)
    return summary


@function_tool
def inspect_trace(ctx: RunContextWrapper[ResearchState], run_id: str, max_chars: int = 2000) -> str:
    """Just-in-time retrieval of one run's transcript tail (bounded)."""
    tail = ctx.context.traces.get(run_id, "")
    return tail[-max_chars:] if tail else "(no trace captured for that run_id)"


@function_tool
def inspect_code(ctx: RunContextWrapper[ResearchState], path: str, max_chars: int = 8000) -> str:
    """Tier-3: read the CURRENT source of a harness file (repo-root-relative, UNDER
    harness/ — e.g. 'harness/agent_loop.py', 'harness/tools.py',
    'harness/adapters/anthropic.py', 'harness/skills/docx/scripts/redline.py') so you
    can propose a full-file rewrite. Returns the champion's current override if it has
    one (iterate on your own edit), else the stock source. Bounded; lists harness
    files if `path` is not found."""
    st = ctx.context
    if path in st.champion.code_overrides:
        return st.champion.code_overrides[path][:max_chars]
    p = BENCH_ROOT / path
    if path.startswith("harness/") and p.suffix == ".py" and p.exists():
        return p.read_text(encoding="utf-8")[:max_chars]
    listing = sorted(
        str(f.relative_to(BENCH_ROOT))
        for f in (BENCH_ROOT / "harness").rglob("*.py")
        if "__pycache__" not in f.parts
    )
    return f"(not found / not under harness/: {path}) editable harness files:\n" + "\n".join(listing[:80])


@function_tool
def set_champion(ctx: RunContextWrapper[ResearchState], variant_id: str) -> str:
    """Promote a previously-evaluated variant to be the new champion scaffold."""
    st = ctx.context
    if variant_id not in st.materialized:
        return f"unknown variant_id {variant_id!r}; evaluate it first"
    st.champion = st.materialized[variant_id]
    return f"champion set to {variant_id!r}"


RESEARCHER_INSTRUCTIONS = """\
You are an autoresearch agent improving a LEGAL AI agent's SCAFFOLD (its system
prompt + skill text + long-context module config) to raise the fraction of
rubric criteria it passes on long, document-heavy legal tasks. You do NOT change
the agent's tools or model — only the scaffold.

Each round:
1. Propose 3-4 variants as scaffold edits, each with a one-line behavioral
   hypothesis (NOT numeric-only tweaks). Strong known levers:
   - a post-draft VALIDATE-then-REVISE pass (check the draft against the
     instructions, then fix gaps) — historically the biggest gain;
   - require broad document COVERAGE before drafting;
   - structure the work: read -> note key facts -> draft -> verify.
   Do NOT spend a slot on a do-nothing "keep"/control variant. The unchanged
   baseline was measured ONCE at the start; its score is handed to you every
   round as baseline_dev_mean — a FIXED bar. Every variant you propose must be a
   real change.
2. Call evaluate(variants). Judge variants by dev_medians (the ROBUST score:
   median over seeds, so one unlucky 0.0 run can't fool you) — NOT by a single
   lucky run or by dev_means. Also read dev_zero_rate: a variant with a good
   median but a high zero_rate is UNRELIABLE; prefer a slightly-lower-median
   variant that fails to produce a deliverable less often.
3. Optionally inspect_trace(run_id) on the best variant to see WHY criteria
   still fail, and use that to form the next round's hypotheses.
4. Call set_champion(best_variant_id) ONLY if its dev_median beats
   baseline_dev_mean by a real margin (>= 0.03) AND its zero_rate is no worse
   than the baseline's. A variant that merely ties the bar within noise is NOT an
   improvement — leave the baseline as champion in that case.
5. Stop after ~3 rounds or when gains plateau (a round where nothing clears the
   margin), then summarize what worked and why in 3-4 sentences.

CONTEXT & COST SIGNALS. The inner agent is claude-haiku-4-5 with a ~200K-token
context window. Matters whose documents exceed it OVERFLOW: the agent dies
mid-task and its pass-rate craters. Each round's results give you, per variant:
dev_means (pass-rate), dev_overflow_rate (0-1), and dev_mean_input_tokens /
dev_mean_turns (cost). Prefer variants that raise pass-rate WITHOUT raising
overflow or cost. The DEV tasks here are small and rarely overflow, but real /
holdout matters can be far larger — so on Tier 2+ tool-result clearing is
low-risk insurance against overflow even when DEV doesn't surface it.

Be concrete and legal-domain aware (cite-checking, completeness, internal
consistency). Avoid hardcoding answers to specific tasks — mutations must
generalize."""


TIER_SURFACE = {
    1: ("MUTATION SURFACE (Tier 1 — TEXT ONLY): use system_prompt_append (or, rarely, "
        "system_prompt_replace). Do NOT set enable_clearing — module toggles are ignored at "
        "this tier. Induce coverage/validation behavior through prompt wording alone."),
    2: ("MUTATION SURFACE (Tier 2 — TEXT + MODULES): use system_prompt_append AND, where a "
        "hypothesis calls for it, these long-context modules: (a) enable_clearing — server-side "
        "tool-result clearing that drops old document reads from context but keeps them "
        "re-fetchable (prevents window overflow on long matters); (b) validate_revise_passes "
        "(1-2) — forces the agent through that many mandatory self-review-and-fix passes before "
        "it may finish (structurally guarantees the validate-then-revise behavior rather than "
        "merely asking for it in the prompt). Compose modules with prompt edits as hypotheses warrant."),
    3: ("MUTATION SURFACE (Tier 3 — TEXT + MODULES + FREE-FORM HARNESS CODE): everything in "
        "Tier 2, PLUS you may REWRITE any harness source file under harness/ — the agent loop and "
        "its CONTEXT/MEMORY MANAGEMENT (harness/agent_loop.py), the tools (harness/tools.py), the "
        "model adapter (harness/adapters/anthropic.py), and skill helper-scripts "
        "(harness/skills/*/scripts/*.py). Call inspect_code(path) to read current source, then "
        "propose code_edits=[{path, content}] with the COMPLETE new file. "
        "EMPIRICAL PRIOR — USE IT: on large matters the agent UNDER-READS and loses coverage; the "
        "biggest measured lever is CONTEXT/MEMORY MANAGEMENT — server-side tool-result clearing let "
        "the agent read ~2x more of a 917K-token matter and DOUBLED pass-rate (0.24->0.51). So edits "
        "to how agent_loop.py manages/compacts context (or what the adapter retains) are the most "
        "promising; adding a compaction/summarize-old-reads step is a strong hypothesis. "
        "BUDGET YOUR VARIANTS: prompt-only edits barely move pass-rate on these long matters (our "
        "text-only Tier-1 run came out ~flat), so do NOT spend variants on prompt tweaks alone — "
        "make MOST of your 3-4 variants enable clearing and/or rewrite agent_loop.py's memory "
        "management, with validate_revise as a secondary lever; at most ONE prompt-only variant, and "
        "use prompt edits mainly to SUPPORT a module/code change. "
        "RULES: only files under harness/ are editable; evaluation/ (the judge) and aso/ (the search) "
        "are FIXED. Every edit must be valid Python (compile gate) and runs against an isolated COPY "
        "of the harness in a fresh subprocess — a crash or regression scores 0 and is discarded "
        "(never promoted; the real harness is untouched), so experiment boldly but keep edits "
        "runnable, and change ONE thing at a time when you want clean attribution. "
        "CONTRACT: if you edit agent_loop.py, run_agent must STILL return a dict with keys "
        "turn_count, input_tokens, output_tokens, peak_input_tokens, wall_clock_seconds, "
        "finished_cleanly, context_overflow, tool_metrics — drop any one and the run scores 0. "
        "When an edit fails, you'll see the reason in this round's run_errors."),
}


def build_researcher(model: str = "gpt-5.5", reasoning_effort: str = "high", tier: int = 1) -> Agent:
    instructions = RESEARCHER_INSTRUCTIONS + "\n\n" + TIER_SURFACE.get(tier, TIER_SURFACE[2])
    return Agent(
        name="scaffold-researcher",
        model=model,  # native OpenAI model string (Agents SDK default provider)
        model_settings=ModelSettings(reasoning=Reasoning(effort=reasoning_effort)),
        instructions=instructions,
        tools=[evaluate, inspect_trace, inspect_code, set_champion],
    )

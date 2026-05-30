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
from aso.scaffold import Scaffold
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


class VariantSpec(BaseModel):
    """One proposed scaffold mutation. Behavioral, not numeric-only."""
    id: str
    hypothesis: str
    system_prompt_append: str | None = None       # text appended to the system prompt
    system_prompt_replace: str | None = None       # full replacement (rare)
    enable_clearing: ClearingSpec | None = None     # Phase 5: turn on tool-result clearing


def apply_patch(champion: Scaffold, spec: VariantSpec, tier: int = 2) -> Scaffold:
    """Build a new Scaffold from the champion + a variant's edits.

    `tier` gates the mutation surface so each tier is a clean experiment:
      tier 1 = text only (system prompt); module toggles are ignored.
      tier >=2 = text + long-context modules (e.g. tool-result clearing).
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
    return champion.copy_with(system_prompt=system_prompt, module_config=module_config)


def _summarize(table: dict, specs: dict[str, VariantSpec], st: ResearchState) -> dict:
    """Compact, bounded result for the agent (keeps its context small)."""
    champ = table["champion"]
    # sample a few failed criteria from the champion's dev runs for hypothesis-forming,
    # and capture every run's transcript tail so inspect_trace can retrieve it
    sample_fails: list[str] = []
    for r in table.get("dev_results", []) + table.get("screen_results", []):
        if r.get("variant_id") == champ:
            sample_fails.extend(r.get("failed_criteria", [])[:3])
        if r.get("run_id") and r.get("transcript_tail"):
            st.traces[r["run_id"]] = r["transcript_tail"]
    return {
        "round": st.rounds_done,
        "screen_means": {k: round(v, 3) for k, v in table["screen_means"].items()},
        "dev_means": {k: round(v, 3) for k, v in table["dev_means"].items()},
        "survivors": table["survivors"],
        "round_best_variant": champ,
        "round_best_dev_mean": round(table["dev_means"].get(champ, 0.0), 3),
        "baseline_dev_mean": st.baseline_dev_mean,
        "sample_failed_criteria": sample_fails[:8],
        "hypotheses_tried": {s.id: s.hypothesis for s in specs.values()},
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
   Always include one variant id "keep" that re-states the current champion
   unchanged, as a control.
2. Call evaluate(variants). Read dev_means vs baseline_dev_mean.
3. Optionally inspect_trace(run_id) on the best variant to see WHY criteria
   still fail, and use that to form the next round's hypotheses.
4. Call set_champion(best_variant_id) if it beats the current champion.
5. Stop after ~3 rounds or when gains plateau, then summarize what worked and
   why in 3-4 sentences.

Be concrete and legal-domain aware (cite-checking, completeness, internal
consistency). Avoid hardcoding answers to specific tasks — mutations must
generalize."""


TIER_SURFACE = {
    1: ("MUTATION SURFACE (Tier 1 — TEXT ONLY): use system_prompt_append (or, rarely, "
        "system_prompt_replace). Do NOT set enable_clearing — module toggles are ignored at "
        "this tier. Induce coverage/validation behavior through prompt wording alone."),
    2: ("MUTATION SURFACE (Tier 2 — TEXT + MODULES): use system_prompt_append AND, where a "
        "hypothesis calls for it, enable_clearing to turn on server-side tool-result clearing "
        "(drops old document reads from context but keeps them re-fetchable — prevents the "
        "agent from overflowing the window and giving up on long matters)."),
}


def build_researcher(model: str = "gpt-5.5", reasoning_effort: str = "high", tier: int = 1) -> Agent:
    instructions = RESEARCHER_INSTRUCTIONS + "\n\n" + TIER_SURFACE.get(tier, TIER_SURFACE[2])
    return Agent(
        name="scaffold-researcher",
        model=model,  # native OpenAI model string (Agents SDK default provider)
        model_settings=ModelSettings(reasoning=Reasoning(effort=reasoning_effort)),
        instructions=instructions,
        tools=[evaluate, inspect_trace, set_champion],
    )

# Autoresearch Scaffold Optimizer — Design Spec

- **Date:** 2026-05-30
- **Status:** Draft (awaiting user review)
- **Context:** Modal hackathon — tracks: *Agent Architectures & Control Loops*, *Retrieval & Knowledge Synthesis*, *Applied Autonomous Research*. Time box: ~6 hours. **Solo build.**
- **Constraints:** $ cost and API rate limits are **not** a concern; **wall-clock is the only budget.** Target **~20 concurrent** agent-runs.
- **Working codename:** ASO (Autoresearch Scaffold Optimizer)

---

## 1. Thesis

Harvey's conclusion from post-training open legal agents: **"harnesses are learnable scaffolds, not just inference wrappers."** The scaffold alone moved their all-pass rate **2.6–3.7×**.

**Our move:** learn the scaffold with an **autoresearch loop on Modal** instead of gradient descent. A **researcher agent built on the OpenAI Agents SDK** (mirroring Modal's orchestrator + SubAgentPool example) reads failure traces, proposes *qualitative* long-context scaffold mutations, and a deterministic controller allocates ~20-wide agent-run budget via subset-screening. Karpathy's autonomous-research swarm, pointed at the agent scaffold itself.

The leverage is specific: Harvey found the biggest lift (+1.5) came from a **validate-then-revise loop** — and LAB has **no `Validate` tool**; that behavior is *emergent from the scaffold*. So mutating the scaffold to induce validation/coverage/revision targets exactly the highest-leverage lever.

## 2. Goals / Non-Goals

**Goals**
- End-to-end loop that measurably raises a forked Harvey-LAB agent's **criterion-pass-rate** on a **holdout** legal long-context set, by mutating the scaffold (prompt + skills + long-context modules) — not training.
- Use the **OpenAI Agents SDK** for the researcher/orchestrator (tools, async subagent pool, hooks), Modal for **~20-wide** autoscaling fan-out + subset-screening allocation.
- Compare **three mutation tiers** (text / +modules / +code) as a first-class result.
- **Live progress + failure visibility** over the fan-out.
- Reusable, MIT-licensed OSS artifact: *the optimizer-over-a-harness*.
- Money-shots: pass-rate-vs-cost **Pareto** across rounds + **freedom-vs-payoff** across tiers + champion-scaffold diff.

**Non-Goals (traps for solo/6h)**
- ❌ No KV-cache cartridges / context-distillation training.
- ❌ No new harness from scratch — fork Harvey LAB; keep its inner loop.
- ❌ **No self-hosted GPU model** (solo) — model-agnostic transfer is shown cheaply via API models (optimize on Haiku → show lift on Sonnet/Opus).
- ❌ Not optimizing binary all-pass directly (too sparse; see §7).
- ❌ No new agent *tools* in Tier 1–2: long-context modules are control-flow/context ops around LAB's fixed 6-tool set.

## 3. Ground Truth on Harvey LAB (verified by code inspection at `/tmp/harvey-lab-inspect`)

- **MIT licensed**, `uv`-based, Python 3.12+.
- **Eval = criterion-scoped LLM-as-judge:** each criterion → one judge call seeing only that criterion's deliverable file(s) (`evaluation/scoring.py:335-349`); default judge `claude-sonnet-4-6`, temp 0.0 (deterministic), parallel (`scoring.py:371-372`); all-pass = 1.0 iff every criterion passes (`scoring.py:374-378`). **Not the bottleneck.**
- **Agent RUN is the wall-clock bottleneck:** up to 200 turns (`harness/agent_loop.py`), reads large docs, tools in a rootless Podman sandbox, network off (`sandbox/sandbox.py`). ~2–15 min/run.
- **6 tools:** `bash`, `read` (docx/pdf/xlsx/pptx auto-extract), `write`, `edit`, `glob`, `grep` (`harness/tools.py:36-194`). **No `Validate` tool.**
- **Scaffold assembly (confirmed):** `run.py:304-311` builds `system_prompt = system_prompt.md (preamble) + load_skills(selected)`; each skill's `SKILL.md` is concatenated and its helper **scripts are copied into the workspace** (`run.py:190-195`) for the agent to invoke via `bash`; `--skills` selects/disables skills. `system_prompt.md` is currently **minimal** (workspace + tool conventions only — nothing on coverage, validation, or compaction → large Tier-1 headroom).
- **No context management — the key failure mode:** the `messages` list grows **unbounded** (every assistant turn + full tool result appended, `agent_loop.py:76-102`); no compaction/summarization. On window overflow the loop sets `context_overflow=True` and **breaks — the agent gives up** (`agent_loop.py:60-74`). This "fill-and-die" is an already-instrumented, **measurable** failure mode our compaction modules target directly.
- **Adapters:** Anthropic/OpenAI/Google/Mistral via `--model provider/id` (`harness/run.py:76-110`).
- **Knobs:** model, temperature, reasoning effort (CLI); max_turns (*hardcoded — patch to expose*); prompt+skills (files); judge model+parallel (CLI).
- **Run/score:** `uv run python -m harness.run --model ... --task ...` → `results/<run-id>/{output/, metrics.json, transcript.jsonl}`; `uv run python -m evaluation.run_eval --run-id ... --task ... --judge-model ...` → `scores.json`.
- **Tasks:** 1,251 across 24 practice areas; `task.json` (criteria w/ `match_criteria`+`deliverables`) + `documents/` (~100–300K). Deps: pandoc, uv, Podman; API keys via env.

## 4. System Architecture

```
   PER-TIER RESEARCHER  (OpenAI Agents SDK agent — one per tier T1/T2/T3)
   ┌───────────────────────────────────────────────────────────────────┐
   │ Agent loop (tools): read_state → propose K variants →               │
   │   evaluate(variants) → inspect_trace(id)? → set_champion → repeat   │
   │ Bounded context: curated state only (champion + lessons ledger +    │
   │   leaderboard + summarized failures). Raw data lives on the Volume. │
   └───────────────┬───────────────────────────────────┬────────────────┘
                   │ evaluate(variants)                 │ AgentHooks
                   ▼                                    ▼
        DETERMINISTIC CONTROLLER                 TRACKING LAYER
        successive-halving (§7):                 rich live view:
        screen K (subset) → PRUNE →              running / done / FAILED,
        promote survivors to dev                 round, best-so-far
                   │                                    ▲
                   ▼ dispatch (variant × task), ~20 concurrent             │
        MODAL FAN-OUT  @app.function(cpu=2)  × ~20                         │
        forked LAB harness + patched ToolExecutor (no nested Podman)       │
        + LC modules → run agent → output/ → judge → score ───────────────┘
                   │ {score, per-criterion, transcript, metrics, status}
                   ▼
        results Volume  (source of truth; researcher's external memory)
                   │
                   ▼  end: compare T1/T2/T3 champions on HOLDOUT → reports
```

## 5. Component Specs

### 5.1 Inner harness (forked Harvey LAB) — runs as a Modal Function
- **Interface:** `run_task(task_id, scaffold, model, max_turns) -> RunResult`; `score_run(run_id, task_id, judge_model) -> Scores{per_criterion, pass_rate, all_pass}`.
- **Mutation surface:** `scaffold = {system_prompt_md, skills:{name:md}, module_config}` (§6) materialized before each run; exposed `max_turns`.
- **Changes:** patch `ToolExecutor` to exec in-container (skip Podman); expose `max_turns`; inject scaffold; module hooks (§6).

### 5.2 Modal execution layer
- **Interface:** `evaluate.map(jobs)`, `job={scaffold_id, task_id, model, judge_model}` → `{score, pass_rate, per_criterion, transcript_ref, cost, status}`.
- **Primitives:** `App`, `@app.function(cpu=2, timeout=...)`, `Image` (uv+pandoc+LAB deps+tasks), `Volume` (scaffolds+results), **concurrency ≈20**, retries on transient errors. Failed runs return `status=failed` + error (loop continues).

### 5.3 Eval & objective
- **Objective:** mean **criterion-pass-rate** (continuous); all-pass secondary. Track cost (tokens/$/wall-clock). Splits/allocation: §7.

### 5.4 Researcher agent (OpenAI Agents SDK) — one per tier
- **Pattern:** mirrors Modal's orchestrator + `SubAgentPool`; async pool drives ~20 concurrent eval subagents.
- **Tools:** `read_state()`; `evaluate(variants)` (delegates to the controller §5.5 → returns scores); `inspect_trace(run_id)` (selective memory retrieval); `set_champion(variant)`.
- **Loop:** read curated state → propose K (≈6) variants as diffs+hypotheses → evaluate → read results → update lessons ledger → set champion → next round (N rounds or plateau).
- **Anti-grid-search:** reject numeric-only diffs; each variant needs a behavioral hypothesis targeting a §6 long-context lever.
- **Model:** GPT (SDK-native) or Claude via litellm — verify early.

### 5.5 Search controller (deterministic)
- Successive-halving (§7): screen K on screen-set → prune to top-m → promote survivors to dev-set → return ranked scores + champion. Owns the Modal fan-out + concurrency cap.

### 5.6 Researcher memory & context management ("what if context fills?")
- **External store (Volume):** all raw runs/traces/scores — researcher never holds raw data.
- **Curated per-round state (bounded):** champion scaffold + compact **lessons ledger** + top-K leaderboard + summarized champion failure modes.
- **On-demand retrieval:** `inspect_trace` pulls one trace summary only when needed.
- **Re-instantiate each round** with curated state → context starts small every round; cannot fill. Ledger compacted by a summarization step if it grows (BRIDGE-style — same long-context technique we optimize in the inner agent).
- **Validated patterns (research):** `inspect_trace` = just-in-time retrieval; the Volume ledger = file-memo; if the researcher spawns sub-agents, use **context isolation** (1–2K distilled returns) but keep exact metrics/configs in the file memo, not prose (per the multi-agent-failure caveat).

### 5.7 Tracking / progress / failure observability
- **Source:** Agents SDK `AgentHooks` + controller callbacks + a **results JSONL** on the Volume (round, tier, variant_id, task, score, pass_rate, cost, **status/error**).
- **Live view:** `rich` progress — running / done / **failed** counts, current round per tier, best-so-far pass-rate. Failed runs inspectable via their transcript/error.

### 5.8 Reporting
- Pareto (pass-rate vs cost across rounds) + freedom-vs-payoff (T1/T2/T3 champions on holdout) + champion-scaffold diff + per-criterion improvement table.

### 5.9 Optional / out-of-scope
- **Raindrop Workshop (optional stretch):** single-trajectory replay for demo polish. Not load-bearing — §5.7 covers tracking.
- **Self-hosted GPU model: OUT of scope (solo).** Transfer shown via API models instead.

## 6. The long-context search space & mutation tiers

**Search space = an evidence-graded library of long-context modules** (verified research pass; caveats in §6a). Modules are **no-ops by default** so baseline == stock LAB; each switched-on module is a clean, attributable delta. **Central evidence finding:** these techniques have **STRONG** evidence for *keeping the agent alive past overflow* but **mixed** evidence for *accuracy* — and **lossy summarization demonstrably drops exact numeric/clause facts** (probe: task-central 3/3 preserved, obscure table/clause specifics 0/3). Since LAB rubrics check exact facts (e.g. "EBITDA $17.1M vs $16.8M"), the optimizer's real job is configs that **survive AND preserve facts** — keep reads lossless (clearing→re-fetch), externalize must-cite facts to a memo, reserve summary-compaction for reasoning.

Modules (priority = research build order, → hook point):
1. **Tool-result clearing (LOSSLESS) — LOW — STRONG.** Drop old `read` payloads from `messages`, keep the tool_use record + placeholder, keep N most-recent; re-fetchable on demand. *Fixes "fill-and-die" without dropping facts.* Hook: `agent_loop.py:76-102` (or Anthropic-native `clear_tool_uses`).
2. **File memo / externalized must-keep facts — LOW-MED — MODERATE.** Agent writes must-cite facts + running notes to a workspace file (reuses LAB `write`+workspace); externalize *before* clearing ages facts out. Hook: new `memory` skill.
3. **Bridge / read-list — LOW — STRONG (mechanism).** Every compaction carries forward "docs already read" → coverage, no re-reads. Harvey's BRIDGE. Hook: compaction `instructions` + workspace read-list.
4. **Summary-compaction of REASONING only — LOW/MED — STRONG bound, LOSSY accuracy.** Summarize reasoning near the limit; route must-cite facts away (use #1/#2); tune instructions to enumerate facts to retain. Orchestrator-primary; sparing on the legal agent.
5. **Coverage policy — LOW — Harvey +0.4.** Require ≥X% doc coverage before drafting; track read-set. Hook: prompt + loop check.
6. **Validate-then-revise — LOW — Harvey +1.5 (biggest lever).** Post-draft self-check vs an instruction-derived checklist, then revise. Control-flow wrapper, **not** a new tool. Hook: prompt + re-invoke wrapper.
7. **Contextual retrieval + reranker — MED — STRONG recall (not end-task).** Hybrid BM25+embedding + reranker over chunked docs, for oversized matters. **Complement, not replacement** for reading. Cheapest high-value piece = the reranker. Hook: `retrieval` skill + helper script.

**Descoped for 6h (evidence-backed):** RAPTOR (HIGH cost + lossy summary nodes); prompt-compression (LLMLingua/selective-context), HyDE, late chunking (**no verified accuracy evidence; inherit fact-dropping risk** — only behind self-verification).

**Mutation tiers (built incrementally; one researcher per tier; compared on holdout):**
- **Tier 1 — text:** edit `system_prompt.md` + `SKILL.md`, toggle `--skills` — induce coverage/validate/memo behaviors by prompt only. **MVP, guaranteed-runnable, build & run FIRST.**
- **Tier 2 — modules:** wire modules into `agent_loop.py` (clearing, compaction, coverage, validate-wrapper) + memory/retrieval skills; researcher composes/parameterizes them.
- **Tier 3 — code (gated):** contained edits to skill helper-scripts behind a compile/import + smoke-test gate. Unlocked only if T1–2 solid (~hour 4).

**Comparison:** one independent researcher per tier from the same stock-LAB baseline; champions compared on holdout → freedom-vs-payoff chart. If T3 isn't reached, T1-vs-T2 is still clean.

### 6a. Evidence caveats (verified research — for honesty + demo credibility)
- Strongest numbers (84% token cut; 35/49/67% retrieval-failure cuts) are **cost/recall**, not end-task accuracy.
- Accuracy numbers (29%/39%) are **single-vendor, internal, undisclosed-methodology** agentic-search — MODERATE; validate on our own eval.
- **Demonstrated risk:** lossy summarization drops exact table/clause facts (0/3) → never route must-cite facts through summary-compaction.
- **Do NOT cite (refuted in verification):** "skip RAG under 200K tokens" (0-3); "context-rot from n² attention" (1-2); the specific cookbook demo token-trajectory numbers (0-3); "memory quality depends on context-mgmt not retrieval mechanism" (0-3).
- Context features are **beta** (`context-management-2025-06-27`, `compact-2026-01-12`) — verify identifiers at build time.
- Key sources: Anthropic context-management / contextual-retrieval / effective-context-engineering / multi-agent posts; Letta memory benchmark; RAPTOR (ICLR'24); Factory.ai & Morph compression probes.

## 7. Objective, data splits & search allocation (how we subset & stop early)
- **Objective:** mean **criterion-pass-rate** (continuous); binary all-pass too sparse on small sets → secondary.
- **Three pools:** **screen** (2–3 tasks, prune only) → **dev** (8–12 tasks, optimize) → **holdout** (never used in decisions; final headline + tier comparison). **From scouting** (1,251 tasks; median 56 criteria, 287KB docs): draw from small *analysis* tasks (~23–32 criteria, 4–8 docs, 150–370KB) like `identify-issues-in-counterparty-*` / `compare-*-against-*`; reserve a couple of oversized matters (up to 2.2MB) as **overflow stress cases** for the compaction demo.
- **Per-round successive-halving:** (1) propose K≈6; (2) **screen** all K on screen-set (fanned out); (3) **prune** to top-m≈3 — *the rest never run the dev set*; (4) **promote** survivors to dev-set → round champion; (5) repeat N rounds / until plateau; (6) **final** champions on holdout.
- **Why subset with free budget:** wall-clock. Runs are minutes each; subset-screen + prune + ~20-wide fan-out maximizes signal-per-minute.
- **Noise/overfit control:** screen scores decide *pruning only*; headline always from the untouched holdout. Diverse screen set, generous top-m, temp 0.0 → reproducible subset scores, behavioral/general mutations only (no hardcoding answers).

## 8. Decisions Locked
1. **Domain:** Legal — fork Harvey LAB (MIT).
2. **Orchestration:** researcher/orchestrator on **OpenAI Agents SDK** (Modal orchestrator+SubAgentPool pattern); inner = LAB harness as a Modal Function.
3. **Compute:** closed API models only (no GPU, solo). $ and rate limits not constraints; **~20 concurrent**; wall-clock is the budget.
4. **Optimizer:** agent proposes → deterministic controller screens/prunes/promotes (successive-halving, §7).
5. **Modal exec:** Function + patched `ToolExecutor` (skip Podman).
6. **Inner model:** cheap (Haiku/Sonnet) in the loop; Sonnet/Opus for the API-only transfer demo.
7. **Tiers:** build T1 → T2 → T3 incrementally; **one researcher per tier**; champions compared on holdout. Tier 3 gated.
8. **Tracking:** Agents SDK hooks + results JSONL + `rich` live progress/failure view. **Raindrop optional**; GPU out of scope.

## 9. Six-Hour Timeline (solo)
- **0:00–0:45** — Fork LAB; **1 task run + scored** locally. *Hard gate.*
- **0:45–1:30** — Expose mutation surface (inject scaffold, expose max_turns); baseline pass-rate; carve screen/dev/holdout.
- **1:30–2:30** — Modal-ify (Function + patched ToolExecutor); ~20-wide `.map`; results Volume; **tracking layer (rich + JSONL) so the fan-out is visible**.
- **2:30–4:00** — **Tier 1 researcher (Agents SDK)** + successive-halving: 2–3 rounds, measurable lift. **MVP demo.**
- **4:00–5:00** — **Tier 2 modules** + researcher track (T3 only if solid).
- **5:00–5:40** — Reporting: Pareto + freedom-vs-payoff (holdout) + champion diff; API-model transfer check.
- **5:40–6:00** — OSS polish (README, MIT attribution to LAB), demo script. *(Raindrop replay if time.)*

## 10. OSS Plan
Artifact = **the optimizer** (`aso/`): Agents-SDK researcher + controller + Modal fan-out + LC module library + tracking + reporting, on a forked LAB. MIT, attribution to Harvey LAB. README: "autoresearch your agent's scaffold."

## 11. Success Criteria (definition of done)
1. Optimizer runs ≥2 rounds and **raises mean criterion-pass-rate on the holdout** over baseline (headline).
2. **Live progress view** shows running/done/**failed** across ~20 concurrent runs.
3. **Pareto** (pass-rate vs cost) moves outward across rounds.
4. **Champion-scaffold diff** shows a human-legible behavioral discovery — ideally a **fact-preserving compaction** strategy (clearing + file-memo over naive summarization) and/or validate-then-revise / coverage-first. **Track `context_overflow`-rate ↓** baseline → champion as the directly-measured survival win.
5. **Freedom-vs-payoff** chart: holdout pass-rate + cost of T1 vs T2 (vs T3 if reached) champions.
6. *(Cheap stretch)* same scaffold lifts a second API model (Sonnet/Opus).

## 12. Open / Verify-First
- ✅ RESOLVED (§3): scaffold assembly = `system_prompt.md` + concatenated `SKILL.md`s (`run.py:304-311`); skill scripts copied to workspace (`run.py:190-195`).
- Claude-as-researcher in the Agents SDK (litellm) vs GPT-native — pick early.
- `max_turns` is already a `run_agent` param (default 200) — thread a CLI/config flag. Module hooks go in the `messages` loop (`agent_loop.py:76-102`).
- Whether LAB tasks run without Podman after the ToolExecutor patch (core assumption).
- Stable Modal concurrency at ~20.

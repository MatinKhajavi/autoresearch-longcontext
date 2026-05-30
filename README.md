# ASO — Autoresearch Scaffold Optimizer

**An autonomous research loop that optimizes an AI agent's *scaffold* — its prompt, skills, and long-context machinery — instead of its weights.**

Harvey's research on legal agents found that the harness is a *learnable scaffold, not just an inference wrapper*: scaffold changes alone moved their all-pass rate 2.6–3.7×. They learned that scaffold with expensive post-training. **ASO learns it with an autoresearch loop on Modal** — a GPT‑5.5 researcher agent that reads failure traces, proposes *behavioral* scaffold mutations (not hyperparameter sweeps), and a successive-halving controller that fans evaluations out across ~20 Modal containers. Karpathy's "autonomous research swarm on a compute megastructure," pointed at the agent scaffold itself.

Built on the [Harvey LAB](https://github.com/harveyai/harvey-labs) legal-agent benchmark (MIT), vendored under [`harvey-labs/`](harvey-labs/).

---

## The idea in one diagram

```
  GPT-5.5 researcher (OpenAI Agents SDK, high reasoning)
      │  proposes behavioral scaffold variants + a hypothesis each
      ▼
  successive-halving controller        screen(3 tasks) → prune → promote(8 tasks)
      │  fans (variant × task) out
      ▼
  Modal: run_eval_job  × ~20 containers      forked LAB legal agent (Claude Haiku)
      │                                        runs in a LocalSandbox (no Podman),
      │                                        criterion-scoped LLM judge scores it
      ▼
  results → live rich progress + JSONL → researcher reads failures → next round
      │
      ▼
  champion scaffold evaluated on a held-out task set  →  the honest headline number
```

- **Researcher** → GPT‑5.5, native OpenAI Agents SDK, high reasoning effort.
- **Inner legal agent + judge** → Claude (Haiku in the loop), via Harvey LAB's own adapter.
- **Objective** → mean criterion-pass-rate (continuous; binary all-pass is too sparse to give a gradient on a small set).

## What gets optimized (the long-context module library)

The search space is an **evidence-graded** library of long-context techniques (from an adversarially-verified research pass — see `docs/superpowers/specs/`):

| Module | What | Evidence |
|---|---|---|
| **Tool-result clearing** | drop old tool-result payloads from context, keep the call record (re-fetchable) — fixes the agent's *fill-the-window-and-die* failure | **strong** for survival; native `clear_tool_uses_20250919` |
| Coverage policy | require broad document reading before drafting | Harvey +0.4 |
| Validate-then-revise | post-draft self-check against the instructions, then fix | Harvey +1.5 (biggest lever) |
| File memo / bridge | externalize must-keep facts; carry a read-list across compaction | strong (mechanism) |

Decisive caveat baked into the design: **lossy summarization silently drops exact numeric/clause facts** (probe: task-central facts 3/3 preserved, obscure table/clause facts 0/3). Legal rubrics check exactly those facts — so reads are kept lossless (clearing → re-fetch), and summary-compaction is reserved for reasoning, never facts.

## Why it uses Modal properly

Each agent-run is minutes long and reads 100–300 KB of documents over up to 200 tool-calling turns. ASO fans these out as autoscaling Modal Functions (`max_containers=20`), so the wall-clock is the slowest single run, not the sum — and the controller spends that parallel budget intelligently (screen cheaply, prune, then promote survivors).

## Repo layout

```
harvey-labs/aso/
  scaffold.py       # the mutable surface: system prompt + skills + module_config
  local_sandbox.py  # drop-in for Podman: runs tools in-container (Approach A)
  harness_api.py    # run_and_score: one task under an injected scaffold, +overflow capture
  modal_app.py      # Modal App: run_eval_job fanned out ~20-wide
  controller.py     # successive-halving (screen → prune → promote)
  researcher.py     # GPT-5.5 Agents-SDK agent: propose / evaluate / inspect_trace / set_champion
  tracking.py       # rich live progress (running/done/failed/overflow/best) + JSONL
  optimize.py       # the loop: baseline → researcher → champion holdout eval
  report.py         # headline table + Pareto + improvement curve + champion diff
harvey-labs/harness/adapters/anthropic.py  # +native tool-result clearing (gated by module_config)
docs/superpowers/   # spec + plan + the verified long-context research
```

## Run it

```bash
cd harvey-labs
uv sync

# keys (local): researcher=OpenAI, inner agent+judge=Anthropic
printf 'OPENAI_API_KEY=...\nANTHROPIC_API_KEY=...\n' > .env

# Modal: secret holds the inner agent's key(s); deploy the fan-out
modal secret create llm-keys ANTHROPIC_API_KEY=... OPENAI_API_KEY=...
uv run modal deploy aso/modal_app.py

# the autoresearch loop
uv run python -m aso.optimize --rounds 3
uv run python -m aso.report          # -> results/aso/report.md + PNGs
```

## Tests

```bash
cd harvey-labs && uv run pytest tests/aso/ -q
```

## Attribution & license

Built on **Harvey LAB** (https://github.com/harveyai/harvey-labs), MIT-licensed, vendored under `harvey-labs/` with its `LICENSE` intact. ASO's additions are under the same terms.

# Autoresearch Scaffold Optimizer (ASO) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autoresearch loop that mutates a forked Harvey-LAB legal agent's scaffold (prompt/skills + long-context modules) and measurably raises criterion-pass-rate on a holdout set, fanned out ~20-wide on Modal, driven by an OpenAI Agents SDK researcher with a live progress view.

**Architecture:** A researcher Agent (OpenAI Agents SDK, Claude via LiteLLM) proposes scaffold variants → a deterministic successive-halving controller fans each (variant × task) out to Modal Functions (~20 concurrent) → each Function runs our forked LAB harness (Podman replaced by an in-process `LocalSandbox`) with an in-memory scaffold, then scores it criterion-by-criterion → results stream to a `rich` progress view and a results JSONL on a Volume → the researcher reads curated state + traces and iterates → champions are compared on a holdout set.

**Tech Stack:** Python 3.12, `uv`; forked `harvey-labs` (MIT); Modal `1.4.x` (`App`, `Image`, `Volume`, `Function.map`, `max_containers`); OpenAI Agents SDK `0.17.x` (`Agent`, `Runner`, `function_tool`, `RunHooks`, `LitellmModel`); `anthropic` `0.105.x` (native context-editing `clear_tool_uses_20250919` for the clearing module); `rich` (progress); `pydantic`.

**Scope of THIS plan:** Phases 0–5 = the demoable MVP **plus** the headline long-context module (tool-result clearing + overflow metric). Tier-2's remaining modules (file-memo, compaction, coverage, validate-revise, retrieval), Tier-3 code-editing, and the parallel-tier freedom-vs-payoff comparison are listed as **Follow-on** (§ Backlog) — each becomes its own plan once the MVP lands. Spec: `docs/superpowers/specs/2026-05-30-autoresearch-scaffold-optimizer-design.md`.

**Working directory:** all `uv run`/`python` commands run from `harvey-labs/` (so `harness`, `evaluation` import natively). Our code lives in `harvey-labs/aso/`.

---

## File Structure

**New (ours), all under `harvey-labs/aso/`:**
- `aso/__init__.py`
- `aso/local_sandbox.py` — `LocalSandbox`: drop-in for Podman `Sandbox` (same attrs + `.exec()`), runs commands via `subprocess` in the current container. *Makes Approach A work.*
- `aso/scaffold.py` — `Scaffold` (pydantic): `system_prompt` text + `skills: dict[str,str]` + `module_config: dict`. Helpers: `baseline()` (read stock files), `diff(other)`.
- `aso/harness_api.py` — `run_and_score(task, scaffold, model, judge_model, max_turns) -> EvalResult`. Replicates `run.py` logic with an **injected in-memory scaffold** + `LocalSandbox`, captures `context_overflow`, then calls `evaluation.run_eval.evaluate_run`. Returns an `EvalResult` (pydantic).
- `aso/modal_app.py` — Modal `App`, `Image`, `Volume`, `@app.function(max_containers=20) run_eval_job(job)`.
- `aso/controller.py` — successive-halving: `screen → prune → promote`; dispatches via Modal `.starmap`; pure allocation logic is unit-tested.
- `aso/researcher.py` — Agents SDK researcher: tools (`propose`/`evaluate`/`inspect_trace`/`set_champion`), `LitellmModel`, `RunHooks`, bounded per-round state.
- `aso/tracking.py` — `rich` live progress (running/done/**failed**, round, best-so-far) + results JSONL writer.
- `aso/datasets.py` — pinned `SCREEN`, `DEV`, `HOLDOUT` task-id lists.
- `aso/report.py` — Pareto (pass-rate vs cost) + improvement curve + champion diff → markdown/PNG.
- `aso/optimize.py` — top-level entrypoint wiring researcher + controller + tracking for one tier.
- `tests/aso/` — unit tests for pure logic (`scaffold`, `controller` allocation, clearing config).

**Modified (forked LAB):**
- `harness/adapters/anthropic.py` — add optional native context-editing (clearing) controlled by `module_config` (Phase 5).
- (No edit to `system_prompt.md`/`run.py` needed — we wrap, not mutate.)

---

## Phase 0 — Setup & Hard Gate (target: 45 min)

### Task 1: Vendor harvey-labs, scaffold `aso/`, install deps

**Files:** Create `harvey-labs/aso/__init__.py`, `harvey-labs/.env`; modify `harvey-labs/pyproject.toml`.

- [ ] **Step 1: Vendor the fork into our repo** (one git history). From repo root:

```bash
cd /Users/matin/Desktop/Projects/autoresearch-longcontext
rm -rf harvey-labs/.git          # vendor: LAB is MIT; attribution stays in harvey-labs/LICENSE + our README
git -C "$PWD" add harvey-labs && git -C "$PWD" commit -m "Vendor harvey-labs fork (MIT) as ASO substrate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
Expected: large commit adding harvey-labs/ tracked files.

- [ ] **Step 2: Add our deps + create package.**

```bash
cd harvey-labs
uv add openai-agents litellm rich pydantic modal   # adds to pyproject; resolves
mkdir -p aso tests/aso && : > aso/__init__.py
```
Expected: `uv` resolves; `openai-agents`, `litellm`, `rich`, `modal` in `pyproject.toml`.

- [ ] **Step 3: Create `.env` with keys** (run.py auto-loads it from BENCH_ROOT):

```bash
printf 'ANTHROPIC_API_KEY=%s\nOPENAI_API_KEY=%s\n' "$ANTHROPIC_API_KEY" "$OPENAI_API_KEY" > .env
echo ".env" >> .gitignore
```
Expected: `.env` exists, gitignored.

- [ ] **Step 4: Confirm podman is available OR plan LocalSandbox now.** Check: `podman --version`. If absent/slow, that's expected — Task 2 uses it once to validate the stock harness; from Task 4 on we use `LocalSandbox` and never need Podman again.

### Task 2: Hard gate — run + score ONE task with the stock harness

**Files:** none (uses stock LAB CLI). **This is the go/no-go gate.**

- [ ] **Step 1: Pick the smallest analysis task** (from scouting): `employment-labor/identify-issues-in-counterparty-motion-brief` (23 criteria, 8 docs, 370KB).

- [ ] **Step 2: Run the agent (cheap model).**

Run:
```bash
cd harvey-labs
uv run python -m harness.run --model anthropic/claude-haiku-4-5 \
  --task employment-labor/identify-issues-in-counterparty-motion-brief --max-turns 60
```
Expected: prints "Run complete", writes `results/<run-id>/{output/,metrics.json,transcript.jsonl,config.json}`. **If Podman fails to start**, jump to Task 4 (LocalSandbox) and re-run via a tiny wrapper instead — do not burn >15 min on Podman.

- [ ] **Step 3: Score it.**

Run (use the printed run-id):
```bash
uv run python -m evaluation.run_eval --run-id "<run-id>" \
  --task employment-labor/identify-issues-in-counterparty-motion-brief \
  --judge-model anthropic/claude-haiku-4-5
```
Expected: prints "N/23 criteria passed", writes `results/<run-id>/scores.json` with `score`, `all_pass`, `criteria_results`.

- [ ] **Step 4: GATE DECISION.** If a run+score completes in < ~8 min → proceed. Record the wall-clock and pass-rate. If it can't, stop and reassess scope (smaller task / fewer turns) before building further.

---

## Phase 1 — Scaffold injection + LocalSandbox + eval API (target: 45 min)

### Task 3: `LocalSandbox` — replace Podman for in-container tool exec

**Files:** Create `aso/local_sandbox.py`; Test `tests/aso/test_local_sandbox.py`. First confirm the contract.

- [ ] **Step 1: Read the Podman `Sandbox` interface to mirror it exactly.**

Run: `sed -n '1,120p' sandbox/sandbox.py` and note: the attributes `ToolExecutor` reads (`documents_dir`, `output_dir`, `workspace_dir`) and the **exact return type of `Sandbox.exec(command, timeout)`** (used at `harness/tools.py:401`). Mirror that return type precisely (likely an object/namedtuple with `.stdout`/`.returncode` or a string).

- [ ] **Step 2: Write the failing test.**

```python
# tests/aso/test_local_sandbox.py
from pathlib import Path
from aso.local_sandbox import LocalSandbox

def test_exec_runs_in_workspace(tmp_path):
    docs, out, ws = tmp_path/"d", tmp_path/"o", tmp_path/"w"
    for p in (docs, out, ws): p.mkdir()
    (docs/"hello.txt").write_text("hi")
    sb = LocalSandbox(documents_dir=docs, output_dir=out, workspace_dir=ws, default_timeout=10)
    sb.start()
    res = sb.exec("ls", timeout=10)            # cwd must be workspace_dir
    sb.stop()
    assert "hello" not in res_text(res)        # ls of workspace, not documents
```
Match `res_text`/return shape to whatever `Sandbox.exec` returns (Step 1).

- [ ] **Step 3: Run test → FAIL** (`ModuleNotFoundError: aso.local_sandbox`). `uv run pytest tests/aso/test_local_sandbox.py -x`.

- [ ] **Step 4: Implement `LocalSandbox`** with the SAME public surface as `Sandbox` but using `subprocess`:

```python
# aso/local_sandbox.py
import subprocess
from pathlib import Path

class LocalSandbox:
    """Drop-in for sandbox.sandbox.Sandbox that runs commands directly in the
    current process's container (no nested Podman). Safe on Modal: synthetic
    data, isolated container. Mirrors the attrs ToolExecutor reads + .exec()."""
    def __init__(self, documents_dir, output_dir, workspace_dir, default_timeout=60, **_):
        self.documents_dir = Path(documents_dir)
        self.output_dir = Path(output_dir)
        self.workspace_dir = Path(workspace_dir)
        self.default_timeout = default_timeout

    def start(self): self.workspace_dir.mkdir(parents=True, exist_ok=True)
    def stop(self): pass

    def exec(self, command, timeout=None):
        # MIRROR Sandbox.exec's return type from Step 1. If it returns an object
        # with .stdout/.returncode, return the same namedtuple/obj here.
        proc = subprocess.run(command, shell=True, cwd=str(self.workspace_dir),
                              capture_output=True, text=True,
                              timeout=timeout or self.default_timeout)
        return _make_result(proc)   # shape-match Sandbox.exec
```
Add `_make_result` to match the real contract. NOTE: `ToolExecutor` maps sandbox paths to host paths (`_sandbox_to_host_path`, tools.py:491); since LocalSandbox dirs ARE host dirs, ensure those mappings resolve to the real dirs (verify against tools.py:485-497 — they key off `documents_dir/output_dir/workspace_dir` which we set, so it works).

- [ ] **Step 5: Run test → PASS.** Commit: `git add aso/local_sandbox.py tests/aso/test_local_sandbox.py && git commit -m "feat(aso): LocalSandbox replaces Podman for in-container tool exec"`.

### Task 4: `Scaffold` model + baseline loader

**Files:** Create `aso/scaffold.py`; Test `tests/aso/test_scaffold.py`.

- [ ] **Step 1: Failing test.**

```python
# tests/aso/test_scaffold.py
from aso.scaffold import Scaffold
def test_baseline_loads_stock_prompt_and_skills():
    s = Scaffold.baseline()
    assert "Workspace layout" in s.system_prompt        # from harness/system_prompt.md
    assert set(s.skills) >= {"docx","xlsx","pptx"}
    assert s.module_config == {}
def test_render_system_prompt_concatenates_skills():
    s = Scaffold(system_prompt="PRE", skills={"docx":"DOCX"}, module_config={})
    rendered = s.render_system_prompt()
    assert rendered.startswith("PRE") and "## Skill: docx" in rendered and "DOCX" in rendered
```

- [ ] **Step 2: Run → FAIL.** `uv run pytest tests/aso/test_scaffold.py -x`.

- [ ] **Step 3: Implement.** Mirror `run.py:178-187` (`load_skills`) and `run.py:308-311` (concatenation) so renders are byte-identical to stock.

```python
# aso/scaffold.py
from pathlib import Path
from pydantic import BaseModel, Field
BENCH_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = BENCH_ROOT / "harness" / "skills"

class Scaffold(BaseModel):
    system_prompt: str
    skills: dict[str, str] = Field(default_factory=dict)   # name -> SKILL.md text
    module_config: dict = Field(default_factory=dict)      # e.g. {"clearing": {"trigger": 30000, "keep": 3}}

    @classmethod
    def baseline(cls) -> "Scaffold":
        preamble = (BENCH_ROOT/"harness"/"system_prompt.md").read_text(encoding="utf-8")
        skills = {p.parent.name: p.read_text(encoding="utf-8")
                  for p in sorted(SKILLS_DIR.glob("*/SKILL.md"))}
        return cls(system_prompt=preamble, skills=skills, module_config={})

    def render_system_prompt(self) -> str:
        out = self.system_prompt
        for name, text in self.skills.items():
            out += f"\n\n## Skill: {name}\n\n{text}"
        return out
```

- [ ] **Step 4: Run → PASS. Commit.** `git commit -am "feat(aso): Scaffold model + baseline loader"`.

### Task 5: `harness_api.run_and_score` — injected scaffold, LocalSandbox, overflow capture

**Files:** Create `aso/harness_api.py`; Test `tests/aso/test_harness_api.py` (integration, 1 cheap run).

- [ ] **Step 1: Define `EvalResult` + signature.**

```python
# aso/harness_api.py
import json, time, shutil
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel
from harness.run import BENCH_ROOT, load_task, create_adapter, setup_skill_scripts
from harness.agent_loop import run_agent
from harness.tools import ToolExecutor, get_all_tool_definitions
from evaluation.judge import Judge
from evaluation.run_eval import evaluate_run
from aso.local_sandbox import LocalSandbox
from aso.scaffold import Scaffold

class EvalResult(BaseModel):
    task: str; run_id: str; model: str
    pass_rate: float                 # n_passed / n_criteria  (continuous objective)
    all_pass: bool
    n_passed: int; n_criteria: int
    context_overflow: bool
    turn_count: int; input_tokens: int; output_tokens: int; wall_clock_seconds: float
    documents_read: int; total_documents: int
    failed_criteria: list[str]       # titles of failed criteria (for the researcher)
    status: str = "ok"               # "ok" | "failed"
    error: str | None = None
```

- [ ] **Step 2: Implement `run_and_score`** — replicate `run.py:234-351` but (a) inject `scaffold.render_system_prompt()` instead of reading the global file, (b) use `LocalSandbox`, (c) thread `module_config` to the adapter, (d) save `context_overflow` into metrics, then call `evaluate_run`.

```python
def run_and_score(task: str, scaffold: Scaffold, model: str, judge_model: str,
                  max_turns: int = 120, run_id: str | None = None) -> EvalResult:
    run_id = run_id or f"{task}/{model.split('/')[-1]}/{int(time.time()*1000)}"
    try:
        t = load_task(task)
        results_dir = BENCH_ROOT/"results"/run_id
        output_dir = results_dir/"output"; workspace_dir = results_dir/"workspace"
        output_dir.mkdir(parents=True, exist_ok=True); workspace_dir.mkdir(parents=True, exist_ok=True)

        sandbox = LocalSandbox(documents_dir=Path(t["docs_dir"]), output_dir=output_dir,
                               workspace_dir=workspace_dir, default_timeout=60)
        sandbox.start()
        # adapter: pass module_config so the clearing module (Phase 5) can enable native context-editing
        adapter = create_adapter(model=model, temperature=0.0)
        adapter.module_config = scaffold.module_config          # adapter reads this (Phase 5)
        tool_executor = ToolExecutor(sandbox=sandbox, shell_timeout=60)
        setup_skill_scripts(list(scaffold.skills), workspace_dir)

        system_prompt = scaffold.render_system_prompt()
        try:
            r = run_agent(adapter=adapter, system_prompt=system_prompt, user_prompt=t["instructions"],
                          tool_executor=tool_executor, tools=get_all_tool_definitions(),
                          max_turns=max_turns, transcript_path=str(results_dir/"transcript.jsonl"))
        finally:
            sandbox.stop()

        metrics = {"model": model, "task": task, "run_id": run_id,
                   "turn_count": r["turn_count"], "input_tokens": r["input_tokens"],
                   "output_tokens": r["output_tokens"], "wall_clock_seconds": r["wall_clock_seconds"],
                   "finished_cleanly": r["finished_cleanly"], "context_overflow": r["context_overflow"],
                   **r["tool_metrics"]}
        (results_dir/"metrics.json").write_text(json.dumps(metrics, indent=2))

        scores = evaluate_run(run_id=run_id, task=task, judge=Judge(model=judge_model), parallel=6)
        nP, nC = scores["n_passed"], scores["n_criteria"]
        failed = [c["title"] for c in scores["criteria_results"] if c["verdict"] != "pass"]
        return EvalResult(task=task, run_id=run_id, model=model,
            pass_rate=(nP/nC if nC else 0.0), all_pass=scores["all_pass"], n_passed=nP, n_criteria=nC,
            context_overflow=r["context_overflow"], turn_count=r["turn_count"],
            input_tokens=r["input_tokens"], output_tokens=r["output_tokens"],
            wall_clock_seconds=r["wall_clock_seconds"],
            documents_read=metrics.get("documents_read", 0), total_documents=metrics.get("total_documents", 0)
                          or metrics.get("total_vdr_files", 0),
            failed_criteria=failed)
    except Exception as e:
        return EvalResult(task=task, run_id=run_id, model=model, pass_rate=0.0, all_pass=False,
            n_passed=0, n_criteria=0, context_overflow=False, turn_count=0, input_tokens=0,
            output_tokens=0, wall_clock_seconds=0.0, documents_read=0, total_documents=0,
            failed_criteria=[], status="failed", error=f"{type(e).__name__}: {e}")
```
NOTE: existing adapters ignore an unknown `module_config` attribute until Phase 5 wires it in — safe.

- [ ] **Step 3: Integration test (1 cheap run).**

```python
# tests/aso/test_harness_api.py
import pytest
from aso.harness_api import run_and_score
from aso.scaffold import Scaffold
@pytest.mark.integration
def test_baseline_run_scores(monkeypatch):
    r = run_and_score("employment-labor/identify-issues-in-counterparty-motion-brief",
                      Scaffold.baseline(), model="anthropic/claude-haiku-4-5",
                      judge_model="anthropic/claude-haiku-4-5", max_turns=40)
    assert r.status == "ok" and 0.0 <= r.pass_rate <= 1.0 and r.n_criteria == 23
```
Run: `uv run pytest tests/aso/test_harness_api.py -x -m integration -s`. Expected: PASS, prints a pass_rate. **This proves Approach A (no Podman) end-to-end.**

- [ ] **Step 4: Commit.** `git commit -am "feat(aso): run_and_score with injected scaffold + LocalSandbox + overflow capture"`.

### Task 6: Pin the datasets + record baseline

**Files:** Create `aso/datasets.py`, `aso/baseline.py` (script); writes `results/aso/baseline.json`.

- [ ] **Step 1: Pin task lists** (from scouting; all are analysis/compare tasks, doc-heavy):

```python
# aso/datasets.py
SCREEN = [  # 3 tasks — fast pruning
    "immigration/compare-uscis-filing-receipt-against-original-petition-submission",
    "tax/review-iss-tax-transaction-structure",
    "banking-finance/identify-issues-in-borrower-financial-statements",
]
DEV = [  # 8 tasks — optimization signal
    "employment-labor/identify-issues-in-counterparty-motion-brief",
    "litigation-dispute-resolution/identify-issues-in-counterparty-complaint",
    "bankruptcy-restructuring/identify-issues-in-counterparty-sale-objection",
    "tax/identify-tax-issues-in-counterpartys-opposition-brief",
    "trusts-estates-private-client/identify-issues-in-counterpartys-proposed-parenting-plan",
    "environmental-esg/identify-issues-in-draft-permit-application",
    "intellectual-property/identify-issues-in-counterpartys-proposed-jury-instructions",
    "insurance/identify-issues-in-coverage-denial-letter",
]
HOLDOUT = [  # 5 tasks — never used in any decision
    "trusts-estates-private-client/compare-trust-documents-against-client-instructions",
    "immigration/compare-draft-eb",
    "international-trade-sanctions/review-draft-voluntary-self",
    "capital-markets/compare-closing-documents-against-closing-checklist",
    "corporate-governance/<pick-one-small>",   # confirm exact id via: ls tasks/corporate-governance
]
OVERFLOW_STRESS = []  # filled in Phase 5: 2 oversized matters (>1MB docs) to exercise compaction
```
Verify every id exists: `for t in $(...); do ls tasks/$t/task.json; done`. Fix the placeholder id.

- [ ] **Step 2: Baseline script** runs `Scaffold.baseline()` over DEV+HOLDOUT (locally, serial is fine — ~13 runs) and writes `results/aso/baseline.json` with per-task pass_rate + means + overflow count.

- [ ] **Step 3: Run baseline.** `uv run python -m aso.baseline`. Expected: `baseline.json` with `dev_mean_pass_rate`, `holdout_mean_pass_rate`, `overflow_count`. **This is the number to beat.** Commit.

---

## Phase 2 — Modal fan-out + tracking (target: 60 min)

### Task 7: `modal_app.py` — image, volume, eval function (~20 concurrent)

**Files:** Create `aso/modal_app.py`.

- [ ] **Step 1: Write the app** (API per Modal 1.4.x cheat-sheet):

```python
# aso/modal_app.py
import modal
from aso.harness_api import run_and_score
from aso.scaffold import Scaffold

app = modal.App("aso")
results_vol = modal.Volume.from_name("aso-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("pandoc")
    .pip_install_from_pyproject("pyproject.toml")     # harvey-labs deps + ours
    .add_local_python_source("harness", "evaluation", "sandbox", "utils", "aso")
    .add_local_dir("tasks", remote_path="/root/tasks")  # OR a curated subset dir; see note
)

@app.function(image=image, max_containers=20, timeout=60*30,
              secrets=[modal.Secret.from_name("llm-keys")],   # ANTHROPIC_API_KEY, OPENAI_API_KEY
              volumes={"/root/results": results_vol})
def run_eval_job(job: dict) -> dict:
    """job = {task, scaffold(dict), model, judge_model, max_turns, variant_id}"""
    s = Scaffold(**job["scaffold"])
    r = run_and_score(task=job["task"], scaffold=s, model=job["model"],
                      judge_model=job["judge_model"], max_turns=job.get("max_turns", 120))
    return {**r.model_dump(), "variant_id": job.get("variant_id")}
```
NOTE on tasks: `add_local_dir("tasks", ...)` bakes the task set into the image. tasks/ is ~428MB; for a lean image, first `cp -r` only the SCREEN+DEV+HOLDOUT+OVERFLOW task dirs into `aso/_devset_tasks/` and add THAT to `/root/tasks` instead. `BENCH_ROOT` in the container = the dir holding `harness/` — ensure `tasks/` sits beside it (or symlink). Confirm `load_task` resolves `/root/tasks/...` (set `cwd`/`BENCH_ROOT` accordingly; simplest: build image with WORKDIR at repo root).

- [ ] **Step 2: Create the Modal secret** (once): `modal secret create llm-keys ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY OPENAI_API_KEY=$OPENAI_API_KEY`.

- [ ] **Step 3: Smoke-test one remote run.** Add a `@app.local_entrypoint` that dispatches ONE job via `run_eval_job.remote(job)` and prints the result.

Run: `uv run modal run aso/modal_app.py`. Expected: image builds, one container runs the task remotely, prints `pass_rate`. **If `load_task` can't find tasks**, fix the image task path here before fanning out. Commit.

### Task 8: Tracking layer (`rich` progress + results JSONL)

**Files:** Create `aso/tracking.py`; Test `tests/aso/test_tracking.py` (counts only).

- [ ] **Step 1: Failing test for the counter logic** (pure): a `RunLedger` that records results and reports `running/done/failed/best_pass_rate`.

- [ ] **Step 2: Implement `RunLedger`** (pure dict/counters) + a `rich` `Progress`/`Live` view that renders it, + `append_jsonl(path, record)`.

```python
# aso/tracking.py  (sketch)
from rich.live import Live; from rich.table import Table
class RunLedger:
    def __init__(self): self.records=[]; self.running=0
    def start(self,n): self.running+=n
    def record(self, res: dict):
        self.running=max(0,self.running-1); self.records.append(res)
    @property
    def done(self): return sum(1 for r in self.records if r.get("status")=="ok")
    @property
    def failed(self): return sum(1 for r in self.records if r.get("status")=="failed")
    @property
    def best(self): return max((r.get("pass_rate",0) for r in self.records), default=0.0)
def render(ledger)->Table: ...   # running/done/FAILED/best/round
```

- [ ] **Step 3: Test → PASS. Commit.**

---

## Phase 3 — Tier-1 researcher + successive-halving (target: 90 min) — MVP

### Task 9: Controller (successive-halving) over Modal

**Files:** Create `aso/controller.py`; Test `tests/aso/test_controller.py` (pure allocation with a fake eval fn).

- [ ] **Step 1: Failing test — allocation logic with an injected `eval_fn`** (no Modal):

```python
# tests/aso/test_controller.py
from aso.controller import successive_halving
def fake_eval(jobs):   # jobs: list[dict] -> list[dict] with pass_rate
    return [{"variant_id": j["variant_id"], "task": j["task"],
             "pass_rate": 0.9 if "good" in j["variant_id"] else 0.1, "status":"ok"} for j in jobs]
def test_prunes_to_top_m_then_promotes():
    variants = {"good": {...}, "bad1": {...}, "bad2": {...}}   # scaffold dicts
    champ, table = successive_halving(variants, screen=["t1"], dev=["t1","t2"],
        eval_fn=fake_eval, keep_m=1, model="m", judge="j")
    assert champ == "good"           # pruned bad ones on screen, promoted good to dev
```

- [ ] **Step 2: Implement** `successive_halving(variants, screen, dev, eval_fn, keep_m, ...)`: build screen jobs (variant×screen) → `eval_fn` → mean pass_rate per variant → keep top `keep_m` → build dev jobs for survivors → `eval_fn` → return best variant_id + score table. `eval_fn` is injected (fake in tests; Modal `.starmap` in prod).

- [ ] **Step 3: Implement the Modal `eval_fn`** in `controller.py`:

```python
def modal_eval_fn(jobs):
    fn = modal.Function.from_name("aso", "run_eval_job")
    return list(fn.map(jobs, return_exceptions=True))   # ~20 concurrent (max_containers)
```
(`.map` over a single list-of-dict arg — positional, one iterable. Map exceptions → mark status failed.)

- [ ] **Step 4: Test → PASS. Commit.**

### Task 10: Researcher agent (Agents SDK)

**Files:** Create `aso/researcher.py`.

- [ ] **Step 1: Disable tracing + build the Claude model** (cheat-sheet: avoids OPENAI_API_KEY 401):

```python
# aso/researcher.py
import os, json
from agents import Agent, Runner, function_tool, RunHooks, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from pydantic import BaseModel
set_tracing_disabled(True)
RESEARCHER_MODEL = LitellmModel(model="anthropic/claude-opus-4-8", api_key=os.environ["ANTHROPIC_API_KEY"])

class VariantProposal(BaseModel):
    variants: list[dict]   # each: {id, hypothesis, system_prompt_patch | skill_edits}
```

- [ ] **Step 2: Define tools** the researcher calls. `propose` is the agent's structured output (not a tool); `evaluate`, `inspect_trace`, `set_champion` are tools. `evaluate` applies each variant's patch to the champion `Scaffold`, builds the variants dict, and calls `successive_halving(..., eval_fn=modal_eval_fn)`; returns a compact score table (NOT raw traces — bounded context).

```python
@function_tool
async def evaluate(variant_patches: list[dict]) -> dict:
    """Score proposed variants via screen→prune→promote on Modal. Returns {variant_id: pass_rate}."""
    variants = {v["id"]: apply_patch(CHAMPION[0], v) for v in variant_patches}   # CHAMPION is module-level state
    champ_id, table = successive_halving(variants, SCREEN, DEV, modal_eval_fn, keep_m=2, ...)
    return {"table": table, "round_best": champ_id}

@function_tool
def inspect_trace(run_id: str, max_chars: int = 2000) -> str:
    """Just-in-time retrieval of ONE failed run's transcript tail (bounded)."""
    p = BENCH_ROOT/"results"/run_id/"transcript.jsonl"
    return p.read_text()[-max_chars:] if p.exists() else "(no trace)"

@function_tool
def set_champion(variant_id: str) -> str:
    CHAMPION[0] = MATERIALIZED[variant_id]; return f"champion={variant_id}"
```

- [ ] **Step 3: Build the agent** with bounded instructions (curated state per round; Harvey priors seeded):

```python
researcher = Agent(name="scaffold-researcher", model=RESEARCHER_MODEL,
    output_type=VariantProposal,
    instructions=("You improve a legal agent's SCAFFOLD (system prompt + skill text) to raise "
      "criterion-pass-rate. Each round: read the failure summary, propose 4 variants as PROMPT/skill "
      "edits with a one-line behavioral hypothesis each (e.g. 'add a post-draft validation+revise pass', "
      "'require reading >=80% of documents before drafting'). Avoid numeric-only tweaks. "
      "Known priors: validate-then-revise and high pre-draft coverage raise scores; over-fanning tools hurts. "
      "Call evaluate(), then set_champion() on the winner. Keep going until gains plateau."),
    tools=[evaluate, inspect_trace, set_champion])
```

- [ ] **Step 4:** `apply_patch(scaffold, variant)` — applies `system_prompt_patch` (append/replace text) or `skill_edits` to a copy of the champion `Scaffold`, returns new Scaffold; store in `MATERIALIZED[id]`. (Tier-1 = text only; `module_config` untouched until Phase 5.) Commit.

### Task 11: `optimize.py` — wire researcher + hooks + tracking; run the loop

**Files:** Create `aso/optimize.py`.

- [ ] **Step 1: `RunHooks` → tracking** (cheat-sheet signatures): `on_tool_start` (mark "evaluating round"), `on_tool_end` (refresh `rich` view with ledger counts). The ledger is updated inside `modal_eval_fn` as results arrive.

- [ ] **Step 2: Run loop:** seed `CHAMPION=[Scaffold.baseline()]`, run `await Runner.run(researcher, "Begin. Baseline failures: <dev failure summary>", max_turns=40, hooks=ProgressHooks())`. Per-round curated state (champion diff + failure summary + score table) is passed as the input string each round; raw traces stay on disk (fetched via `inspect_trace`).

- [ ] **Step 3: Run it.** `uv run python -m aso.optimize --tier 1 --rounds 3`. Expected: live `rich` progress; ≥2 rounds; dev mean pass-rate rises vs baseline. Record champion scaffold.

- [ ] **Step 4: Evaluate champion on HOLDOUT** (the honest number): dispatch champion × HOLDOUT via Modal, compare to `baseline.json` holdout mean. Commit results + champion scaffold JSON.

---

## Phase 4 — Reporting (target: 40 min)

### Task 12: `report.py` — the demo artifacts

**Files:** Create `aso/report.py`; reads `results/aso/*.json`.

- [ ] **Step 1:** `improvement curve` — dev mean pass-rate per round (matplotlib PNG) + a markdown table baseline→champion (dev + holdout).
- [ ] **Step 2:** `Pareto` — scatter of (mean pass-rate vs mean cost/tokens) per evaluated variant, champions highlighted.
- [ ] **Step 3:** `champion diff` — unified diff of `Scaffold.baseline().system_prompt` vs champion (the human-legible discovered behavior).
- [ ] **Step 4:** `uv run python -m aso.report` → writes `results/aso/report.md` + PNGs. Commit. **MVP demo is now complete.**

---

## Phase 5 — Long-context module: tool-result clearing + overflow metric (target: 60 min)

### Task 13: Native context-editing in the Anthropic adapter (the clearing module)

**Files:** Modify `harness/adapters/anthropic.py`; Test `tests/aso/test_clearing_config.py`.

- [ ] **Step 1: Read `harness/adapters/anthropic.py`** — find the `chat()` method and how it calls `self.client.messages.create(...)`. Confirm it constructs `messages`/`tools`/`max_tokens`.

- [ ] **Step 2: Failing unit test for the config builder** (pure):

```python
# tests/aso/test_clearing_config.py
from harness.adapters.anthropic import build_context_management
def test_clearing_config_emitted_when_enabled():
    cm = build_context_management({"clearing": {"trigger": 30000, "keep": 3}})
    assert cm["edits"][0]["type"] == "clear_tool_uses_20250919"
    assert cm["edits"][0]["trigger"] == {"type":"input_tokens","value":30000}
def test_no_config_when_disabled():
    assert build_context_management({}) is None
```

- [ ] **Step 3: Implement** `build_context_management(module_config)` + wire into `chat()` (verified API):

```python
# in harness/adapters/anthropic.py
def build_context_management(module_config: dict | None):
    cfg = (module_config or {}).get("clearing")
    if not cfg: return None
    return {"edits": [{
        "type": "clear_tool_uses_20250919",
        "trigger": {"type": "input_tokens", "value": cfg.get("trigger", 100000)},
        "keep": {"type": "tool_uses", "value": cfg.get("keep", 3)},
        "clear_at_least": {"type": "input_tokens", "value": cfg.get("clear_at_least", 5000)},
    }]}

# in chat(): use the beta endpoint + header when enabled
cm = build_context_management(getattr(self, "module_config", None))
if cm:
    resp = self.client.beta.messages.create(..., betas=["context-management-2025-06-27"],
                                             context_management=cm)
else:
    resp = self.client.messages.create(...)   # unchanged path
```
(Keep full local `messages`; clearing is server-side. This removes the overflow-and-die for Claude.)

- [ ] **Step 4: Test → PASS. Commit.** `git commit -am "feat(harness): native tool-result clearing in Anthropic adapter, gated by module_config"`.

### Task 14: Overflow-stress tasks + Tier-2 clearing run

**Files:** modify `aso/datasets.py` (`OVERFLOW_STRESS`); reuse `optimize.py`.

- [ ] **Step 1: Pick 2 oversized matters** (>1MB docs) via `ls -lS tasks/*/*/documents` heuristics; add to `OVERFLOW_STRESS`.
- [ ] **Step 2: Confirm baseline overflow.** Run baseline scaffold (no clearing) on OVERFLOW_STRESS with `max_turns=200` → expect `context_overflow=True` on at least one. Record.
- [ ] **Step 3: Tier-2 variant** = champion + `module_config={"clearing":{"trigger":30000,"keep":3}}`. Let the researcher toggle/tune it (extend `apply_patch` to allow `module_config` edits in Tier-2). Run `aso.optimize --tier 2 --rounds 2` over DEV+OVERFLOW_STRESS.
- [ ] **Step 4: Verify the win:** clearing variant shows `context_overflow` rate ↓ (ideally →0) AND pass_rate ≥ baseline on the stress tasks. Add to report (overflow-rate bar + the clearing config in the champion diff). Commit. **This is the long-context payoff.**

---

## Self-review checklist (run before execution)
- Spec coverage: harness fork ✓(T1-5), Modal fan-out ✓(T7), tracking/progress+failures ✓(T8), researcher/agent-proposes ✓(T10), successive-halving subset/prune ✓(T9), holdout headline ✓(T11), reporting/Pareto ✓(T12), long-context module + overflow metric ✓(T13-14). Tier-2-rest/Tier-3/parallel-tracks → Backlog (intentional).
- Types consistent: `Scaffold`, `EvalResult`, `run_and_score`, `successive_halving`, `modal_eval_fn`, `build_context_management` used identically across tasks. ✓

## Backlog (each its own follow-on plan after MVP)
- **Tier-2 remaining modules:** file-memo (Anthropic `memory_20250818` / workspace file), summary-compaction (`compact_20260112`, reasoning-only + bridge read-list), coverage policy (prompt + check on `documents_read`), validate-then-revise wrapper, contextual-retrieval + reranker skill.
- **Tier-3:** researcher edits skill helper-scripts behind a compile/import smoke-test gate.
- **Parallel tier tracks + freedom-vs-payoff chart:** run T1/T2/T3 researchers from the same baseline; compare champions on holdout.
- **Model-agnostic clearing/compaction** in `agent_loop.py` (subagent provided the pure-Python logic) for non-Claude inner models.

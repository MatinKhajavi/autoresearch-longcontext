"""run_and_score — run ONE LAB task under an injected Scaffold, then score it.

This is the single unit the autoresearcher fans out. It replicates the wiring
in `harness.run.main` but with three changes:
  1. the system prompt comes from a `Scaffold` (in-memory), not the global file
     — so 20 parallel runs can use 20 different scaffolds safely;
  2. tools run in a `LocalSandbox` (no nested Podman);
  3. `context_overflow` is captured into metrics and surfaced (the directly-
     measurable "fill-and-die" signal the long-context modules target).

Returns an `EvalResult` (always — failures are captured as status="failed",
never raised — so a single bad run never crashes a fan-out).
"""

import json
import time
from pathlib import Path

from pydantic import BaseModel

from harness.run import BENCH_ROOT, _load_env, create_adapter, load_task, setup_skill_scripts
from harness.agent_loop import run_agent
from harness.tools import ToolExecutor, get_all_tool_definitions
from evaluation.judge import Judge
from evaluation.run_eval import evaluate_run

from aso.local_sandbox import LocalSandbox
from aso.scaffold import Scaffold

_load_env()  # load BENCH_ROOT/.env if present (no-op on Modal where Secret injects env)


class EvalResult(BaseModel):
    task: str
    run_id: str
    model: str
    pass_rate: float          # n_passed / n_criteria — the continuous objective
    all_pass: bool
    n_passed: int
    n_criteria: int
    context_overflow: bool
    turn_count: int
    input_tokens: int
    output_tokens: int
    wall_clock_seconds: float
    documents_read: int
    total_documents: int
    failed_criteria: list[str] = []   # titles of failed criteria (signal for the researcher)
    status: str = "ok"                # "ok" | "failed"
    error: str | None = None
    variant_id: str | None = None


def _slug(s: str) -> str:
    return s.replace("/", "_")


def run_and_score(
    task: str,
    scaffold: Scaffold,
    model: str = "anthropic/claude-haiku-4-5",
    judge_model: str = "claude-haiku-4-5",
    max_turns: int = 120,
    run_id: str | None = None,
    variant_id: str | None = None,
    judge_parallel: int = 6,
) -> EvalResult:
    run_id = run_id or f"aso/{_slug(task)}/{model.split('/')[-1]}/{int(time.time() * 1000)}"

    def _fail(e: Exception) -> EvalResult:
        return EvalResult(
            task=task, run_id=run_id, model=model, pass_rate=0.0, all_pass=False,
            n_passed=0, n_criteria=0, context_overflow=False, turn_count=0,
            input_tokens=0, output_tokens=0, wall_clock_seconds=0.0,
            documents_read=0, total_documents=0, failed_criteria=[],
            status="failed", error=f"{type(e).__name__}: {e}", variant_id=variant_id,
        )

    try:
        t = load_task(task)
        results_dir = BENCH_ROOT / "results" / run_id
        output_dir = results_dir / "output"
        workspace_dir = results_dir / "workspace"
        output_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir.mkdir(parents=True, exist_ok=True)

        sandbox = LocalSandbox(
            documents_dir=Path(t["docs_dir"]),
            output_dir=output_dir,
            workspace_dir=workspace_dir,
            default_timeout=60,
        )
        sandbox.start()

        adapter = create_adapter(model=model, temperature=0.0)
        # The clearing/compaction modules (Phase 5) read this; harmless before then.
        adapter.module_config = scaffold.module_config

        tool_executor = ToolExecutor(sandbox=sandbox, shell_timeout=60)
        setup_skill_scripts(list(scaffold.skills), workspace_dir)

        system_prompt = scaffold.render_system_prompt()
        try:
            r = run_agent(
                adapter=adapter,
                system_prompt=system_prompt,
                user_prompt=t["instructions"],
                tool_executor=tool_executor,
                tools=get_all_tool_definitions(),
                max_turns=max_turns,
                transcript_path=str(results_dir / "transcript.jsonl"),
            )
        finally:
            sandbox.stop()

        metrics = {
            "model": model, "task": task, "run_id": run_id,
            "turn_count": r["turn_count"], "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"], "wall_clock_seconds": r["wall_clock_seconds"],
            "finished_cleanly": r["finished_cleanly"], "context_overflow": r["context_overflow"],
            **r["tool_metrics"],
        }
        (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        scores = evaluate_run(run_id=run_id, task=task, judge=Judge(model=judge_model),
                              parallel=judge_parallel)
        nP, nC = scores["n_passed"], scores["n_criteria"]
        failed = [c["title"] for c in scores["criteria_results"] if c["verdict"] != "pass"]

        return EvalResult(
            task=task, run_id=run_id, model=model,
            pass_rate=(nP / nC if nC else 0.0), all_pass=scores["all_pass"],
            n_passed=nP, n_criteria=nC, context_overflow=r["context_overflow"],
            turn_count=r["turn_count"], input_tokens=r["input_tokens"],
            output_tokens=r["output_tokens"], wall_clock_seconds=r["wall_clock_seconds"],
            documents_read=metrics.get("documents_read", 0),
            total_documents=(metrics.get("total_documents") or metrics.get("total_vdr_files") or 0),
            failed_criteria=failed, variant_id=variant_id,
        )
    except Exception as e:  # never crash a fan-out
        return _fail(e)

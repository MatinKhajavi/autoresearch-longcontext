"""harness_overlay.py — run a Tier-3 (code-editing) variant against an EDITED
copy of the harness, in a throwaway subprocess. Karpathy-style isolation/reset:

  * the variant's `code_overrides` are applied to a COPY of harness/ in a temp
    overlay — the real harness is NEVER mutated (that IS our "reset");
  * a fresh `python -m aso.run_one` subprocess imports the edited harness, runs
    the agent + judge, and writes an EvalResult JSON (process exit = clean state);
  * a compile gate rejects un-parseable edits before we spend a run; a crashing
    or regressing edit just yields status="failed" (scored 0, never promoted).

Only files under harness/ may be edited. evaluation/ (the judge) and aso/ (the
optimizer/search) are symlinked read-only so the METRIC and the SEARCH stay
fixed — same discipline as Karpathy keeping prepare.py's eval read-only.
"""

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from aso.scaffold import BENCH_ROOT, Scaffold

EDITABLE_PREFIX = "harness/"
# COPIED into the overlay so each module's __file__/BENCH_ROOT resolves to the
# overlay (not the real repo). Required so the WRITER (harness) and the READER
# (evaluation's judge/scorer) agree on results/ — even on Modal, where the image
# excludes results/ so a symlink can't line them up. harness/ also receives edits.
_COPY_DIRS = ("harness", "evaluation")
# Symlinked (shared, read-only); results/ is created fresh under the overlay.
_SYMLINK_DIRS = ("aso", "sandbox", "utils", "tasks")


def check_code_overrides(overrides: dict[str, str]) -> tuple[bool, str]:
    """Gate (pure): each edit must be a relative .py path UNDER harness/ that parses
    as valid Python. Forbids touching evaluation/ (the judge), aso/, tasks/, etc."""
    for relpath, content in overrides.items():
        p = Path(relpath)
        if p.is_absolute() or ".." in p.parts:
            return False, f"{relpath}: must be a relative path"
        if not relpath.startswith(EDITABLE_PREFIX):
            return False, f"{relpath}: only files under {EDITABLE_PREFIX} may be edited"
        if p.suffix != ".py":
            return False, f"{relpath}: only .py files may be edited"
        try:
            ast.parse(content)
        except SyntaxError as e:
            return False, f"{relpath}: SyntaxError: {e.msg} (line {e.lineno})"
    return True, ""


def build_overlay(code_overrides: dict[str, str], repo: Path | None = None) -> Path:
    """Create a temp repo overlay: a deep COPY of harness/ (edits applied) plus
    symlinks to the unchanged dirs. Returns the overlay root (caller deletes it)."""
    repo = Path(repo or BENCH_ROOT)
    overlay = Path(tempfile.mkdtemp(prefix="aso_overlay_"))
    for name in _COPY_DIRS:
        src = repo / name
        if src.exists():
            shutil.copytree(src, overlay / name)
    for name in _SYMLINK_DIRS:
        src = repo / name
        if src.exists():
            (overlay / name).symlink_to(src)
    for relpath, content in code_overrides.items():
        target = overlay / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return overlay


def run_in_overlay(task, scaffold: Scaffold, model, judge_model, max_turns,
                   run_id, variant_id=None, judge_parallel=6, timeout=3300):
    """Run ONE code-editing variant in an isolated edited-harness subprocess.

    Returns an EvalResult (status="failed" on gate rejection, crash, or timeout —
    so a bad harness edit never crashes the fan-out and is simply not promoted).
    """
    from aso.harness_api import EvalResult  # lazy: avoid circular import

    def _fail(msg: str) -> "EvalResult":
        return EvalResult(
            task=task, run_id=run_id, model=model, pass_rate=0.0, all_pass=False,
            n_passed=0, n_criteria=0, context_overflow=False, turn_count=0,
            input_tokens=0, output_tokens=0, wall_clock_seconds=0.0,
            documents_read=0, total_documents=0, failed_criteria=[],
            status="failed", error=msg, variant_id=variant_id,
        )

    ok, err = check_code_overrides(scaffold.code_overrides)
    if not ok:
        return _fail(f"code-edit gate: {err}")

    overlay = build_overlay(scaffold.code_overrides)
    try:
        # Edits already live in the overlay's files, so clear code_overrides on the
        # scaffold the subprocess sees → it takes the normal in-process path inside
        # the edited harness (no recursion back into run_in_overlay).
        inner = scaffold.copy_with(code_overrides={})
        scaffold_path = overlay / "_scaffold.json"
        scaffold_path.write_text(json.dumps(inner.model_dump()), encoding="utf-8")
        out_path = overlay / "_result.json"

        env = os.environ.copy()
        env["PYTHONPATH"] = str(overlay)
        cmd = [
            sys.executable, "-m", "aso.run_one",
            "--scaffold", str(scaffold_path), "--task", task, "--out", str(out_path),
            "--model", model, "--judge-model", judge_model,
            "--max-turns", str(max_turns), "--run-id", run_id,
            "--variant-id", variant_id or "", "--judge-parallel", str(judge_parallel),
        ]
        try:
            proc = subprocess.run(cmd, cwd=str(overlay), env=env,
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return _fail(f"overlay run timed out after {timeout}s")

        if not out_path.exists():
            tail = (proc.stderr or proc.stdout or "")[-1000:]
            return _fail(f"overlay subprocess failed (rc={proc.returncode}): {tail}")
        result = EvalResult.model_validate_json(out_path.read_text(encoding="utf-8"))
        # Carry the transcript tail OUT of the overlay before it's deleted, so the
        # researcher's inspect_trace works for tier-3 runs too (P1 fix).
        tpath = overlay / "results" / run_id / "transcript.jsonl"
        if tpath.exists():
            result.transcript_tail = tpath.read_text(errors="replace")[-4000:]
        return result
    finally:
        shutil.rmtree(overlay, ignore_errors=True)

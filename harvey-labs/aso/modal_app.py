"""Modal app — fan one (scaffold x task) agent-run out to a container.

Each eval is one container running the forked LAB harness (via LocalSandbox,
no nested Podman). The controller's `make_modal_eval_fn` calls `run_eval_job`
~20-wide. Results return by value (pass_rate, overflow, a transcript tail for
the researcher's inspect_trace) — no shared Volume needed for the MVP.

Deploy once:   uv run modal deploy aso/modal_app.py
Smoke test:    uv run modal run aso/modal_app.py
"""

from pathlib import Path

import modal

REPO_LOCAL = Path(__file__).resolve().parent.parent   # .../harvey-labs
REPO_REMOTE = "/root/harvey-labs"

app = modal.App("aso")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("pandoc")
    .pip_install_from_pyproject(str(REPO_LOCAL / "pyproject.toml"))
    # Ship the whole repo (code + curated tasks) to a fixed path; exclude heavy
    # / irrelevant dirs. Tasks are baked in so workers need no network for data.
    .add_local_dir(
        str(REPO_LOCAL),
        remote_path=REPO_REMOTE,
        ignore=[
            "**/.venv", "**/.git", "**/results", "**/__pycache__",
            "**/.pytest_cache", "**/.playwright-mcp", "**/node_modules",
            "**/docs", "**/*.pyc",
        ],
    )
    .workdir(REPO_REMOTE)
    .env({"PYTHONPATH": REPO_REMOTE})
)


@app.function(
    image=image,
    max_containers=20,                       # cap parallel containers ~20
    timeout=60 * 30,                         # generous per-run budget
    secrets=[modal.Secret.from_name("llm-keys")],   # ANTHROPIC_API_KEY (+OPENAI optional)
)
def run_eval_job(job: dict) -> dict:
    """job = {variant_id, task, scaffold(dict), model, judge_model, max_turns}."""
    import sys
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)

    from aso.harness_api import run_and_score
    from aso.scaffold import Scaffold
    from harness.run import BENCH_ROOT

    scaffold = Scaffold(**job["scaffold"])
    r = run_and_score(
        task=job["task"],
        scaffold=scaffold,
        model=job.get("model", "anthropic/claude-haiku-4-5"),
        judge_model=job.get("judge_model", "claude-haiku-4-5"),
        max_turns=job.get("max_turns", 120),
        variant_id=job.get("variant_id"),
    )

    # Attach a transcript tail so the researcher's inspect_trace works off the
    # returned data (the container's filesystem is ephemeral).
    transcript_tail = ""
    tpath = Path(BENCH_ROOT) / "results" / r.run_id / "transcript.jsonl"
    if tpath.exists():
        transcript_tail = tpath.read_text(errors="replace")[-4000:]

    return {**r.model_dump(), "transcript_tail": transcript_tail}


@app.local_entrypoint()
def main(task: str = "employment-labor/identify-issues-in-counterparty-motion-brief"):
    """Smoke test: run ONE baseline job remotely and print the result."""
    import sys
    sys.path.insert(0, str(REPO_LOCAL))
    from aso.scaffold import Scaffold

    job = {
        "variant_id": "smoke",
        "task": task,
        "scaffold": Scaffold.baseline().model_dump(),
        "model": "anthropic/claude-haiku-4-5",
        "judge_model": "claude-haiku-4-5",
        "max_turns": 50,
    }
    res = run_eval_job.remote(job)
    print("SMOKE pass_rate=%.3f  passed=%d/%d  overflow=%s  status=%s  turns=%s" % (
        res.get("pass_rate", 0.0), res.get("n_passed", 0), res.get("n_criteria", 0),
        res.get("context_overflow"), res.get("status"), res.get("turn_count"),
    ))
    if res.get("status") == "failed":
        print("ERROR:", res.get("error"))

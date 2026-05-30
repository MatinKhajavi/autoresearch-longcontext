"""Local smoke of the Tier-3 overlay+subprocess path (no Modal).

A trivial, harmless harness override must still produce a real EvalResult via the
edited-harness subprocess — proving the whole plumbing: build overlay -> launch
`aso.run_one` subprocess -> import the EDITED harness -> run agent -> judge ->
serialize EvalResult -> read it back. If status == "ok", the path is live.

Run:  uv run python -m aso.smoke_tier3
"""

from harness.run import _load_env

_load_env()

from aso.harness_overlay import run_in_overlay
from aso.scaffold import Scaffold

TASK = "employment-labor/identify-issues-in-counterparty-motion-brief"

base = Scaffold.baseline()
# No-op but REAL edit: rewrite a tiny imported harness file. If the overlay's
# edited harness is what actually runs, this still completes end to end.
edited = base.copy_with(code_overrides={"harness/__init__.py": "# tier-3 overlay smoke (no-op)\n"})

r = run_in_overlay(
    task=TASK, scaffold=edited, model="anthropic/claude-haiku-4-5",
    judge_model="claude-haiku-4-5", max_turns=40,
    run_id="aso/tier3-smoke", variant_id="tier3-smoke", timeout=900,
)
print(f"TIER3 SMOKE: status={r.status} pass_rate={r.pass_rate:.3f} "
      f"overflow={r.context_overflow} n={r.n_passed}/{r.n_criteria} err={r.error}")

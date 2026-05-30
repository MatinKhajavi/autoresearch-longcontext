"""Modal smoke of the Tier-3 overlay path: dispatch ONE code-editing job to the
DEPLOYED run_eval_job so the overlay build + subprocess + edited-harness import
all happen INSIDE a Modal container (the part the local smoke can't prove). Run
after `modal deploy`. A trivial no-op harness edit must still return status=ok.

Run:  uv run python -m aso.smoke_tier3_modal
"""

import modal

from aso.scaffold import Scaffold

fn = modal.Function.from_name("aso", "run_eval_job")
edited = Scaffold.baseline().copy_with(
    code_overrides={"harness/__init__.py": "# modal tier-3 smoke (no-op)\n"}
)
job = {
    "variant_id": "t3-modal-smoke",
    "task": "employment-labor/identify-issues-in-counterparty-motion-brief",
    "scaffold": edited.model_dump(),
    "model": "anthropic/claude-haiku-4-5",
    "judge_model": "claude-haiku-4-5",
    "max_turns": 40,
}
r = fn.remote(job)
print(f"MODAL T3 SMOKE: status={r.get('status')} pass_rate={r.get('pass_rate')} "
      f"overflow={r.get('context_overflow')} turns={r.get('turn_count')} err={r.get('error')}")

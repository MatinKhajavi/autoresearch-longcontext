"""run_one.py — execute ONE run_and_score and write its EvalResult to --out.

Invoked as a subprocess by aso.harness_overlay inside an edited-harness overlay
(PYTHONPATH=overlay, cwd=overlay), so the harness imports below resolve to the
EDITED harness copy. Kept tiny on purpose. The scaffold it receives has had its
code_overrides cleared (the edits are already on disk in the overlay), so this
takes the normal in-process path.
"""

import argparse
import json
from pathlib import Path

from aso.harness_api import run_and_score
from aso.scaffold import Scaffold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaffold", required=True, help="path to scaffold JSON")
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True, help="path to write EvalResult JSON")
    ap.add_argument("--model", default="anthropic/claude-haiku-4-5")
    ap.add_argument("--judge-model", default="claude-sonnet-4-6")
    ap.add_argument("--max-turns", type=int, default=120)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--variant-id", default=None)
    ap.add_argument("--judge-parallel", type=int, default=6)
    a = ap.parse_args()

    scaffold = Scaffold(**json.loads(Path(a.scaffold).read_text(encoding="utf-8")))
    result = run_and_score(
        task=a.task, scaffold=scaffold, model=a.model, judge_model=a.judge_model,
        max_turns=a.max_turns, run_id=(a.run_id or None),
        variant_id=(a.variant_id or None), judge_parallel=a.judge_parallel,
    )
    Path(a.out).write_text(result.model_dump_json(), encoding="utf-8")


if __name__ == "__main__":
    main()

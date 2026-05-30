"""Tracking — a live progress view over the fan-out + a results JSONL log.

`RunLedger` is pure bookkeeping (unit-tested): how many runs are in flight,
done, failed, and the best pass-rate so far. `live()` renders it with `rich`
so you can watch ~20 concurrent agent-runs and spot failures in real time.
Every result is also appended to a JSONL on disk (the source of truth).
"""

import json
from contextlib import contextmanager
from pathlib import Path

from rich.live import Live
from rich.table import Table


class RunLedger:
    def __init__(self, round_label: str = ""):
        self.records: list[dict] = []
        self.running: int = 0
        self.round_label = round_label

    def start(self, n: int) -> None:
        self.running += n

    def record(self, res: dict) -> None:
        self.running = max(0, self.running - 1)
        self.records.append(res)

    @property
    def total(self) -> int:
        return len(self.records)

    @property
    def done(self) -> int:
        return sum(1 for r in self.records if r.get("status") == "ok")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.records if r.get("status") == "failed")

    @property
    def best(self) -> float:
        return max((r.get("pass_rate", 0.0) for r in self.records), default=0.0)

    @property
    def overflows(self) -> int:
        return sum(1 for r in self.records if r.get("context_overflow"))


def append_jsonl(path: str | Path, record: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record) + "\n")


def render_table(ledger: RunLedger) -> Table:
    t = Table(title=f"ASO fan-out{(' — ' + ledger.round_label) if ledger.round_label else ''}")
    for col in ("running", "done", "failed", "overflow", "best pass-rate"):
        t.add_column(col, justify="right")
    t.add_row(
        str(ledger.running),
        str(ledger.done),
        f"[red]{ledger.failed}[/red]" if ledger.failed else "0",
        str(ledger.overflows),
        f"{ledger.best:.3f}",
    )
    return t


@contextmanager
def live(ledger: RunLedger):
    """Context manager yielding a refresh() callback that redraws the table."""
    with Live(render_table(ledger), refresh_per_second=4) as live_view:
        def refresh() -> None:
            live_view.update(render_table(ledger))
        yield refresh

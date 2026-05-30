from aso.tracking import RunLedger, append_jsonl
import json


def test_ledger_counts_running_done_failed_best():
    led = RunLedger()
    led.start(3)
    assert led.running == 3
    led.record({"status": "ok", "pass_rate": 0.4})
    led.record({"status": "failed", "pass_rate": 0.0, "error": "boom"})
    led.record({"status": "ok", "pass_rate": 0.7, "context_overflow": True})
    assert led.running == 0
    assert led.done == 2
    assert led.failed == 1
    assert led.total == 3
    assert led.best == 0.7
    assert led.overflows == 1


def test_append_jsonl_roundtrip(tmp_path):
    p = tmp_path / "results.jsonl"
    append_jsonl(p, {"variant_id": "v1", "pass_rate": 0.5})
    append_jsonl(p, {"variant_id": "v2", "pass_rate": 0.6})
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert [r["variant_id"] for r in rows] == ["v1", "v2"]

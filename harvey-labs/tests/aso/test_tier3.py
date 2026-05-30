"""Tier-3 (free-form harness editing) pure logic: the compile/allowlist gate, the
overlay builder's isolation, and apply_patch's tier gating. No Modal / no agent
runs / no subprocess here."""

from aso.harness_overlay import build_overlay, check_code_overrides
from aso.researcher import CodeEdit, VariantSpec, apply_patch
from aso.scaffold import Scaffold


# ── gate ──────────────────────────────────────────────────────────────────
def test_gate_accepts_harness_py():
    ok, err = check_code_overrides({"harness/agent_loop.py": "x = 1\n"})
    assert ok and err == ""


def test_gate_rejects_non_harness_path():
    # the judge (evaluation/) must stay FIXED — editing it is rejected
    ok, err = check_code_overrides({"evaluation/judge.py": "x = 1\n"})
    assert not ok and "harness/" in err


def test_gate_rejects_non_py():
    ok, err = check_code_overrides({"harness/system_prompt.md": "hi"})
    assert not ok


def test_gate_rejects_path_escape():
    ok, err = check_code_overrides({"../evil.py": "x = 1\n"})
    assert not ok


def test_gate_rejects_syntax_error():
    ok, err = check_code_overrides({"harness/tools.py": "def f(:\n    pass\n"})
    assert not ok and "SyntaxError" in err


# ── overlay builder (against a fake repo) ─────────────────────────────────
def _fake_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "harness" / "adapters").mkdir(parents=True)
    (repo / "harness" / "agent_loop.py").write_text("ORIGINAL = 1\n")
    for d in ("aso", "evaluation", "tasks"):
        (repo / d).mkdir()
    (repo / "evaluation" / "judge.py").write_text("JUDGE = 1\n")
    return repo


def test_overlay_applies_edit_and_keeps_real_harness_intact(tmp_path):
    repo = _fake_repo(tmp_path)
    overlay = build_overlay({"harness/agent_loop.py": "ORIGINAL = 2\n"}, repo=repo)
    # edit landed in the overlay COPY ...
    assert (overlay / "harness" / "agent_loop.py").read_text() == "ORIGINAL = 2\n"
    # ... and the REAL harness is untouched (this is the "reset")
    assert (repo / "harness" / "agent_loop.py").read_text() == "ORIGINAL = 1\n"
    # the judge/scorer is COPIED (so its BENCH_ROOT == overlay) and stays stock
    assert (overlay / "evaluation").is_dir() and not (overlay / "evaluation").is_symlink()
    assert (overlay / "evaluation" / "judge.py").read_text() == "JUDGE = 1\n"


# ── apply_patch tier gating ───────────────────────────────────────────────
def _base():
    return Scaffold(system_prompt="P", skills={}, module_config={}, code_overrides={})


def test_apply_patch_ignores_code_edits_below_tier3():
    spec = VariantSpec(id="v", hypothesis="h",
        code_edits=[CodeEdit(path="harness/agent_loop.py", content="z = 1\n")])
    assert apply_patch(_base(), spec, tier=1).code_overrides == {}
    assert apply_patch(_base(), spec, tier=2).code_overrides == {}


def test_apply_patch_applies_code_edits_at_tier3():
    spec = VariantSpec(id="v", hypothesis="h",
        code_edits=[CodeEdit(path="harness/agent_loop.py", content="z = 1\n")])
    out = apply_patch(_base(), spec, tier=3)
    assert out.code_overrides == {"harness/agent_loop.py": "z = 1\n"}

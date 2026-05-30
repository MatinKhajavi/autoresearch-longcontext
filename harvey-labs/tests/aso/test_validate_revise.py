"""Tests for the Tier-2 validate-then-revise module in the agent loop."""

from harness.agent_loop import run_agent
from harness.adapters.base import ModelResponse
from aso.researcher import VariantSpec, apply_patch
from aso.scaffold import Scaffold


class _FakeAdapter:
    """Always 'done' (no tool calls), counts how many times chat() is called."""
    def __init__(self):
        self.calls = 0

    def make_system_message(self, c):
        return {"role": "system", "content": c}

    def make_user_message(self, c):
        return {"role": "user", "content": c}

    def make_tool_result_messages(self, results):
        return [{"role": "user", "content": "tool-results"}]

    def chat(self, messages, tools):
        self.calls += 1
        return ModelResponse(
            message={"role": "assistant", "content": "done"},
            tool_calls=[], text="done", input_tokens=1, output_tokens=1,
        )


class _FakeTool:
    def execute(self, name, args):
        return "ok"

    def get_metrics(self):
        return {}


def test_validate_revise_forces_one_extra_pass():
    a = _FakeAdapter()
    run_agent(a, "sys", "task", _FakeTool(), tools=[], max_turns=10,
              module_config={"validate_revise": 1})
    # 1st chat = agent 'done' -> forced validation injected -> 2nd chat = done -> stop
    assert a.calls == 2


def test_validate_revise_two_passes():
    a = _FakeAdapter()
    run_agent(a, "sys", "task", _FakeTool(), tools=[], max_turns=10,
              module_config={"validate_revise": 2})
    assert a.calls == 3


def test_no_module_stops_immediately():
    a = _FakeAdapter()
    run_agent(a, "sys", "task", _FakeTool(), tools=[], max_turns=10, module_config={})
    assert a.calls == 1


def test_apply_patch_gates_validate_revise_by_tier():
    base = Scaffold.baseline()
    spec = VariantSpec(id="v", hypothesis="h", validate_revise_passes=2)
    assert "validate_revise" not in apply_patch(base, spec, tier=1).module_config
    assert apply_patch(base, spec, tier=2).module_config["validate_revise"] == 2

from harness.adapters.anthropic import build_context_management


def test_clearing_config_emitted_when_enabled():
    cm = build_context_management({"clearing": {"trigger": 30000, "keep": 3}})
    edit = cm["edits"][0]
    assert edit["type"] == "clear_tool_uses_20250919"
    assert edit["trigger"] == {"type": "input_tokens", "value": 30000}
    assert edit["keep"] == {"type": "tool_uses", "value": 3}


def test_defaults_applied_when_partial():
    cm = build_context_management({"clearing": {}})
    edit = cm["edits"][0]
    assert edit["trigger"]["value"] == 100000   # default trigger
    assert edit["keep"]["value"] == 3            # default keep


def test_none_when_disabled():
    assert build_context_management({}) is None
    assert build_context_management(None) is None
    assert build_context_management({"something_else": 1}) is None

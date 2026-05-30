from aso.scaffold import Scaffold


def test_baseline_loads_stock_prompt_and_skills():
    s = Scaffold.baseline()
    assert "Workspace layout" in s.system_prompt          # from harness/system_prompt.md
    assert set(s.skills) >= {"docx", "xlsx", "pptx"}
    assert s.module_config == {}


def test_render_system_prompt_concatenates_skills():
    s = Scaffold(system_prompt="PRE", skills={"docx": "DOCX-BODY"}, module_config={})
    rendered = s.render_system_prompt()
    assert rendered.startswith("PRE")
    assert "## Skill: docx" in rendered
    assert "DOCX-BODY" in rendered


def test_baseline_render_matches_stock_assembly():
    """A baseline Scaffold must reproduce harness.run's system prompt exactly."""
    from harness.run import SYSTEM_PROMPT_PREAMBLE, load_skills, DEFAULT_SKILLS

    expected = SYSTEM_PROMPT_PREAMBLE + load_skills(DEFAULT_SKILLS)
    assert Scaffold.baseline().render_system_prompt() == expected


def test_copy_with_is_immutable():
    base = Scaffold.baseline()
    mutated = base.copy_with(system_prompt=base.system_prompt + "\n\nALWAYS VALIDATE.")
    assert "ALWAYS VALIDATE." in mutated.system_prompt
    assert "ALWAYS VALIDATE." not in base.system_prompt    # original untouched
    assert mutated.skills == base.skills

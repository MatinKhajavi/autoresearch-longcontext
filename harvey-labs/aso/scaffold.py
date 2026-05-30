"""Scaffold — the mutable surface the autoresearcher optimizes.

A Scaffold is the part of the harness that determines agent *behavior* without
changing the agent's tools or the model:

  - `system_prompt`: the preamble (workspace + tool conventions). Stock value
    lives in `harness/system_prompt.md`.
  - `skills`: name -> SKILL.md text. Stock values live in `harness/skills/*/`.
  - `module_config`: long-context module knobs (e.g. tool-result clearing).
    Empty by default so a baseline Scaffold reproduces stock LAB exactly.

`render_system_prompt()` reproduces `harness.run`'s assembly byte-for-byte
(preamble + concatenated skill manuals) so a baseline run is identical to the
stock harness. The autoresearcher mutates these fields to induce new
long-context behaviors (coverage-first, validate-then-revise, compaction, ...).
"""

from pathlib import Path

from pydantic import BaseModel, Field

BENCH_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT_PATH = BENCH_ROOT / "harness" / "system_prompt.md"
SKILLS_DIR = BENCH_ROOT / "harness" / "skills"


class Scaffold(BaseModel):
    system_prompt: str
    skills: dict[str, str] = Field(default_factory=dict)
    module_config: dict = Field(default_factory=dict)

    @classmethod
    def baseline(cls) -> "Scaffold":
        """Load the stock LAB scaffold (default prompt + all skills, no modules)."""
        preamble = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        skills = {
            p.parent.name: p.read_text(encoding="utf-8")
            for p in sorted(SKILLS_DIR.glob("*/SKILL.md"))
        }
        return cls(system_prompt=preamble, skills=skills, module_config={})

    def render_system_prompt(self) -> str:
        """Reproduce harness.run's assembly exactly: preamble + load_skills(...).

        `harness.run.load_skills` joins the per-skill sections with "\\n", so we
        must too — otherwise a "baseline" run wouldn't be byte-identical to stock.
        """
        sections = [f"\n\n## Skill: {name}\n\n{text}" for name, text in self.skills.items()]
        return self.system_prompt + "\n".join(sections)

    def copy_with(
        self,
        *,
        system_prompt: str | None = None,
        skills: dict[str, str] | None = None,
        module_config: dict | None = None,
    ) -> "Scaffold":
        """Return a new Scaffold with selected fields overridden (immutability helper)."""
        return Scaffold(
            system_prompt=system_prompt if system_prompt is not None else self.system_prompt,
            skills=skills if skills is not None else dict(self.skills),
            module_config=module_config if module_config is not None else dict(self.module_config),
        )

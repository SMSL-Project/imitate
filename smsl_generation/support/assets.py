from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

from smsl_generation.support.files import read_text
from smsl_generation.support.scenarios import GENERATION_DIR, get_scenario_spec


def render_template(template: str, replacements: Dict[str, str]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered


def _resolve_template_dir(style: str) -> str:
    """Return the common template directory for the given style."""
    return os.path.join(GENERATION_DIR, "templates", style, "common")


@dataclass(frozen=True)
class PromptAssets:
    direct_prompt: str
    indirect_prompt: str
    smsl_template: str
    phase_templates: List[str]

    @classmethod
    def load(cls, scenario_name: str, style: str = "detailed", mode: str = "indirect") -> "PromptAssets":
        scenario = get_scenario_spec(scenario_name)
        template_dir = _resolve_template_dir(style)

        # Indirect prompt: prompts/indirect/<style>/<scenario>/indirect.txt
        indirect_path = os.path.join(GENERATION_DIR, "prompts", "indirect", style, scenario.name, "indirect.txt")
        indirect_prompt = read_text(indirect_path) if os.path.exists(indirect_path) else ""

        # Direct prompt: prompts/direct/<style>/<scenario>/direct.txt
        direct_path = os.path.join(GENERATION_DIR, "prompts", "direct", style, scenario.name, "direct.txt")
        if not os.path.exists(direct_path):
            # Fall back to detailed style for direct prompts
            direct_path = os.path.join(GENERATION_DIR, "prompts", "direct", "detailed", scenario.name, "direct.txt")
        direct_prompt = read_text(direct_path) if os.path.exists(direct_path) else ""

        return cls(
            direct_prompt=direct_prompt,
            indirect_prompt=indirect_prompt,
            smsl_template=read_text(
                os.path.join(GENERATION_DIR, "templates", scenario.name, "smsl_template.json")
            ),
            phase_templates=[
                read_text(os.path.join(template_dir, "phase_1.py")),
                read_text(os.path.join(template_dir, "phase_2.py")),
                read_text(os.path.join(template_dir, "phase_3.py")),
                read_text(os.path.join(template_dir, "phase_4.py")),
            ],
        )

    def build_direct_prompt(self) -> str:
        return render_template(
            self.direct_prompt,
            {"SMSL_TEMPLATE": self.smsl_template.strip()},
        )

    def build_indirect_phase_prompt(self, phase_index: int) -> str:
        if not 1 <= phase_index <= len(self.phase_templates):
            raise ValueError("`phase_index` must be between 1 and {}.".format(len(self.phase_templates)))
        return render_template(
            self.indirect_prompt,
            {
                "SMSL_TEMPLATE": self.smsl_template.strip(),
                "CURRENT_PHASE_NUMBER": str(phase_index),
                "CURRENT_PHASE_TEMPLATE": self.phase_templates[phase_index - 1].strip(),
            },
        )

    def build_indirect_prompt(self) -> str:
        return self.build_indirect_phase_prompt(1)

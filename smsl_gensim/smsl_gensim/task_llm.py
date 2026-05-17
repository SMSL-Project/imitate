from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from smsl_gensim.scenes import ASSETS_DIR
from smsl_gensim.task_registry import ROOT_DIR


PROMPT_DIR = ROOT_DIR / "prompts" / "smsl_task_generation_prompt"


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    task_description: str
    urdf_to_use_fixed: tuple[str, ...]
    urdf_to_use_rigid: tuple[str, ...]
    urdf_to_inspect: tuple[str, ...]


def generate_llm_task_code(
    scenario: str,
    task_name: str,
    class_name: str,
    model: str,
    temperature: float,
    client: Any,
    validation_error: str | None = None,
) -> str:
    spec = _generate_task_spec(scenario, task_name, model, temperature, client)
    prompt = _code_prompt(scenario, class_name, spec, validation_error)
    return _extract_python_code(_call_openai(prompt, model, temperature, client, _CODE_SYSTEM))


def load_env_file(path: Union[str, Path] = ROOT_DIR / ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{env_path}:{line_number} must use KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{env_path}:{line_number} has an empty key")
        os.environ.setdefault(key, _env_value(value.strip()))


def _generate_task_spec(
    scenario: str,
    task_name: str,
    model: str,
    temperature: float,
    client: Any,
) -> TaskSpec:
    text = _call_openai(_task_spec_prompt(scenario, task_name), model, temperature, client, _SPEC_SYSTEM)
    return _parse_task_spec(text, task_name)


def _call_openai(prompt: str, model: str, temperature: float, client: Any, system: str) -> str:
    if client is None:
        from openai import OpenAI

        load_env_file()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in .env to use the llm backend")
        client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def _task_spec_prompt(scenario: str, task_name: str) -> str:
    return _render_prompt(
        "task_spec_prompt.txt",
        scenario=scenario,
        task_name=task_name,
        scene_initializer=_scenario_init_prompt(scenario),
    )


def _code_prompt(scenario: str, class_name: str, spec: TaskSpec, validation_error: str | None = None) -> str:
    prompt = _render_prompt(
        "task_code_prompt.txt",
        scenario=scenario,
        class_name=class_name,
        task_spec_json=json.dumps(_task_spec_dict(spec), indent=4),
        urdf_inspection=_inspection_prompt(spec.urdf_to_inspect),
    )
    if validation_error:
        prompt += (
            "\n\nValidation failed:\n"
            f"{validation_error}\n"
            "Regenerate the complete module and fix this validation error. Return only the Python code block."
        )
    return prompt


def _render_prompt(filename: str, **values: str) -> str:
    text = (PROMPT_DIR / filename).read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text.strip()


def _scenario_init_prompt(scenario: str) -> str:
    common = """
    env.smsl_reset()
    env.smsl_scene_state = state
    env.asset_ids_dict = {"fixed": {}, "rigid": {}, "deformable": {}}
    env.asset_poses_dict = {state: {}}
    for obj in scene_objects:
        obj_id = env.add_object(obj.urdf, obj.pose, obj.category, obj.color)
        env.asset_ids_dict[obj.category][obj.urdf] = obj_id
        env.asset_poses_dict[state][obj.urdf] = obj.pose
    Chess rigid objects may start with small random XY offsets inside their symbolic region.
    River_crossing rigid objects may start in permuted slots with small random XY offsets inside their symbolic region.
    """
    if scenario == "hanoi":
        return common + """
        scene_objects include fixed hanoi/stand_brown.urdf, hanoi/stand_red.urdf, hanoi/stand_blue.urdf.
        scene_objects include rigid hanoi/disk_gray.urdf, hanoi/disk_yellow.urdf, hanoi/disk_green.urdf.
        Disk poses vary by SMSL state. Stand poses are fixed supports.
        """
    if scenario == "chess":
        return common + """
        scene_objects include fixed chess/board.urdf.
        scene_objects include rigid chess/star_chess.urdf and chess/circle_chess.urdf.
        Chess target offsets are supplied by the code prompt. Do not inspect board slot joints.
        Chess piece target height is 0.04.
        """
    if scenario == "river_crossing":
        return common + """
        scene_objects include fixed minecraft/boat.urdf, minecraft/red_land.urdf, minecraft/green_land.urdf, minecraft/sea.urdf.
        scene_objects include rigid minecraft/grass.urdf, minecraft/sheep.urdf, minecraft/steve.urdf, minecraft/wolf.urdf.
        Start slots on each fixed place are permuted among (0.02, 0.02, 0.04), (-0.02, 0.02, 0.04),
        (-0.02, -0.02, 0.04), and (0.02, -0.02, 0.04).
        """
    return common


def _inspection_prompt(urdfs: tuple[str, ...]) -> str:
    if not urdfs:
        return "No URDF inspection requested."
    blocks = []
    for urdf in urdfs:
        path = ASSETS_DIR / urdf
        blocks.append(f"{urdf}\n```xml\n{path.read_text(encoding='utf-8')}\n```")
    return "\n\n".join(blocks)


def _extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if not match:
        raise ValueError("LLM response must contain one Python code block")
    return match.group(1).strip() + "\n"


def _parse_task_spec(text: str, task_name: str) -> TaskSpec:
    data = json.loads(_extract_json_text(text))
    if data["task-name"] != task_name:
        raise ValueError(f"Task spec name must be {task_name}")
    return TaskSpec(
        task_name=data["task-name"],
        task_description=data["task-description"],
        urdf_to_use_fixed=tuple(data["urdf-to-use-fixed"]),
        urdf_to_use_rigid=tuple(data["urdf-to-use-rigid"]),
        urdf_to_inspect=tuple(data["urdf-to-inspect"]),
    )


def _extract_json_text(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


def _task_spec_dict(spec: TaskSpec) -> dict:
    return {
        "task-name": spec.task_name,
        "task-description": spec.task_description,
        "urdf-to-use-fixed": list(spec.urdf_to_use_fixed),
        "urdf-to-use-rigid": list(spec.urdf_to_use_rigid),
        "urdf-to-inspect": list(spec.urdf_to_inspect),
    }


def _env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


_SPEC_SYSTEM = "You design concise SMSL task specifications. Return only valid JSON."
_CODE_SYSTEM = "You generate concise CLIPort Task Python modules. Return exactly one Python code block."

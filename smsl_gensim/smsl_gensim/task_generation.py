from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from smsl_gensim.scenarios import TASK_LIST_DIR, operation_names
from smsl_gensim.task_llm import generate_llm_task_code
from smsl_gensim.task_registry import GENERATED_TASKS_BACKUP_DIR, GENERATED_TASKS_DIR
from smsl_gensim.task_templates import render_task
from smsl_gensim.task_validation import validate_task_code


LLM_VALIDATION_ATTEMPTS = 3


@dataclass(frozen=True)
class GeneratedTask:
    task_name: str
    module_name: str
    class_name: str
    path: Path
    written: bool


@dataclass(frozen=True)
class TaskGenerationResult:
    scenario: str
    backend: str
    model: str
    output_dir: Path
    task_list_path: Path
    backup_dir: Path
    tasks: tuple[GeneratedTask, ...]

    @property
    def written_count(self) -> int:
        return sum(task.written for task in self.tasks)


def generate_scenario_tasks(
    scenario: str,
    output_dir: Union[str, Path] = GENERATED_TASKS_DIR,
    task_list_dir: Union[str, Path] = TASK_LIST_DIR,
    backend: str = "llm",
    model: str = "gpt-4o",
    temperature: float = 0.0,
    overwrite: bool = False,
    client: Any = None,
    progress: Any = None,
) -> TaskGenerationResult:
    if backend not in ("llm", "template"):
        raise ValueError("backend must be llm or template")

    output_path = Path(output_dir)
    task_list_path = Path(task_list_dir) / f"{scenario}.json"
    task_names = tuple(operation.replace("_", "-") for operation in operation_names(scenario))

    output_path.mkdir(parents=True, exist_ok=True)
    task_list_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_init(output_path)

    tasks = []
    generated = []
    progress_items = _progress_items(task_names, progress)
    for task_name in progress_items:
        module_name = task_name.replace("-", "_")
        class_name = _class_name(task_name)
        path = output_path / f"{module_name}.py"
        written = overwrite or not path.exists()
        _progress_description(progress_items, task_name)
        code = None
        if written:
            code = _generate_valid_task_code(scenario, task_name, class_name, backend, model, temperature, client)
        generated.append((path, code))
        tasks.append(GeneratedTask(task_name, module_name, class_name, path, written))
        _progress_postfix(progress_items, "ready" if written else "kept")

    _write_json(task_list_path, task_names)
    for path, code in generated:
        if code is not None:
            path.write_text(code, encoding="utf-8")

    return TaskGenerationResult(
        scenario=scenario,
        backend=backend,
        model=model if backend == "llm" else "",
        output_dir=output_path,
        task_list_path=task_list_path,
        backup_dir=GENERATED_TASKS_BACKUP_DIR,
        tasks=tuple(tasks),
    )


def _generate_valid_task_code(
    scenario: str,
    task_name: str,
    class_name: str,
    backend: str,
    model: str,
    temperature: float,
    client: Any,
) -> str:
    feedback = None
    attempts = LLM_VALIDATION_ATTEMPTS if backend == "llm" else 1
    for attempt in range(attempts):
        code = _generate_task_code(scenario, task_name, class_name, backend, model, temperature, client, feedback)
        try:
            validate_task_code(code, class_name, task_name)
            return code
        except ValueError as exc:
            if backend != "llm" or attempt == attempts - 1:
                raise
            feedback = str(exc)
    raise RuntimeError(f"{task_name} could not be generated")


def _generate_task_code(
    scenario: str,
    task_name: str,
    class_name: str,
    backend: str,
    model: str,
    temperature: float,
    client: Any,
    validation_error: str | None = None,
) -> str:
    if backend == "template":
        return render_task(task_name, class_name)
    return generate_llm_task_code(scenario, task_name, class_name, model, temperature, client, validation_error)


def _progress_items(items: tuple[str, ...], progress: Any):
    return progress(items) if progress else items


def _progress_description(progress: Any, text: str) -> None:
    if hasattr(progress, "set_description_str"):
        progress.set_description_str(text[:60])


def _progress_postfix(progress: Any, status: str) -> None:
    if hasattr(progress, "set_postfix_str"):
        progress.set_postfix_str(status)


def _class_name(task_name: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in task_name.replace("-", "_").split("_"))


def _write_json(path: Path, items: tuple[str, ...]) -> None:
    path.write_text(json.dumps(list(items), indent=4) + "\n", encoding="utf-8")


def _ensure_init(path: Path) -> None:
    init = path / "__init__.py"
    if not init.exists():
        init.write_text(_INIT_SOURCE, encoding="utf-8")


_INIT_SOURCE = '''from importlib import import_module

from smsl_gensim.task_registry import task_class_names


new_names = {}
for module_name, class_name in task_class_names().items():
    module = import_module(f"{__name__}.{module_name}")
    new_names[module_name.replace("_", "-")] = getattr(module, class_name)
'''

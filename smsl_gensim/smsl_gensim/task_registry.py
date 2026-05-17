from __future__ import annotations

import ast
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
GENERATED_TASKS_DIR = ROOT_DIR / "cliport" / "generated_tasks"
GENERATED_TASKS_BACKUP_DIR = ROOT_DIR / "cliport" / "generated_tasks_backup"


def task_class_names(directory: Path = GENERATED_TASKS_DIR) -> dict[str, str]:
    return {
        path.stem: _class_name(path)
        for path in sorted(directory.glob("*.py"))
        if path.stem != "__init__"
    }


def generated_task_names(directory: Path = GENERATED_TASKS_DIR) -> tuple[str, ...]:
    return tuple(name.replace("_", "-") for name in task_class_names(directory))


def _class_name(path: Path) -> str:
    classes = [
        node.name
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef)
    ]
    if len(classes) != 1:
        raise ValueError(f"{path} must define exactly one task class")
    return classes[0]

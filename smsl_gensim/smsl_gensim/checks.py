from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from smsl_gensim.scenarios import load_task_list
from smsl_gensim.task_registry import GENERATED_TASKS_DIR, ROOT_DIR, generated_task_names, task_class_names
from smsl_gensim.task_validation import validate_task_code


ASSETS_DIR = ROOT_DIR / "cliport" / "environments" / "assets"

REQUIRED_ASSETS = {
    "chess": (
        "chess/board.mtl",
        "chess/board.obj",
        "chess/board.png",
        "chess/board.urdf",
        "chess/circle_chess.urdf",
        "chess/star_chess.obj",
        "chess/star_chess.urdf",
    ),
    "hanoi": (
        "hanoi/disk0.obj",
        "hanoi/disk1.obj",
        "hanoi/disk2.obj",
        "hanoi/disk_gray.urdf",
        "hanoi/disk_green.urdf",
        "hanoi/disk_yellow.urdf",
        "hanoi/stand_blue.urdf",
        "hanoi/stand_brown.urdf",
        "hanoi/stand_red.urdf",
    ),
    "river_crossing": (
        "minecraft/boat.obj",
        "minecraft/boat.png",
        "minecraft/boat.urdf",
        "minecraft/grass.obj",
        "minecraft/grass.png",
        "minecraft/grass.urdf",
        "minecraft/green_land.urdf",
        "minecraft/red_land.urdf",
        "minecraft/sea.urdf",
        "minecraft/sheep.obj",
        "minecraft/sheep.png",
        "minecraft/sheep.urdf",
        "minecraft/steve.obj",
        "minecraft/steve.png",
        "minecraft/steve.urdf",
        "minecraft/wolf.obj",
        "minecraft/wolf.png",
        "minecraft/wolf.urdf",
    ),
}


@dataclass(frozen=True)
class TaskCheck:
    scenario: str
    task_count: int
    missing_task_files: tuple[Path, ...]
    missing_registry_names: tuple[str, ...]
    invalid_task_files: tuple[str, ...]
    missing_assets: tuple[Path, ...]

    @property
    def ok(self) -> bool:
        return not (
            self.missing_task_files
            or self.missing_registry_names
            or self.invalid_task_files
            or self.missing_assets
        )


def check_scenario_tasks(name: str) -> TaskCheck:
    tasks = tuple(load_task_list(name))
    registered = set(generated_task_names())
    return TaskCheck(
        scenario=name,
        task_count=len(tasks),
        missing_task_files=tuple(path for path in task_paths(tasks) if not path.exists()),
        missing_registry_names=tuple(task for task in tasks if task not in registered),
        invalid_task_files=_invalid_task_files(tasks),
        missing_assets=tuple(path for path in asset_paths(name) if not path.exists()),
    )


def task_paths(tasks: tuple[str, ...]) -> tuple[Path, ...]:
    return tuple(GENERATED_TASKS_DIR / f"{task.replace('-', '_')}.py" for task in tasks)


def asset_paths(name: str) -> tuple[Path, ...]:
    return tuple(ASSETS_DIR / path for path in REQUIRED_ASSETS[name])


def _invalid_task_files(tasks: tuple[str, ...]) -> tuple[str, ...]:
    class_names = task_class_names()
    invalid = []
    for task_name in tasks:
        module_name = task_name.replace("-", "_")
        path = GENERATED_TASKS_DIR / f"{module_name}.py"
        if not path.exists() or module_name not in class_names:
            continue
        try:
            validate_task_code(path.read_text(encoding="utf-8"), class_names[module_name], task_name)
        except ValueError as exc:
            invalid.append(f"{path}: {exc}")
    return tuple(invalid)

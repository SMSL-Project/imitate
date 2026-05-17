from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
SMSL_DIR = PACKAGE_DIR / "SMSL_JSON"
TASK_LIST_DIR = PACKAGE_DIR / "TASK_LIST"


@dataclass(frozen=True)
class Scenario:
    name: str
    graph_name: str
    initial_state: str
    state_count: int
    transition_count: int
    operation_count: int
    task_count: int


def available_scenarios() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in SMSL_DIR.glob("*.json")))


def load_smsl(name: str) -> dict[str, Any]:
    return _load_json(SMSL_DIR / f"{name}.json")


def load_task_list(name: str) -> list[str]:
    return _load_json(TASK_LIST_DIR / f"{name}.json")


def operation_names(name: str) -> tuple[str, ...]:
    _, body = _graph(load_smsl(name))
    return tuple(sorted({op for state, edges in body.items() if state != "HEADER" for op in edges}))


def state_names(name: str) -> tuple[str, ...]:
    _, body = _graph(load_smsl(name))
    return tuple(state for state in body if state != "HEADER")


def scenario_summary(name: str) -> Scenario:
    graph_name, body = _graph(load_smsl(name))
    states = [state for state in body if state != "HEADER"]
    transitions = sum(len(body[state]) for state in states)
    return Scenario(
        name=name,
        graph_name=graph_name,
        initial_state=body["HEADER"]["INITIAL"],
        state_count=len(states),
        transition_count=transitions,
        operation_count=len(operation_names(name)),
        task_count=len(load_task_list(name)),
    )


def _graph(data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return next(iter(data.items()))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

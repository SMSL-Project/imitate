from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

from smsl_gensim.scenarios import load_smsl, state_names
from smsl_gensim.scenes import check_scene
from smsl_gensim.task_registry import generated_task_names


@dataclass(frozen=True)
class Transition:
    scenario: str
    from_state: str
    operation: str
    to_state: str
    task_name: str


@dataclass(frozen=True)
class DemoPlanCheck:
    scenario: str
    transitions: Tuple[Transition, ...]
    missing_task_names: Tuple[str, ...]
    missing_scene_states: Tuple[str, ...]
    missing_scene_assets: Tuple[Path, ...]

    @property
    def transition_count(self) -> int:
        return len(self.transitions)

    @property
    def ok(self) -> bool:
        return not (
            self.missing_task_names
            or self.missing_scene_states
            or self.missing_scene_assets
        )


def iter_transitions(name: str) -> Iterator[Transition]:
    for from_state, edges in _body(name).items():
        if from_state != "HEADER":
            for operation, to_state in edges.items():
                yield Transition(
                    scenario=name,
                    from_state=from_state,
                    operation=operation,
                    to_state=to_state,
                    task_name=operation.replace("_", "-"),
                )


def check_demo_plan(name: str) -> DemoPlanCheck:
    transitions = tuple(iter_transitions(name))
    registered = set(generated_task_names())
    known_states = set(state_names(name))
    scene_states = _unique(
        tuple(t.from_state for t in transitions)
        + tuple(t.to_state for t in transitions)
    )
    valid_scene_states = tuple(state for state in scene_states if state in known_states)
    return DemoPlanCheck(
        scenario=name,
        transitions=transitions,
        missing_task_names=_unique(tuple(t.task_name for t in transitions if t.task_name not in registered)),
        missing_scene_states=tuple(state for state in scene_states if state not in known_states),
        missing_scene_assets=tuple(
            path
            for state in valid_scene_states
            for path in check_scene(name, state).missing_assets
        ),
    )


def _body(name: str) -> dict:
    return next(iter(load_smsl(name).values()))


def _unique(items: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(items))

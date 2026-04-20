from __future__ import annotations

import itertools
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Tuple


SUPPORT_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATION_DIR = os.path.dirname(SUPPORT_DIR)

VALID_CHESS_POSITIONS = {"A", "B", "C", "D", "a", "c", "d", "0", "1"}
RIVER_CODES = ("0", "B", "1")


@lru_cache(maxsize=None)
def _hanoi_state_names() -> Tuple[str, ...]:
    return tuple("State_" + "".join(state) for state in itertools.product("abc", repeat=3))


def _is_hanoi_state_syntax(name: str) -> bool:
    return re.fullmatch(r"State_[abc]{3}", name) is not None


def _is_valid_river_state_code(code: str) -> bool:
    grass, sheep, steve, wolf = code
    if sheep == wolf != steve and sheep != "B":
        return False
    if sheep == grass != steve and sheep != "B":
        return False

    boat_count = code.count("B")
    if boat_count > 2:
        return False
    if boat_count >= 2 and steve != "B":
        return False
    return True


@lru_cache(maxsize=None)
def _river_state_names() -> Tuple[str, ...]:
    return tuple(
        "State_" + "".join(code)
        for code in itertools.product(RIVER_CODES, repeat=4)
        if _is_valid_river_state_code("".join(code))
    )


@lru_cache(maxsize=None)
def _chess_state_names() -> Tuple[str, ...]:
    names = []
    positions = tuple(sorted(VALID_CHESS_POSITIONS, key="ABCDacd01".index))
    for star_position, circle_position in itertools.product(positions, repeat=2):
        if star_position == circle_position:
            names.append("State_{}T{}B".format(star_position, circle_position))
            names.append("State_{}B{}T".format(star_position, circle_position))
        else:
            names.append("State_{}B{}B".format(star_position, circle_position))
    return tuple(names)


def _is_hanoi_state_name(name: str) -> bool:
    return _is_hanoi_state_syntax(name)


def _is_hanoi_operation_name(name: str) -> bool:
    return re.fullmatch(
        r"move_(gray|yellow|green)_hanoi_ring_to_the_center_of_(brown|blue|red)_hanoi_stand",
        name,
    ) is not None


def _is_hanoi_phase_operation_name(name: str) -> bool:
    return re.fullmatch(r"Op_[123][abc]", name) is not None


def _is_river_state_syntax(name: str) -> bool:
    return re.fullmatch(r"State_[01B]{4}", name) is not None


def _is_river_state_name(name: str) -> bool:
    if not _is_river_state_syntax(name):
        return False
    return _is_valid_river_state_code(name[len("State_"):])


def _is_river_operation_name(name: str) -> bool:
    return re.fullmatch(
        r"move_(grass|sheep|steve|wolf)_from_(red_land|boat|green_land)_to_(red_land|boat|green_land)",
        name,
    ) is not None


def _is_river_phase_operation_name(name: str) -> bool:
    return re.fullmatch(r"Op_(grass|sheep|steve|wolf)_[01B]_[01B]", name) is not None


def _is_chess_state_syntax(name: str) -> bool:
    if not name.startswith("State_"):
        return False
    payload = name[len("State_"):]
    if len(payload) != 4:
        return False
    star_code, circle_code = payload[:2], payload[2:]
    if star_code[0] not in VALID_CHESS_POSITIONS or circle_code[0] not in VALID_CHESS_POSITIONS:
        return False
    if star_code[1] not in {"T", "B"} or circle_code[1] not in {"T", "B"}:
        return False
    return True


def _is_chess_state_name(name: str) -> bool:
    if not _is_chess_state_syntax(name):
        return False
    payload = name[len("State_"):]
    star_code, circle_code = payload[:2], payload[2:]
    star_position, circle_position = star_code[0], circle_code[0]
    star_top = star_code[1] == "T"
    circle_top = circle_code[1] == "T"
    if star_position != circle_position:
        return not star_top and not circle_top
    return (star_top and not circle_top) or (circle_top and not star_top)


def _is_chess_operation_name(name: str) -> bool:
    return re.fullmatch(
        r"move-(star|circle)-chess-to-block-(0|1|lowercase-(a|c|d)|uppercase-(A|B|C|D))",
        name,
    ) is not None


def _is_chess_phase_operation_name(name: str) -> bool:
    return re.fullmatch(r"Op_(star|circle)_(0|1|a|c|d|A|B|C|D)", name) is not None


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    phase_state_name_is_valid: Callable[[str], bool]
    phase_operation_name_is_valid: Callable[[str], bool]
    state_name_is_valid: Callable[[str], bool]
    operation_name_is_valid: Callable[[str], bool]
    num_facts: int
    state_count: int
    state_names: Callable[[], Tuple[str, ...]]

    @property
    def prompt_dir(self) -> str:
        return os.path.join(GENERATION_DIR, "prompts", self.name)


SCENARIOS: Dict[str, ScenarioSpec] = {
    "hanoi": ScenarioSpec(
        name="hanoi",
        phase_state_name_is_valid=_is_hanoi_state_syntax,
        phase_operation_name_is_valid=_is_hanoi_phase_operation_name,
        state_name_is_valid=_is_hanoi_state_name,
        operation_name_is_valid=_is_hanoi_operation_name,
        num_facts=3,
        state_count=27,
        state_names=_hanoi_state_names,
    ),
    "river_crossing": ScenarioSpec(
        name="river_crossing",
        phase_state_name_is_valid=_is_river_state_syntax,
        phase_operation_name_is_valid=_is_river_phase_operation_name,
        state_name_is_valid=_is_river_state_name,
        operation_name_is_valid=_is_river_operation_name,
        num_facts=4,
        state_count=40,
        state_names=_river_state_names,
    ),
    "chess": ScenarioSpec(
        name="chess",
        phase_state_name_is_valid=_is_chess_state_syntax,
        phase_operation_name_is_valid=_is_chess_phase_operation_name,
        state_name_is_valid=_is_chess_state_name,
        operation_name_is_valid=_is_chess_operation_name,
        num_facts=2,
        state_count=90,
        state_names=_chess_state_names,
    ),
}


def get_scenario_spec(name: str) -> ScenarioSpec:
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        raise ValueError("Unsupported scenario: {}".format(name)) from exc


def supported_scenarios() -> tuple[str, ...]:
    return tuple(SCENARIOS)

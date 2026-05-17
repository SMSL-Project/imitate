from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy as np

from smsl_gensim.scenarios import state_names
from smsl_gensim.task_registry import ROOT_DIR


Position = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]
Pose = Tuple[Position, Quaternion]

ASSETS_DIR = ROOT_DIR / "cliport" / "environments" / "assets"
IDENTITY: Quaternion = (0.0, 0.0, 0.0, 1.0)
HANOI_ROT: Quaternion = (0.0, 0.0, sin(1.57 / 2), cos(1.57 / 2))


@dataclass(frozen=True)
class SceneObject:
    urdf: str
    pose: Pose
    category: str
    color: Optional[str] = None
    texture: Optional[str] = None


@dataclass(frozen=True)
class SceneSpec:
    scenario: str
    state: str
    objects: tuple[SceneObject, ...]

    @property
    def fixed_count(self) -> int:
        return sum(obj.category == "fixed" for obj in self.objects)

    @property
    def rigid_count(self) -> int:
        return sum(obj.category == "rigid" for obj in self.objects)


@dataclass(frozen=True)
class SceneCheck:
    scenario: str
    state: str
    fixed_count: int
    rigid_count: int
    missing_assets: tuple[Path, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_assets


def build_scene(scenario: str, state: str) -> SceneSpec:
    if state not in state_names(scenario):
        raise ValueError(f"{state} is not a state in {scenario}")
    return _BUILDERS[scenario](state)


def check_scene(scenario: str, state: str) -> SceneCheck:
    spec = build_scene(scenario, state)
    return SceneCheck(
        scenario=scenario,
        state=state,
        fixed_count=spec.fixed_count,
        rigid_count=spec.rigid_count,
        missing_assets=tuple(path for path in _asset_paths(spec) if not path.exists()),
    )


def reset_scene(env, scenario: str, state: str):
    spec = _randomized_scene(build_scene(scenario, state))
    env.smsl_reset()
    env.smsl_scene_state = state
    env.asset_ids_dict = {"fixed": {}, "rigid": {}, "deformable": {}}
    env.asset_poses_dict = {state: {}}
    for obj in spec.objects:
        obj_id = env.add_object(obj.urdf, obj.pose, obj.category, obj.color)
        if obj.texture:
            _apply_texture(obj_id, obj.texture)
        env.asset_ids_dict[obj.category][obj.urdf] = obj_id
        env.asset_poses_dict[state][obj.urdf] = obj.pose
    return env


def _randomized_scene(spec: SceneSpec) -> SceneSpec:
    if spec.scenario == "chess":
        return _randomized_chess_scene(spec)
    if spec.scenario == "river_crossing":
        return _randomized_river_scene(spec)
    return spec


def _randomized_chess_scene(spec: SceneSpec) -> SceneSpec:
    payload = spec.state.replace("State_", "", 1)
    slot_by_urdf = {
        "chess/star_chess.urdf": payload[0],
        "chess/circle_chess.urdf": payload[2],
    }
    jitter_by_slot = {slot: _xy_jitter(CHESS_START_JITTER) for slot in set(slot_by_urdf.values())}
    return SceneSpec(
        spec.scenario,
        spec.state,
        tuple(
            _moved(obj, jitter_by_slot[slot_by_urdf[obj.urdf]])
            if obj.urdf in slot_by_urdf else obj
            for obj in spec.objects
        ),
    )


def _randomized_river_scene(spec: SceneSpec) -> SceneSpec:
    payload = spec.state.replace("State_", "", 1)
    location_by_urdf = {
        f"minecraft/{item}.urdf": location
        for item, location in zip(RIVER_ITEMS, payload)
    }
    slot_by_urdf = _river_permuted_slots(location_by_urdf)
    return SceneSpec(
        spec.scenario,
        spec.state,
        tuple(
            _river_moved(obj, location_by_urdf[obj.urdf], slot_by_urdf[obj.urdf])
            if obj.urdf in location_by_urdf else obj
            for obj in spec.objects
        ),
    )


def _river_permuted_slots(location_by_urdf: dict[str, str]) -> dict[str, Position]:
    slot_by_urdf = {}
    for location in RIVER_BASES:
        urdfs = [f"minecraft/{item}.urdf" for item in RIVER_ITEMS if location_by_urdf[f"minecraft/{item}.urdf"] == location]
        slots = np.random.permutation(len(RIVER_START_SLOTS))[:len(urdfs)]
        for urdf, slot_index in zip(urdfs, slots):
            slot_by_urdf[urdf] = RIVER_START_SLOTS[int(slot_index)]
    return slot_by_urdf


def _chess_scene(state: str) -> SceneSpec:
    payload = state.replace("State_", "", 1)
    board_pose = _pose(0.5, 0.0, 0.0)
    star_pos, star_layer = payload[0], payload[1]
    circle_pos, circle_layer = payload[2], payload[3]
    objects = (
        SceneObject("chess/board.urdf", board_pose, "fixed", texture="chess/board.png"),
        SceneObject(
            "chess/star_chess.urdf",
            _slot_pose(board_pose, CHESS_SLOTS[star_pos], star_layer),
            "rigid",
            "yellow",
        ),
        SceneObject(
            "chess/circle_chess.urdf",
            _slot_pose(board_pose, CHESS_SLOTS[circle_pos], circle_layer),
            "rigid",
            "red",
        ),
    )
    return SceneSpec("chess", state, objects)


def _hanoi_scene(state: str) -> SceneSpec:
    payload = state.replace("State_", "", 1)
    stand_by_disk = dict(zip(("gray", "yellow", "green"), payload))
    objects = [
        SceneObject(f"hanoi/stand_{color}.urdf", pose, "fixed")
        for code, color, pose in HANOI_STANDS
    ]
    for code, _, stand_pose in HANOI_STANDS:
        z = 0.02
        for disk in ("green", "yellow", "gray"):
            if stand_by_disk[disk] == code:
                objects.append(
                    SceneObject(
                        f"hanoi/disk_{disk}.urdf",
                        _offset(stand_pose, (0.0, 0.0, z)),
                        "rigid",
                    )
                )
                z += 0.02
    return SceneSpec("hanoi", state, tuple(objects))


def _river_scene(state: str) -> SceneSpec:
    payload = state.replace("State_", "", 1)
    location_by_item = dict(zip(RIVER_ITEMS, payload))
    objects = [
        SceneObject(
            "minecraft/sea.urdf",
            _pose(0.5, 0.0, 0.001, _yaw(1.57)),
            "fixed",
        ),
        SceneObject("minecraft/red_land.urdf", RIVER_BASES["0"], "fixed", "red"),
        SceneObject("minecraft/green_land.urdf", RIVER_BASES["1"], "fixed", "green"),
        SceneObject("minecraft/boat.urdf", RIVER_BASES["B"], "fixed", "brown", "minecraft/boat.png"),
    ]
    for item in RIVER_ITEMS:
        objects.append(
            SceneObject(
                f"minecraft/{item}.urdf",
                _offset(RIVER_BASES[location_by_item[item]], RIVER_SLOTS[item]),
                "rigid",
                texture=f"minecraft/{item}.png",
            )
        )
    return SceneSpec("river_crossing", state, tuple(objects))


def _asset_paths(spec: SceneSpec) -> tuple[Path, ...]:
    paths = []
    for obj in spec.objects:
        paths.append(ASSETS_DIR / obj.urdf)
        if obj.texture:
            paths.append(ASSETS_DIR / obj.texture)
    return tuple(paths)


def _apply_texture(obj_id: int, texture: str) -> None:
    import pybullet as p

    texture_id = p.loadTexture(str(ASSETS_DIR / texture))
    p.changeVisualShape(obj_id, -1, textureUniqueId=texture_id)


def _pose(x: float, y: float, z: float, rot: Quaternion = IDENTITY) -> Pose:
    return (x, y, z), rot


def _offset(pose: Pose, offset: Position) -> Pose:
    pos, rot = pose
    return (pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2]), rot


def _moved(obj: SceneObject, offset: tuple[float, float]) -> SceneObject:
    pose = _offset(obj.pose, (offset[0], offset[1], 0.0))
    return SceneObject(obj.urdf, pose, obj.category, obj.color, obj.texture)


def _river_moved(obj: SceneObject, location: str, slot: Position) -> SceneObject:
    jitter = _xy_jitter(RIVER_START_JITTER[location])
    pose = _offset(RIVER_BASES[location], (slot[0] + jitter[0], slot[1] + jitter[1], slot[2]))
    return SceneObject(obj.urdf, pose, obj.category, obj.color, obj.texture)


def _xy_jitter(bounds: tuple[float, float]) -> tuple[float, float]:
    return (
        float(np.random.uniform(-bounds[0], bounds[0])),
        float(np.random.uniform(-bounds[1], bounds[1])),
    )


def _slot_pose(base: Pose, offset: Position, layer: str) -> Pose:
    z = offset[2] + (CHESS_LAYER_Z if layer == "T" else 0.0)
    return _offset(base, (offset[0], offset[1], z))


def _yaw(radians: float) -> Quaternion:
    return 0.0, 0.0, sin(radians / 2), cos(radians / 2)


CHESS_BOTTOM_Z = 0.04
CHESS_LAYER_Z = 0.02
CHESS_START_JITTER = (0.015, 0.015)
CHESS_SLOTS: dict[str, Position] = {
    "0": (-0.15, -0.10, CHESS_BOTTOM_Z),
    "1": (0.15, -0.10, CHESS_BOTTOM_Z),
    "a": (-0.15, 0.00, CHESS_BOTTOM_Z),
    "c": (0.05, 0.00, CHESS_BOTTOM_Z),
    "d": (0.15, 0.00, CHESS_BOTTOM_Z),
    "A": (-0.15, 0.10, CHESS_BOTTOM_Z),
    "B": (-0.05, 0.10, CHESS_BOTTOM_Z),
    "C": (0.05, 0.10, CHESS_BOTTOM_Z),
    "D": (0.15, 0.10, CHESS_BOTTOM_Z),
}

HANOI_STANDS: tuple[tuple[str, str, Pose], ...] = (
    ("a", "brown", _pose(0.5, -0.2, 0.0, HANOI_ROT)),
    ("c", "red", _pose(0.5, 0.0, 0.0, HANOI_ROT)),
    ("b", "blue", _pose(0.5, 0.2, 0.0, HANOI_ROT)),
)

RIVER_ITEMS = ("grass", "sheep", "steve", "wolf")
RIVER_ITEM_Z = 0.04
RIVER_BASES: dict[str, Pose] = {
    "0": _pose(0.35, -0.22, 0.0),
    "B": _pose(0.50, 0.00, 0.0),
    "1": _pose(0.65, 0.22, 0.0),
}
RIVER_SLOTS: dict[str, Position] = {
    "grass": (0.02, 0.02, RIVER_ITEM_Z),
    "sheep": (-0.02, 0.02, RIVER_ITEM_Z),
    "steve": (-0.02, -0.02, RIVER_ITEM_Z),
    "wolf": (0.02, -0.02, RIVER_ITEM_Z),
}
RIVER_START_SLOTS = tuple(RIVER_SLOTS.values())
RIVER_START_JITTER = {
    "0": (0.01, 0.025),
    "B": (0.006, 0.006),
    "1": (0.01, 0.025),
}

_BUILDERS: dict[str, Callable[[str], SceneSpec]] = {
    "chess": _chess_scene,
    "hanoi": _hanoi_scene,
    "river_crossing": _river_scene,
}

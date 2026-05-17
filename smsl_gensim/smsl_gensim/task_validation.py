from __future__ import annotations

import ast
from typing import Optional


def validate_task_code(code: str, class_name: str, task_name: str) -> None:
    tree = ast.parse(code)
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    if classes != [class_name]:
        raise ValueError(f"{task_name} must define exactly class {class_name}")
    if "env.add_object" in code:
        raise ValueError(f"{task_name} must use the constrained SMSL scene, not spawn objects")
    _validate_utils_import(tree, task_name)
    _validate_utils_apply(tree, task_name)
    _validate_asset_ids(tree, task_name)
    _validate_add_goal_objects(tree, task_name)
    _validate_reward_metric(tree, task_name)
    _validate_region_sampling(tree, task_name)
    _validate_region_zone_params(tree, task_name)
    _validate_region_target_pose(tree, task_name)
    _validate_region_zone_size(tree, task_name)
    _validate_chess_target_offset(tree, task_name)
    _validate_chess_target_jitter(tree, task_name)
    _validate_river_target_jitter(tree, task_name)
    _validate_river_permutable_slots(tree, task_name)
    compile(code, f"<{task_name}>", "exec")


def _validate_utils_import(tree: ast.AST, task_name: str) -> None:
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "cliport.utils" and alias.asname == "utils":
                    raise ValueError(f"{task_name} must import utils with: from cliport.utils import utils")
        if isinstance(node, ast.ImportFrom) and node.module == "cliport":
            if any(alias.name == "utils" for alias in node.names):
                raise ValueError(f"{task_name} must import utils with: from cliport.utils import utils")


def _validate_utils_apply(tree: ast.AST, task_name: str) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_utils_apply_call(node) and len(node.args) != 2:
            raise ValueError(f"{task_name} utils.apply must receive exactly base_pose and (x, y, z)")


def _validate_asset_ids(tree: ast.AST, task_name: str) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and _is_asset_ids_dict(node.value):
            key = _subscript_key(node)
            if key not in ("fixed", "rigid", "deformable"):
                raise ValueError(f"{task_name} must access env.asset_ids_dict by category before URDF path")


def _validate_add_goal_objects(tree: ast.AST, task_name: str) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_add_goal_call(node):
            objs = next((keyword.value for keyword in node.keywords if keyword.arg == "objs"), None)
            if isinstance(objs, (ast.List, ast.Tuple)):
                for item in objs.elts:
                    if isinstance(item, ast.Name) and "pose" in item.id.lower():
                        raise ValueError(f"{task_name} add_goal objs must contain object IDs, not poses")


def _validate_reward_metric(tree: ast.AST, task_name: str) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_add_goal_call(node):
            metric = _keyword_value(node, "metric")
            params = _keyword_value(node, "params")
            rotations = _keyword_value(node, "rotations")
            if _keyword_value(node, "targ_poses") is None:
                raise ValueError(f"{task_name} add_goal must pass targ_poses")
            if _uses_region_reward(task_name):
                if _resolved_constant(tree, metric) != "zone":
                    raise ValueError(f"{task_name} must use zone metric")
                if params is None or (isinstance(params, ast.Constant) and params.value is None):
                    raise ValueError(f"{task_name} zone metric must define params")
            elif "-hanoi-" in task_name and _resolved_constant(tree, metric) != "pose":
                raise ValueError(f"{task_name} must use pose metric")
            if _resolved_constant(tree, rotations) is not False:
                raise ValueError(f"{task_name} must use rotations=False")


def _validate_region_sampling(tree: ast.AST, task_name: str) -> None:
    if not _uses_region_reward(task_name):
        return
    if not any(_is_np_random_uniform_call(node) for node in ast.walk(tree) if isinstance(node, ast.Call)):
        raise ValueError(f"{task_name} must sample target pose inside the zone with np.random.uniform")


def _validate_region_zone_params(tree: ast.AST, task_name: str) -> None:
    if not _uses_region_reward(task_name):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "zone_pose" and _is_utils_apply_call(node.value):
                    raise ValueError(
                        f"{task_name} zone_pose must be a pose tuple: "
                        "zone_position = utils.apply(...); zone_pose = (zone_position, base_pose[1])"
                    )
                if isinstance(target, ast.Name) and target.id == "target_position" and _is_np_random_uniform_call(node.value):
                    raise ValueError(f"{task_name} target_position must come from utils.apply")
        if isinstance(node, ast.Call) and _is_add_goal_call(node):
            params = _keyword_value(node, "params")
            if _contains_name(params, "target_pose"):
                raise ValueError(f"{task_name} zone metric params must use zone_pose, not target_pose")


def _validate_region_target_pose(tree: ast.AST, task_name: str) -> None:
    if not _uses_region_reward(task_name):
        return
    value = _assigned_value(tree, "target_pose")
    if not isinstance(value, ast.Tuple) or len(value.elts) != 2:
        raise ValueError(f"{task_name} target_pose must be a pose tuple")


def _validate_region_zone_size(tree: ast.AST, task_name: str) -> None:
    expected = _expected_zone_size(task_name)
    if expected is None:
        return
    value = _literal_number_tuple(_assigned_value(tree, "zone_size"))
    if value != expected:
        raise ValueError(f"{task_name} zone_size must be {expected}")


def _validate_river_target_jitter(tree: ast.AST, task_name: str) -> None:
    maximum = _river_target_jitter_bounds(task_name)
    if maximum is None:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and _is_np_random_uniform_call(node.value):
            bounds = _uniform_bounds(node.value)
            if bounds:
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "dx" and _max_abs(bounds) > maximum[0]:
                        raise ValueError(f"{task_name} river target jitter dx must be <= {maximum[0]}")
                    if isinstance(target, ast.Name) and target.id == "dy" and _max_abs(bounds) > maximum[1]:
                        raise ValueError(f"{task_name} river target jitter dy must be <= {maximum[1]}")


def _validate_river_permutable_slots(tree: ast.AST, task_name: str) -> None:
    if _river_target_jitter_bounds(task_name) is None:
        return
    if not any(_is_np_random_call(node, "randint") for node in ast.walk(tree) if isinstance(node, ast.Call)):
        raise ValueError(f"{task_name} river target slot must be sampled from permutable slots")


def _validate_chess_target_offset(tree: ast.AST, task_name: str) -> None:
    expected = _chess_target_offset(task_name)
    if expected is None:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_utils_apply_call(node):
            if len(node.args) >= 2 and _literal_number_tuple(node.args[1]) == expected:
                return
    raise ValueError(f"{task_name} chess target offset must be {expected}")


def _validate_chess_target_jitter(tree: ast.AST, task_name: str) -> None:
    if _chess_target_offset(task_name) is None:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and _is_np_random_uniform_call(node.value):
            bounds = _uniform_bounds(node.value)
            if bounds:
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ("dx", "dy") and _max_abs(bounds) > 0.02:
                        raise ValueError(f"{task_name} chess target jitter must be <= 0.02")


def _is_asset_ids_dict(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "asset_ids_dict"
        and isinstance(node.value, ast.Name)
        and node.value.id == "env"
    )


def _is_add_goal_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Attribute) and node.func.attr == "add_goal"


def _is_utils_apply_call(node: ast.Call) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "apply"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "utils"
    )


def _is_np_random_uniform_call(node: ast.Call) -> bool:
    return _is_np_random_call(node, "uniform")


def _is_np_random_call(node: ast.Call, name: str) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == name
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "random"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "np"
    )


def _subscript_key(node: ast.Subscript) -> Optional[str]:
    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    if hasattr(ast, "Index") and isinstance(node.slice, ast.Index):
        value = node.slice.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
        if isinstance(value, ast.Str):
            return value.s
    if isinstance(node.slice, ast.Str):
        return node.slice.s
    return None


def _keyword_value(node: ast.Call, name: str) -> Optional[ast.AST]:
    return next((keyword.value for keyword in node.keywords if keyword.arg == name), None)


def _constant_value(node: Optional[ast.AST]):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Str):
        return node.s
    if isinstance(node, ast.NameConstant):
        return node.value
    return None


def _resolved_constant(tree: ast.AST, node: Optional[ast.AST]):
    value = _constant_value(node)
    if value is not None:
        return value
    if isinstance(node, ast.Name):
        return _constant_value(_assigned_value(tree, node.id))
    return None


def _contains_name(node: Optional[ast.AST], name: str) -> bool:
    return node is not None and any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))


def _assigned_value(tree: ast.AST, name: str) -> Optional[ast.AST]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return node.value
    return None


def _literal_number_tuple(node: Optional[ast.AST]) -> Optional[tuple[float, ...]]:
    if not isinstance(node, ast.Tuple):
        return None
    values = tuple(_number_value(item) for item in node.elts)
    if any(not isinstance(value, (int, float)) for value in values):
        return None
    return tuple(round(float(value), 3) for value in values)


def _expected_zone_size(task_name: str) -> Optional[tuple[float, float, float]]:
    if "-chess-" in task_name:
        return (0.15, 0.15, 0.08)
    if "-to-boat" in task_name:
        return (0.18, 0.2, 0.1)
    if "-to-red-land" in task_name or "-to-green-land" in task_name:
        return (0.26, 0.28, 0.1)
    return None


def _river_target_jitter_bounds(task_name: str) -> Optional[tuple[float, float]]:
    if "-to-boat" in task_name:
        return 0.015, 0.010
    if "-to-red-land" in task_name or "-to-green-land" in task_name:
        return 0.035, 0.050
    return None


def _chess_target_offset(task_name: str) -> Optional[tuple[float, float, float]]:
    marker = "-chess-to-block-"
    if marker not in task_name:
        return None
    label = task_name.split(marker, 1)[1]
    if label.startswith("uppercase-"):
        label = label[len("uppercase-"):]
    elif label.startswith("lowercase-"):
        label = label[len("lowercase-"):]
    return CHESS_TARGET_OFFSETS.get(label)


def _uses_region_reward(task_name: str) -> bool:
    return "-chess-" in task_name or ("-from-" in task_name and "-to-" in task_name)


def _uniform_bounds(node: ast.Call) -> Optional[tuple[float, float]]:
    if len(node.args) >= 2:
        low = _number_value(node.args[0])
        high = _number_value(node.args[1])
    else:
        low = _number_value(_keyword_value(node, "low"))
        high = _number_value(_keyword_value(node, "high"))
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        return float(low), float(high)
    return None


def _max_abs(values: tuple[float, float]) -> float:
    return max(abs(values[0]), abs(values[1]))


def _number_value(node: Optional[ast.AST]):
    value = _constant_value(node)
    if isinstance(value, (int, float)):
        return value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(_constant_value(node.operand), (int, float))
    ):
        return -_constant_value(node.operand)
    return None


CHESS_TARGET_OFFSETS = {
    "0": (-0.15, -0.1, 0.04),
    "1": (0.15, -0.1, 0.04),
    "a": (-0.15, 0.0, 0.04),
    "c": (0.05, 0.0, 0.04),
    "d": (0.15, 0.0, 0.04),
    "A": (-0.15, 0.1, 0.04),
    "B": (-0.05, 0.1, 0.04),
    "C": (0.05, 0.1, 0.04),
    "D": (0.15, 0.1, 0.04),
}

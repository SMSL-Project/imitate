from __future__ import annotations

import re

from smsl_gensim.scenes import CHESS_SLOTS


def render_task(task_name: str, class_name: str) -> str:
    for pattern, renderer in (
        (_HANOI, _render_hanoi),
        (_CHESS, _render_chess),
        (_RIVER, _render_river),
    ):
        match = pattern.fullmatch(task_name)
        if match:
            return renderer(class_name, **match.groupdict())
    raise ValueError(f"Unsupported SMSL operation: {task_name}")


def _render_hanoi(class_name: str, ring: str, stand: str) -> str:
    return f'''import numpy as np
from cliport.tasks.task import Task


class {class_name}(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move the {ring} hanoi ring to the center of the {stand} hanoi stand"
        self.task_completed_desc = "done moving the {ring} hanoi ring."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        ring_id = env.asset_ids_dict["rigid"]["hanoi/disk_{ring}.urdf"]
        target_pose = env.asset_poses_dict[env.smsl_scene_state]["hanoi/stand_{stand}.urdf"]

        self.add_goal(
            objs=[ring_id],
            matches=np.ones((1, 1)),
            targ_poses=[target_pose],
            replace=False,
            rotations=False,
            metric="pose",
            params=None,
            step_max_reward=1,
            language_goal=self.lang_template,
        )
'''


def _render_chess(class_name: str, piece: str, label: str) -> str:
    block = _chess_block(label)
    x, y, z = CHESS_SLOTS[block]
    phrase = _chess_phrase(label, block)
    sx, sy, sz = CHESS_ZONE_SIZE
    jx, jy = CHESS_JITTER
    return f'''import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class {class_name}(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "pick up the {piece} chess piece and place it on the {phrase}"
        self.task_completed_desc = "done moving the {piece} chess piece to the {phrase}."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        board_id = env.asset_ids_dict["fixed"]["chess/board.urdf"]
        board_pose = env.get_object_pose(board_id)
        zone_position = utils.apply(board_pose, ({_fmt3(x)}, {_fmt3(y)}, {_fmt3(z)}))
        zone_pose = (zone_position, board_pose[1])
        dx = np.random.uniform(-{_fmt3(jx)}, {_fmt3(jx)})
        dy = np.random.uniform(-{_fmt3(jy)}, {_fmt3(jy)})
        target_position = utils.apply(zone_pose, (dx, dy, 0.0))
        target_pose = (target_position, zone_pose[1])
        zone_size = ({_fmt3(sx)}, {_fmt3(sy)}, {_fmt3(sz)})

        self.add_goal(
            objs=[env.asset_ids_dict["rigid"]["chess/{piece}_chess.urdf"]],
            matches=np.ones((1, 1)),
            targ_poses=[target_pose],
            replace=False,
            rotations=False,
            metric="zone",
            params=[(zone_pose, zone_size)],
            step_max_reward=1,
            language_goal=self.lang_template,
        )
'''


def _render_river(class_name: str, item: str, source: str, target: str) -> str:
    target_urdf = _RIVER_TARGETS[target]
    sx, sy, sz = _RIVER_ZONE_SIZES[target]
    jx, jy = _RIVER_JITTER[target]
    item_text = "Steve" if item == "steve" else f"the {item}"
    return f'''import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class {class_name}(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move {item_text} from the {source.replace("-", " ")} to the {target.replace("-", " ")}"
        self.task_completed_desc = "done moving {item_text}."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        target_id = env.asset_ids_dict["fixed"]["{target_urdf}"]
        zone_pose = env.get_object_pose(target_id)
        slots = ((0.02, 0.02), (-0.02, 0.02), (-0.02, -0.02), (0.02, -0.02))
        slot_x, slot_y = slots[np.random.randint(len(slots))]
        dx = np.random.uniform(-{_fmt3(jx)}, {_fmt3(jx)})
        dy = np.random.uniform(-{_fmt3(jy)}, {_fmt3(jy)})
        target_position = utils.apply(zone_pose, (slot_x + dx, slot_y + dy, 0.04))
        target_pose = (target_position, zone_pose[1])
        zone_size = ({_fmt2(sx)}, {_fmt2(sy)}, {_fmt2(sz)})

        self.add_goal(
            objs=[env.asset_ids_dict["rigid"]["minecraft/{item}.urdf"]],
            matches=np.ones((1, 1)),
            targ_poses=[target_pose],
            replace=False,
            rotations=False,
            metric="zone",
            params=[(zone_pose, zone_size)],
            step_max_reward=1,
            language_goal=self.lang_template,
        )
'''


def _chess_block(label: str) -> str:
    if label.startswith("uppercase-"):
        return label[len("uppercase-"):]
    if label.startswith("lowercase-"):
        return label[len("lowercase-"):]
    return label


def _chess_phrase(label: str, block: str) -> str:
    if label.startswith("uppercase-"):
        return f"square labeled '{block}'"
    if label.startswith("lowercase-"):
        return f"lowercase '{block}' block"
    return f"block labeled '{block}'"


def _fmt2(value: float) -> str:
    return f"{value:.2f}"


def _fmt3(value: float) -> str:
    return f"{value:.3f}"


_HANOI = re.compile(
    r"move-(?P<ring>gray|yellow|green)-hanoi-ring-to-the-center-of-"
    r"(?P<stand>blue|brown|red)-hanoi-stand"
)
_CHESS = re.compile(r"move-(?P<piece>circle|star)-chess-to-block-(?P<label>.+)")
_RIVER = re.compile(
    r"move-(?P<item>grass|sheep|steve|wolf)-from-"
    r"(?P<source>boat|red-land|green-land)-to-"
    r"(?P<target>boat|red-land|green-land)"
)
_RIVER_TARGETS = {
    "boat": "minecraft/boat.urdf",
    "red-land": "minecraft/red_land.urdf",
    "green-land": "minecraft/green_land.urdf",
}
CHESS_ZONE_SIZE = (0.15, 0.15, 0.08)
RIVER_BOAT_ZONE_SIZE = (0.18, 0.20, 0.10)
RIVER_LAND_ZONE_SIZE = (0.26, 0.28, 0.10)
_RIVER_ZONE_SIZES = {
    "boat": RIVER_BOAT_ZONE_SIZE,
    "red-land": RIVER_LAND_ZONE_SIZE,
    "green-land": RIVER_LAND_ZONE_SIZE,
}
CHESS_JITTER = (0.02, 0.02)
RIVER_BOAT_JITTER = (0.015, 0.010)
RIVER_LAND_JITTER = (0.035, 0.050)
_RIVER_JITTER = {
    "boat": RIVER_BOAT_JITTER,
    "red-land": RIVER_LAND_JITTER,
    "green-land": RIVER_LAND_JITTER,
}

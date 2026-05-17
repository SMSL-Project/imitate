import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class MoveCircleChessToBlockLowercaseD(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "Move the circle chess piece to the board slot labeled with the lowercase letter 'd'."
        self.task_completed_desc = "done moving circle chess piece to block lowercase d."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        obj_id = env.asset_ids_dict["rigid"]["chess/circle_chess.urdf"]
        fixed_id = env.asset_ids_dict["fixed"]["chess/board.urdf"]
        base_pose = env.get_object_pose(fixed_id)
        
        zone_position = utils.apply(base_pose, (0.150, 0.000, 0.040))
        zone_pose = (zone_position, base_pose[1])
        dx = np.random.uniform(-0.020, 0.020)
        dy = np.random.uniform(-0.020, 0.020)
        target_position = utils.apply(zone_pose, (dx, dy, 0.0))
        target_pose = (target_position, zone_pose[1])
        zone_size = (0.150, 0.150, 0.080)

        self.add_goal(
            objs=[obj_id],
            matches=np.ones((1, 1)),
            targ_poses=[target_pose],
            replace=False,
            rotations=False,
            metric="zone",
            params=[(zone_pose, zone_size)],
            step_max_reward=1,
            language_goal=self.lang_template,
        )

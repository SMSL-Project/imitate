import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class MoveCircleChessToBlock1(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "Move the circle chess piece to the designated block 1 position on the chessboard."
        self.task_completed_desc = "done moving circle chess piece to block 1."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        obj_id = env.asset_ids_dict["rigid"]["chess/circle_chess.urdf"]
        fixed_id = env.asset_ids_dict["fixed"]["chess/board.urdf"]
        base_pose = env.get_object_pose(fixed_id)
        
        # Define the zone position and pose for block 1
        zone_position = utils.apply(base_pose, (0.150, -0.100, 0.040))
        zone_pose = (zone_position, base_pose[1])
        
        # Sample target position within the zone
        dx = np.random.uniform(-0.020, 0.020)
        dy = np.random.uniform(-0.020, 0.020)
        target_position = utils.apply(zone_pose, (dx, dy, 0.0))
        target_pose = (target_position, zone_pose[1])
        
        # Define the zone size
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

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class MoveStarChessToBlock0(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move the yellow star chess piece to the block labeled '0' on the chessboard"
        self.task_completed_desc = "done moving the yellow star chess piece to block '0'."
        self.additional_reset()

    def reset(self, env):
        board_id = env.asset_ids_dict['fixed']["chess/board.urdf"]
        board_pose = env.get_object_pose(board_id)

        trans = (-0.150, -0.100, 0.040)
        target_pose = utils.apply(board_pose, trans)

        self.add_goal(
            objs=[env.asset_ids_dict['rigid'].get("chess/star_chess.urdf")],
            matches=np.ones((1, 1)),
            targ_poses=[target_pose],
            replace=False,
            rotations=True,
            metric='pose',
            params=None,
            step_max_reward=1,
            language_goal=self.lang_template
        )

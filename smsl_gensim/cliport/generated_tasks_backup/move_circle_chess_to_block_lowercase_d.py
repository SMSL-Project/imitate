import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class MoveCircleChessToBlockLowercaseD(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "pick up the circle chess piece and place it on the lowercase 'd' block"
        self.task_completed_desc = "done moving the circle chess piece to the lowercase 'd' block."
        self.additional_reset()

    def reset(self, env):
        chessboard_id = env.asset_ids_dict['fixed']["chess/board.urdf"]
        chessboard_pose = env.get_object_pose(chessboard_id)

        trans = (0.150, 0.000, 0.040)
        target_pose = utils.apply(chessboard_pose, trans)

        self.add_goal(
            objs=[env.asset_ids_dict['rigid'].get("chess/circle_chess.urdf")],
            matches=np.ones((1, 1)),
            targ_poses=[target_pose],
            replace=False,
            rotations=True,
            metric='pose',
            params=None,
            step_max_reward=1,
            language_goal=self.lang_template
        )

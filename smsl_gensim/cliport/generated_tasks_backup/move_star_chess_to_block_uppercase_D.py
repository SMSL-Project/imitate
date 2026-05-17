import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class MoveStarChessToBlockUppercaseD(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "pick up the star chess piece and place it on the square labeled 'D' on the chessboard"
        self.task_completed_desc = "done moving the star chess piece to the square labeled 'D'."
        self.additional_reset()

    def reset(self, env):
        chessboard_id = env.asset_ids_dict['fixed']["chess/board.urdf"]
        chessboard_pose = env.get_object_pose(chessboard_id)

        trans = (0.150, 0.100, 0.040)
        target_pose = utils.apply(chessboard_pose, trans)

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

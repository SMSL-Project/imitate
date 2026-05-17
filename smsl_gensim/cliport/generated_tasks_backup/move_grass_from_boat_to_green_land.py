import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class MoveGrassFromBoatToGreenLand(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "pick up the grass from the boat and place it on the green land"
        self.task_completed_desc = "done moving the grass from the boat to the green land."
        self.additional_reset()

    def reset(self, env):
        green_land_id = env.asset_ids_dict['fixed']["minecraft/green_land.urdf"]
        green_land_pose = env.get_object_pose(green_land_id)

        trans = (0.02, 0.02, 0.04)
        target_pose = utils.apply(green_land_pose, trans)

        self.add_goal(
            objs=[env.asset_ids_dict['rigid'].get("minecraft/grass.urdf")],
            matches=np.ones((1, 1)),
            targ_poses=[target_pose],
            replace=False,
            rotations=True,
            metric='pose',
            params=None,
            step_max_reward=1,
            language_goal=self.lang_template
        )

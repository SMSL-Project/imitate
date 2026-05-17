import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class MoveSheepFromBoatToRedLand(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "pick up the sheep from the boat and place it on the red land"
        self.task_completed_desc = "done moving the sheep from the boat to the red land."
        self.additional_reset()

    def reset(self, env):
        red_land_id = env.asset_ids_dict['fixed']["minecraft/red_land.urdf"]
        red_land_pose = env.get_object_pose(red_land_id)

        trans = (-0.02, 0.02, 0.04)
        target_pose = utils.apply(red_land_pose, trans)

        self.add_goal(
            objs=[env.asset_ids_dict['rigid'].get("minecraft/sheep.urdf")],
            matches=np.ones((1, 1)),
            targ_poses=[target_pose],
            replace=False,
            rotations=True,
            metric='pose',
            params=None,
            step_max_reward=1,
            language_goal=self.lang_template
        )

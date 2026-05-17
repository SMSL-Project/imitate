import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class MoveSteveFromGreenLandToBoat(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "pick up the Steve figure from the green land and place it onto the boat"
        self.task_completed_desc = "done moving Steve from the green land to the boat."
        self.additional_reset()

    def reset(self, env):
        boat_id = env.asset_ids_dict['fixed']["minecraft/boat.urdf"]
        boat_pose = env.get_object_pose(boat_id)

        trans = (-0.02, -0.02, 0.04)
        target_pose = utils.apply(boat_pose, trans)

        self.add_goal(
            objs=[env.asset_ids_dict['rigid'].get("minecraft/steve.urdf")],
            matches=np.ones((1, 1)),
            targ_poses=[target_pose],
            replace=False,
            rotations=True,
            metric='pose',
            params=None,
            step_max_reward=1,
            language_goal=self.lang_template
        )

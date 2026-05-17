import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class MoveGreenHanoiRingToTheCenterOfRedHanoiStand(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "Move the green hanoi ring to the center of the red hanoi stand."
        self.task_completed_desc = "done moving the green hanoi ring to the center of the red hanoi stand."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        obj_id = env.asset_ids_dict["rigid"]["hanoi/disk_green.urdf"]
        fixed_id = env.asset_ids_dict["fixed"]["hanoi/stand_red.urdf"]
        base_pose = env.get_object_pose(fixed_id)
        target_pose = (base_pose[0], base_pose[1])

        self.add_goal(
            objs=[obj_id],
            matches=np.ones((1, 1)),
            targ_poses=[target_pose],
            replace=False,
            rotations=False,
            metric="pose",
            params=None,
            step_max_reward=1,
            language_goal=self.lang_template,
        )

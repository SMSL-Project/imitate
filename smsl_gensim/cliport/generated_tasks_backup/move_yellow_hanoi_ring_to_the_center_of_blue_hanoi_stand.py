import numpy as np
from cliport.tasks.task import Task


class MoveYellowHanoiRingToCenterOfBlueHanoiStand(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move the yellow hanoi ring to the center of the blue hanoi stand"
        self.task_completed_desc = "done moving the yellow hanoi ring."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        ring_id = env.asset_ids_dict["rigid"]["hanoi/disk_yellow.urdf"]
        target_pose = env.asset_poses_dict[env.smsl_scene_state]["hanoi/stand_blue.urdf"]

        self.add_goal(
            objs=[ring_id],
            matches=np.ones((1, 1)),
            targ_poses=[target_pose],
            replace=False,
            rotations=True,
            metric="pose",
            params=None,
            step_max_reward=1,
            language_goal=self.lang_template,
        )

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class MoveSteveFromBoatToRedLand(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "Move the character Steve from the boat to the red land."
        self.task_completed_desc = "Steve has been successfully moved to the red land."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        obj_id = env.asset_ids_dict["rigid"]["minecraft/steve.urdf"]
        fixed_id = env.asset_ids_dict["fixed"]["minecraft/red_land.urdf"]
        base_pose = env.get_object_pose(fixed_id)

        # Define the zone for the red land
        zone_position = utils.apply(base_pose, (0, 0, 0))
        zone_pose = (zone_position, base_pose[1])
        zone_size = (0.26, 0.28, 0.10)

        # Sample a destination slot
        slot_offsets = [(0.02, 0.02), (-0.02, 0.02), (-0.02, -0.02), (0.02, -0.02)]
        slot_x, slot_y = slot_offsets[np.random.randint(0, 4)]

        # Apply jitter
        dx = np.random.uniform(-0.035, 0.035)
        dy = np.random.uniform(-0.050, 0.050)

        # Calculate the target position
        target_position = utils.apply(zone_pose, (slot_x + dx, slot_y + dy, 0.04))
        target_pose = (target_position, zone_pose[1])

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

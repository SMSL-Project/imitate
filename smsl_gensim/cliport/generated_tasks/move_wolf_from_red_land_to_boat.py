import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class MoveWolfFromRedLandToBoat(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "Move the wolf from the red land to the boat."
        self.task_completed_desc = "The wolf has been moved to the boat."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        wolf_id = env.asset_ids_dict["rigid"]["minecraft/wolf.urdf"]
        boat_id = env.asset_ids_dict["fixed"]["minecraft/boat.urdf"]
        base_pose = env.get_object_pose(boat_id)

        # Define the zone for the boat
        zone_position = utils.apply(base_pose, (0.0, 0.0, 0.0))
        zone_pose = (zone_position, base_pose[1])
        dx = np.random.uniform(-0.015, 0.015)
        dy = np.random.uniform(-0.010, 0.010)
        slot_offsets = [(0.02, 0.02), (-0.02, 0.02), (-0.02, -0.02), (0.02, -0.02)]
        slot_x, slot_y = slot_offsets[np.random.randint(0, len(slot_offsets))]
        target_position = utils.apply(zone_pose, (slot_x + dx, slot_y + dy, 0.04))
        target_pose = (target_position, zone_pose[1])
        zone_size = (0.18, 0.20, 0.10)

        self.add_goal(
            objs=[wolf_id],
            matches=np.ones((1, 1)),
            targ_poses=[target_pose],
            replace=False,
            rotations=False,
            metric="zone",
            params=[(zone_pose, zone_size)],
            step_max_reward=1,
            language_goal=self.lang_template,
        )

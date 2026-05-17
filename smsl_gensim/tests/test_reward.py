import unittest
from unittest.mock import patch

import numpy as np
import pybullet as p

from cliport.tasks.task import Task


class RewardTests(unittest.TestCase):
    def test_zone_reward_uses_xy_only(self):
        task = Task()
        task.goals = [(
            [(1, (False, None))],
            np.ones((1, 1)),
            [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))],
            False,
            False,
            "zone",
            [(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)), (0.2, 0.2, 0.01))],
            1,
        )]
        task.obj_points_cache = {1: np.array([[0.0], [0.0], [10.0]], dtype=np.float32)}
        with patch("pybullet.getBasePositionAndOrientation", return_value=((0.0, 0.0, 10.0), (0.0, 0.0, 0.0, 1.0))):
            reward, _ = task.reward()
        self.assertEqual(reward, 1)

    def test_cylinder_object_points_are_not_empty(self):
        task = Task()
        shape = [(None, None, p.GEOM_CYLINDER, (0.002, 0.025, 0.0))]

        with patch("pybullet.getVisualShapeData", return_value=shape):
            points = task.get_box_object_points(1)

        self.assertGreater(points.shape[1], 0)


if __name__ == "__main__":
    unittest.main()

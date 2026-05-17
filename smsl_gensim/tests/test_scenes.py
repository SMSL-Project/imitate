import unittest
from unittest.mock import patch
import numpy as np

from smsl_gensim import available_scenarios, check_scene, reset_scene, state_names
from smsl_gensim.scenes import RIVER_BASES, RIVER_SLOTS, build_scene


class FakeEnv:
    def __init__(self):
        self.reset_count = 0
        self.added = []

    def smsl_reset(self):
        self.reset_count += 1

    def add_object(self, urdf, pose, category="rigid", color=None):
        self.added.append((urdf, pose, category, color))
        return len(self.added)


class SceneTests(unittest.TestCase):
    def test_all_scenes_have_assets(self):
        for scenario in available_scenarios():
            for state in state_names(scenario):
                check = check_scene(scenario, state)
                self.assertTrue(check.ok, check)

    def test_hanoi_state_places_stacked_rings(self):
        env = FakeEnv()
        reset_scene(env, "hanoi", "State_aaa")
        poses = env.asset_poses_dict["State_aaa"]
        self.assertEqual(env.reset_count, 1)
        self.assertEqual(env.smsl_scene_state, "State_aaa")
        self.assertEqual(len(env.asset_ids_dict["fixed"]), 3)
        self.assertEqual(len(env.asset_ids_dict["rigid"]), 3)
        self.assertLess(poses["hanoi/disk_green.urdf"][0][2], poses["hanoi/disk_yellow.urdf"][0][2])
        self.assertLess(poses["hanoi/disk_yellow.urdf"][0][2], poses["hanoi/disk_gray.urdf"][0][2])

    def test_chess_state_places_top_piece_above_bottom_piece(self):
        scene = build_scene("chess", "State_ATAB")
        objects = {obj.urdf: obj.pose for obj in scene.objects}
        star_pos = objects["chess/star_chess.urdf"][0]
        circle_pos = objects["chess/circle_chess.urdf"][0]
        self.assertEqual(star_pos[:2], circle_pos[:2])
        self.assertGreater(star_pos[2], circle_pos[2])
        self.assertAlmostEqual(circle_pos[2], 0.04)
        self.assertAlmostEqual(star_pos[2], 0.06)

    def test_chess_reset_applies_board_texture(self):
        env = FakeEnv()
        with patch("pybullet.loadTexture", return_value=7) as load_texture, \
                patch("pybullet.changeVisualShape") as change_visual:
            reset_scene(env, "chess", "State_BBAB")
        self.assertTrue(load_texture.call_args.args[0].endswith("chess/board.png"))
        change_visual.assert_called_once_with(1, -1, textureUniqueId=7)

    def test_chess_reset_randomizes_stacked_piece_xy_together(self):
        env = FakeEnv()
        np.random.seed(0)
        with patch("pybullet.loadTexture", return_value=7), patch("pybullet.changeVisualShape"):
            reset_scene(env, "chess", "State_ATAB")
        poses = env.asset_poses_dict["State_ATAB"]
        star = poses["chess/star_chess.urdf"][0]
        circle = poses["chess/circle_chess.urdf"][0]
        self.assertEqual(star[:2], circle[:2])
        self.assertNotEqual(star[:2], (0.35, 0.1))
        self.assertGreater(star[2], circle[2])

    def test_river_reset_applies_minecraft_textures(self):
        env = FakeEnv()
        with patch("pybullet.loadTexture", side_effect=range(10, 15)) as load_texture, \
                patch("pybullet.changeVisualShape") as change_visual:
            reset_scene(env, "river_crossing", "State_B1B0")
        texture_names = [call.args[0].split("minecraft/")[-1] for call in load_texture.call_args_list]
        self.assertEqual(texture_names, ["boat.png", "grass.png", "sheep.png", "steve.png", "wolf.png"])
        self.assertEqual([call.args[0] for call in change_visual.call_args_list], [4, 5, 6, 7, 8])

    def test_river_reset_randomizes_rigid_items_only(self):
        env = FakeEnv()
        np.random.seed(0)
        with patch("pybullet.loadTexture", side_effect=range(10, 15)), patch("pybullet.changeVisualShape"):
            reset_scene(env, "river_crossing", "State_B1B0")
        poses = env.asset_poses_dict["State_B1B0"]
        self.assertEqual(poses["minecraft/boat.urdf"], RIVER_BASES["B"])
        grass = poses["minecraft/grass.urdf"][0]
        expected_grass = (
            RIVER_BASES["B"][0][0] + RIVER_SLOTS["grass"][0],
            RIVER_BASES["B"][0][1] + RIVER_SLOTS["grass"][1],
            RIVER_SLOTS["grass"][2],
        )
        self.assertNotEqual(grass[:2], expected_grass[:2])
        self.assertAlmostEqual(grass[2], expected_grass[2])

    def test_river_reset_permutates_start_slots(self):
        env = FakeEnv()
        np.random.seed(0)
        with patch("smsl_gensim.scenes._xy_jitter", return_value=(0.0, 0.0)), \
                patch("pybullet.loadTexture", side_effect=range(10, 15)), \
                patch("pybullet.changeVisualShape"):
            reset_scene(env, "river_crossing", "State_1111")
        poses = env.asset_poses_dict["State_1111"]
        base = RIVER_BASES["1"][0]
        slots = {
            (
                round(poses[f"minecraft/{item}.urdf"][0][0] - base[0], 2),
                round(poses[f"minecraft/{item}.urdf"][0][1] - base[1], 2),
            )
            for item in ("grass", "sheep", "steve", "wolf")
        }
        self.assertEqual(slots, {(0.02, 0.02), (-0.02, 0.02), (-0.02, -0.02), (0.02, -0.02)})

    def test_river_state_decodes_locations(self):
        scene = build_scene("river_crossing", "State_B1B0")
        objects = {obj.urdf: obj.pose for obj in scene.objects}
        grass_pos = objects["minecraft/grass.urdf"][0]
        sheep_pos = objects["minecraft/sheep.urdf"][0]
        steve_pos = objects["minecraft/steve.urdf"][0]
        wolf_pos = objects["minecraft/wolf.urdf"][0]
        self.assertAlmostEqual(grass_pos[0], steve_pos[0] + 0.04)
        self.assertAlmostEqual(grass_pos[2], 0.04)
        self.assertGreater(sheep_pos[0], grass_pos[0])
        self.assertLess(wolf_pos[0], grass_pos[0])


if __name__ == "__main__":
    unittest.main()

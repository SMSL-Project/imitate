import unittest
from pathlib import Path

from smsl_gensim import available_scenarios
from smsl_gensim.checks import check_scenario_tasks
from smsl_gensim.task_registry import task_class_names


class SMSLTaskTests(unittest.TestCase):
    def test_all_scenarios_pass_task_check(self):
        for name in available_scenarios():
            check = check_scenario_tasks(name)
            self.assertTrue(check.ok, check)

    def test_hanoi_tasks_use_constrained_scene_objects(self):
        for module in task_class_names():
            if "_hanoi_ring_to_the_center_of_" in module:
                source = (Path("cliport/generated_tasks") / f"{module}.py").read_text(encoding="utf-8")
                self.assertIn("env.asset_ids_dict", source)
                self.assertIn("env.asset_poses_dict[env.smsl_scene_state]", source)
                self.assertIn("rotations=False", source)
                self.assertNotIn("env.add_object", source)
                self.assertNotIn("import cliport.utils as utils", source)

    def test_chess_tasks_target_visible_piece_height(self):
        for module in task_class_names():
            if "_chess_to_block_" in module:
                source = (Path("cliport/generated_tasks") / f"{module}.py").read_text(encoding="utf-8")
                self.assertIn("zone_position = utils.apply", source)
                self.assertIn(", 0.040))", source)
                self.assertRegex(source, r"metric=[\"']zone[\"']")
                self.assertIn("np.random.uniform", source)
                self.assertIn("zone_size)]", source)
                self.assertIn("rotations=False", source)
                self.assertNotIn("import cliport.utils as utils", source)

    def test_river_tasks_target_visible_item_height(self):
        for module in task_class_names():
            if module.startswith("move_") and "_from_" in module and "_to_" in module:
                source = (Path("cliport/generated_tasks") / f"{module}.py").read_text(encoding="utf-8")
                if "minecraft/" in source:
                    self.assertIn(", 0.04))", source)
                    self.assertRegex(source, r"metric=[\"']zone[\"']")
                    self.assertIn("np.random.uniform", source)
                    self.assertIn("zone_size)]", source)
                    self.assertIn("rotations=False", source)


if __name__ == "__main__":
    unittest.main()

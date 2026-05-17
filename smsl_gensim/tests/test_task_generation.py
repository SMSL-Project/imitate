import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from smsl_gensim import available_scenarios
from smsl_gensim.task_generation import generate_scenario_tasks
from smsl_gensim.task_llm import PROMPT_DIR, _task_spec_prompt, load_env_file
from smsl_gensim.task_registry import GENERATED_TASKS_BACKUP_DIR, GENERATED_TASKS_DIR
from smsl_gensim.task_validation import validate_task_code


class TaskGenerationTests(unittest.TestCase):
    def test_generate_all_included_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            count = 0
            for scenario in available_scenarios():
                result = generate_scenario_tasks(scenario, root / "tasks", root / "task_lists", backend="template")
                count += len(result.tasks)
            self.assertEqual(count, 43)
            chess = (root / "tasks" / "move_star_chess_to_block_uppercase_A.py").read_text(encoding="utf-8")
            river = (root / "tasks" / "move_sheep_from_red_land_to_boat.py").read_text(encoding="utf-8")
            river_land = (root / "tasks" / "move_sheep_from_boat_to_red_land.py").read_text(encoding="utf-8")
            self.assertIn(", 0.040)", chess)
            self.assertIn("zone_size = (0.150, 0.150, 0.080)", chess)
            self.assertIn("np.random.uniform", chess)
            self.assertIn('metric="zone"', chess)
            self.assertIn("params=[(zone_pose, zone_size)]", chess)
            self.assertIn(", 0.04)", river)
            self.assertIn("zone_size = (0.18, 0.20, 0.10)", river)
            self.assertIn("slots[np.random.randint(len(slots))]", river)
            self.assertIn("np.random.uniform(-0.015, 0.015)", river)
            self.assertIn("np.random.uniform(-0.010, 0.010)", river)
            self.assertIn("np.random.uniform", river)
            self.assertIn('metric="zone"', river)
            self.assertIn("np.random.uniform(-0.035, 0.035)", river_land)
            self.assertIn("np.random.uniform(-0.050, 0.050)", river_land)

    def test_llm_backend_writes_model_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeOpenAIClient()
            result = generate_scenario_tasks(
                "hanoi",
                root / "tasks",
                root / "task_lists",
                backend="llm",
                model="fake-model",
                overwrite=True,
                client=client,
            )
            source = (root / "tasks" / "move_gray_hanoi_ring_to_the_center_of_blue_hanoi_stand.py").read_text(
                encoding="utf-8"
            )
            self.assertEqual(result.backend, "llm")
            self.assertEqual(result.model, "fake-model")
            self.assertEqual(client.calls, 18)
            self.assertIn("class MoveGrayHanoiRingToTheCenterOfBlueHanoiStand(Task):", source)
            self.assertIn("env.asset_poses_dict[env.smsl_scene_state]", source)

    def test_llm_backend_repairs_failed_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = RepairingRegionOpenAIClient()
            result = generate_scenario_tasks(
                "chess",
                root / "tasks",
                root / "task_lists",
                backend="llm",
                model="fake-model",
                overwrite=True,
                client=client,
            )
            source = (root / "tasks" / "move_circle_chess_to_block_0.py").read_text(encoding="utf-8")
            self.assertEqual(result.written_count, 18)
            self.assertGreater(client.repair_prompts, 0)
            self.assertIn("np.random.uniform", source)

    def test_llm_backend_does_not_fall_back_to_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                generate_scenario_tasks(
                    "hanoi",
                    root / "tasks",
                    root / "task_lists",
                    backend="llm",
                    client=BadOpenAIClient(),
                )
            self.assertFalse((root / "tasks" / "move_gray_hanoi_ring_to_the_center_of_blue_hanoi_stand.py").exists())

    def test_failed_overwrite_keeps_existing_task_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "tasks" / "move_gray_hanoi_ring_to_the_center_of_blue_hanoi_stand.py"
            path.parent.mkdir()
            path.write_text("old task file\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                generate_scenario_tasks(
                    "hanoi",
                    root / "tasks",
                    root / "task_lists",
                    backend="llm",
                    overwrite=True,
                    client=LateFailingOpenAIClient(),
                )
            self.assertEqual(path.read_text(encoding="utf-8"), "old task file\n")

    def test_llm_backend_rejects_flat_asset_ids_dict_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "by category"):
                generate_scenario_tasks(
                    "hanoi",
                    Path(tmp) / "tasks",
                    Path(tmp) / "task_lists",
                    backend="llm",
                    client=FlatAssetOpenAIClient(),
                )

    def test_llm_backend_rejects_pose_as_goal_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "object IDs"):
                generate_scenario_tasks(
                    "hanoi",
                    Path(tmp) / "tasks",
                    Path(tmp) / "task_lists",
                    backend="llm",
                    client=PoseObjectOpenAIClient(),
                )

    def test_llm_backend_rejects_bad_utils_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "from cliport.utils import utils"):
                generate_scenario_tasks(
                    "hanoi",
                    Path(tmp) / "tasks",
                    Path(tmp) / "task_lists",
                    backend="llm",
                    client=BadUtilsImportOpenAIClient(),
                )

    def test_llm_backend_rejects_bad_utils_apply_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "utils.apply"):
                generate_scenario_tasks(
                    "hanoi",
                    Path(tmp) / "tasks",
                    Path(tmp) / "task_lists",
                    backend="llm",
                    client=BadUtilsApplyOpenAIClient(),
                )

    def test_llm_backend_rejects_rotation_sensitive_goals(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "rotations=False"):
                generate_scenario_tasks(
                    "hanoi",
                    Path(tmp) / "tasks",
                    Path(tmp) / "task_lists",
                    backend="llm",
                    client=RotationOpenAIClient(),
                )

    def test_llm_backend_rejects_pose_metric_for_region_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "zone metric"):
                generate_scenario_tasks(
                    "chess",
                    Path(tmp) / "tasks",
                    Path(tmp) / "task_lists",
                    backend="llm",
                    client=RegionPoseMetricOpenAIClient(),
                )

    def test_llm_backend_rejects_bad_chess_target_height(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "chess target offset"):
                generate_scenario_tasks(
                    "chess",
                    Path(tmp) / "tasks",
                    Path(tmp) / "task_lists",
                    backend="llm",
                    client=ChessBadHeightOpenAIClient(),
                )

    def test_llm_backend_rejects_region_goal_without_target_poses(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "targ_poses"):
                generate_scenario_tasks(
                    "river_crossing",
                    Path(tmp) / "tasks",
                    Path(tmp) / "task_lists",
                    backend="llm",
                    client=RegionNoTargetPosesOpenAIClient(),
                )

    def test_llm_backend_rejects_region_goal_without_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "np.random.uniform"):
                generate_scenario_tasks(
                    "chess",
                    Path(tmp) / "tasks",
                    Path(tmp) / "task_lists",
                    backend="llm",
                    client=RegionNoSamplingOpenAIClient(),
                )

    def test_llm_backend_rejects_region_goal_with_small_zone_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "zone_size"):
                generate_scenario_tasks(
                    "river_crossing",
                    Path(tmp) / "tasks",
                    Path(tmp) / "task_lists",
                    backend="llm",
                    client=RegionSmallZoneOpenAIClient(),
                )

    def test_llm_backend_rejects_large_river_target_jitter(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "river target jitter"):
                generate_scenario_tasks(
                    "river_crossing",
                    Path(tmp) / "tasks",
                    Path(tmp) / "task_lists",
                    backend="llm",
                    client=RegionLargeJitterOpenAIClient(),
                )

    def test_llm_backend_rejects_fixed_river_target_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "permutable slots"):
                generate_scenario_tasks(
                    "river_crossing",
                    Path(tmp) / "tasks",
                    Path(tmp) / "task_lists",
                    backend="llm",
                    client=RegionFixedSlotOpenAIClient(),
                )

    def test_llm_backend_rejects_region_goal_with_raw_target_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "target_pose"):
                generate_scenario_tasks(
                    "river_crossing",
                    Path(tmp) / "tasks",
                    Path(tmp) / "task_lists",
                    backend="llm",
                    client=RegionRawTargetPoseOpenAIClient(),
                )

    def test_llm_backend_rejects_region_goal_that_samples_target_position_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "target_position"):
                generate_scenario_tasks(
                    "river_crossing",
                    Path(tmp) / "tasks",
                    Path(tmp) / "task_lists",
                    backend="llm",
                    client=RegionRandomTargetPositionOpenAIClient(),
                )

    def test_load_env_file_sets_openai_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("OPENAI_API_KEY='abc123'\n", encoding="utf-8")
            old = os.environ.pop("OPENAI_API_KEY", None)
            try:
                load_env_file(path)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "abc123")
            finally:
                os.environ.pop("OPENAI_API_KEY", None)
                if old is not None:
                    os.environ["OPENAI_API_KEY"] = old

    def test_task_spec_prompt_uses_initializer_context(self):
        prompt = _task_spec_prompt("chess", "move-star-chess-to-block-uppercase-A")
        self.assertIn("Scene initializer", prompt)
        self.assertIn("Do not inspect board slot joints", prompt)
        self.assertIn("Chess target offsets are supplied by the code prompt", prompt)
        self.assertNotIn("Inspect chess/board.urdf", prompt)
        self.assertNotIn('"A":', prompt)

    def test_code_prompt_keeps_skeleton_without_target_templates(self):
        prompt = (PROMPT_DIR / "task_code_prompt.txt").read_text(encoding="utf-8")
        self.assertIn("for chess target labels", prompt)
        self.assertIn("destination slots are permutable", prompt)
        self.assertIn("only the XY extents", prompt)
        self.assertIn("target z is only the oracle placement height", prompt)
        self.assertIn("Compact module skeleton", prompt)
        self.assertIn("Zone target skeleton", prompt)
        self.assertIn("class {class_name}(Task):", prompt)
        self.assertIn("add one goal", prompt)
        self.assertIn("self.add_goal(", prompt)
        self.assertIn("objs=[obj_id]", prompt)
        self.assertIn('metric="<pose_or_zone>"', prompt)
        self.assertIn("zone_pose = (zone_position, base_pose[1])", prompt)
        self.assertIn("target_pose = (target_position, zone_pose[1])", prompt)
        self.assertNotIn("Chess target pattern", prompt)
        self.assertNotIn("River crossing target pattern", prompt)
        self.assertNotIn("self.add_goal(\n    objs=[env.asset_ids_dict", prompt)
        self.assertNotIn("metric=metric", prompt)

    def test_validator_accepts_simple_metric_alias(self):
        validate_task_code(
            '''import numpy as np
from cliport.tasks.task import Task


class MoveGrayHanoiRingToTheCenterOfBlueHanoiStand(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move the gray hanoi ring"
        self.task_completed_desc = "done moving the gray hanoi ring."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        ring_id = env.asset_ids_dict["rigid"]["hanoi/disk_gray.urdf"]
        target_pose = env.asset_poses_dict[env.smsl_scene_state]["hanoi/stand_blue.urdf"]
        metric = "pose"
        rotations = False
        self.add_goal(
            objs=[ring_id],
            matches=np.ones((1, 1)),
            targ_poses=[target_pose],
            replace=False,
            rotations=rotations,
            metric=metric,
            params=None,
            step_max_reward=1,
            language_goal=self.lang_template,
        )
''',
            "MoveGrayHanoiRingToTheCenterOfBlueHanoiStand",
            "move-gray-hanoi-ring-to-the-center-of-blue-hanoi-stand",
        )

    def test_existing_generated_tasks_are_backed_up(self):
        backup = {path.name for path in GENERATED_TASKS_BACKUP_DIR.glob("*.py")}
        active = {path.name for path in GENERATED_TASKS_DIR.glob("*.py")}
        self.assertIn("align_balls_in_colored_boxes.py", backup)
        self.assertIn("move_gray_hanoi_ring_to_the_center_of_blue_hanoi_stand.py", backup)
        self.assertNotIn("align_balls_in_colored_boxes.py", active)


class FakeOpenAIClient:
    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        prompt = kwargs["messages"][1]["content"]
        if _is_spec_prompt(prompt):
            return _spec_response(prompt)
        class_name = _prompt_value(prompt, "Class name")
        code = f'''```python
import numpy as np
from cliport.tasks.task import Task


class {class_name}(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move the gray hanoi ring"
        self.task_completed_desc = "done moving the gray hanoi ring."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        ring_id = env.asset_ids_dict["rigid"]["hanoi/disk_gray.urdf"]
        target_pose = env.asset_poses_dict[env.smsl_scene_state]["hanoi/stand_blue.urdf"]
        self.add_goal(
            objs=[ring_id],
            matches=np.ones((1, 1)),
            targ_poses=[target_pose],
            replace=False,
            rotations=False,
            metric="pose",
            params=None,
            step_max_reward=1,
            language_goal=self.lang_template,
        )
```'''
        message = SimpleNamespace(content=code)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class RepairingRegionOpenAIClient:
    def __init__(self):
        self.failed_once = False
        self.repair_prompts = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        if _is_spec_prompt(prompt):
            return _spec_response(prompt)
        class_name = _prompt_value(prompt, "Class name")
        task_name = _code_prompt_task_name(prompt)
        if "Validation failed" in prompt:
            self.repair_prompts += 1
            return _response(_valid_chess_code(class_name, task_name))
        if not self.failed_once:
            self.failed_once = True
            return _response(_invalid_chess_code(class_name, task_name))
        return _response(_valid_chess_code(class_name, task_name))


class BadOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        message = SimpleNamespace(content="```python\nclass Wrong(Task):\n    pass\n```")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class LateFailingOpenAIClient:
    def __init__(self):
        self.code_calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        if _is_spec_prompt(prompt):
            return _spec_response(prompt)
        self.code_calls += 1
        if self.code_calls == 1:
            return FakeOpenAIClient().create(**kwargs)
        return BadOpenAIClient().create(**kwargs)


class FlatAssetOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        if _is_spec_prompt(prompt):
            return _spec_response(prompt)
        class_name = _prompt_value(prompt, "Class name")
        return _response(f'''```python
import numpy as np
from cliport.tasks.task import Task


class {class_name}(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move the gray ring"
        self.task_completed_desc = "done moving the gray ring."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        ring_id = env.asset_ids_dict["hanoi/disk_gray.urdf"]
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
```''')


class PoseObjectOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        if _is_spec_prompt(prompt):
            return _spec_response(prompt)
        class_name = _prompt_value(prompt, "Class name")
        return _response(f'''```python
import numpy as np
from cliport.tasks.task import Task


class {class_name}(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move the gray ring"
        self.task_completed_desc = "done moving the gray ring."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        ring_id = env.asset_ids_dict["rigid"]["hanoi/disk_gray.urdf"]
        ring_pose = env.get_object_pose(ring_id)
        target_pose = env.asset_poses_dict[env.smsl_scene_state]["hanoi/stand_blue.urdf"]
        self.add_goal(
            objs=[ring_pose],
            matches=np.ones((1, 1)),
            targ_poses=[target_pose],
            replace=False,
            rotations=True,
            metric="pose",
            params=None,
            step_max_reward=1,
            language_goal=self.lang_template,
        )
```''')


class BadUtilsImportOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        if _is_spec_prompt(prompt):
            return _spec_response(prompt)
        class_name = _prompt_value(prompt, "Class name")
        return _response(f'''```python
import numpy as np
from cliport.tasks.task import Task
import cliport.utils as utils


class {class_name}(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move the gray ring"
        self.task_completed_desc = "done moving the gray ring."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        ring_id = env.asset_ids_dict["rigid"]["hanoi/disk_gray.urdf"]
        target_pose = utils.apply(
            env.asset_poses_dict[env.smsl_scene_state]["hanoi/stand_blue.urdf"],
            (0, 0, 0),
        )
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
```''')


class BadUtilsApplyOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        if _is_spec_prompt(prompt):
            return _spec_response(prompt)
        class_name = _prompt_value(prompt, "Class name")
        return _response(f'''```python
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class {class_name}(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move the gray ring"
        self.task_completed_desc = "done moving the gray ring."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        ring_id = env.asset_ids_dict["rigid"]["hanoi/disk_gray.urdf"]
        target_pose = utils.apply(
            env.asset_poses_dict[env.smsl_scene_state]["hanoi/stand_blue.urdf"],
            (0, 0, 0),
            (0, 0, 0),
        )
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
```''')


class RotationOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        if _is_spec_prompt(prompt):
            return _spec_response(prompt)
        class_name = _prompt_value(prompt, "Class name")
        return _response(f'''```python
import numpy as np
from cliport.tasks.task import Task


class {class_name}(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move the gray ring"
        self.task_completed_desc = "done moving the gray ring."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        ring_id = env.asset_ids_dict["rigid"]["hanoi/disk_gray.urdf"]
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
```''')


class RegionPoseMetricOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        if _is_spec_prompt(prompt):
            return _spec_response(prompt)
        class_name = _prompt_value(prompt, "Class name")
        return _response(f'''```python
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class {class_name}(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move the circle chess piece"
        self.task_completed_desc = "done moving the circle chess piece."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        obj_id = env.asset_ids_dict["rigid"]["chess/circle_chess.urdf"]
        board_id = env.asset_ids_dict["fixed"]["chess/board.urdf"]
        board_pose = env.get_object_pose(board_id)
        target_pose = utils.apply(board_pose, (-0.150, 0.100, 0.040))
        self.add_goal(
            objs=[obj_id],
            matches=np.ones((1, 1)),
            targ_poses=[target_pose],
            replace=False,
            rotations=True,
            metric="pose",
            params=None,
            step_max_reward=1,
            language_goal=self.lang_template,
        )
```''')


class ChessBadHeightOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        if _is_spec_prompt(prompt):
            return _spec_response(prompt)
        class_name = _prompt_value(prompt, "Class name")
        return _response(f'''```python
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class {class_name}(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move the star chess piece"
        self.task_completed_desc = "done moving the star chess piece."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        obj_id = env.asset_ids_dict["rigid"]["chess/star_chess.urdf"]
        board_id = env.asset_ids_dict["fixed"]["chess/board.urdf"]
        board_pose = env.get_object_pose(board_id)
        zone_position = utils.apply(board_pose, (-0.150, -0.100, 0.020))
        zone_pose = (zone_position, board_pose[1])
        dx = np.random.uniform(-0.020, 0.020)
        dy = np.random.uniform(-0.020, 0.020)
        target_position = utils.apply(zone_pose, (dx, dy, 0.0))
        target_pose = (target_position, zone_pose[1])
        zone_size = (0.150, 0.150, 0.080)
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
```''')


class RegionNoTargetPosesOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        if _is_spec_prompt(prompt):
            return _spec_response(prompt)
        class_name = _prompt_value(prompt, "Class name")
        return _response(f'''```python
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class {class_name}(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move the grass"
        self.task_completed_desc = "done moving the grass."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        obj_id = env.asset_ids_dict["rigid"]["minecraft/grass.urdf"]
        place_id = env.asset_ids_dict["fixed"]["minecraft/red_land.urdf"]
        zone_pose = env.get_object_pose(place_id)
        dx = np.random.uniform(-0.02, 0.02)
        dy = np.random.uniform(-0.04, 0.04)
        target_pose = utils.apply(zone_pose, (0.02 + dx, 0.02 + dy, 0.04))
        zone_size = (0.26, 0.28, 0.10)
        self.add_goal(
            objs=[obj_id],
            matches=np.ones((1, 1)),
            replace=False,
            rotations=False,
            metric="zone",
            params=[(zone_pose, zone_size)],
            step_max_reward=1,
            language_goal=self.lang_template,
        )
```''')


class RegionNoSamplingOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        if _is_spec_prompt(prompt):
            return _spec_response(prompt)
        class_name = _prompt_value(prompt, "Class name")
        return _response(f'''```python
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class {class_name}(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move the circle chess piece"
        self.task_completed_desc = "done moving the circle chess piece."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        obj_id = env.asset_ids_dict["rigid"]["chess/circle_chess.urdf"]
        board_id = env.asset_ids_dict["fixed"]["chess/board.urdf"]
        board_pose = env.get_object_pose(board_id)
        zone_pose = utils.apply(board_pose, (-0.150, -0.100, 0.040))
        target_pose = utils.apply(zone_pose, (0.0, 0.0, 0.0))
        zone_size = (0.150, 0.150, 0.080)
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
```''')


class RegionSmallZoneOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        if _is_spec_prompt(prompt):
            return _spec_response(prompt)
        class_name = _prompt_value(prompt, "Class name")
        return _response(f'''```python
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class {class_name}(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move Steve to the boat"
        self.task_completed_desc = "done moving Steve."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        obj_id = env.asset_ids_dict["rigid"]["minecraft/steve.urdf"]
        target_id = env.asset_ids_dict["fixed"]["minecraft/boat.urdf"]
        zone_pose = env.get_object_pose(target_id)
        dx = np.random.uniform(-0.01, 0.01)
        dy = np.random.uniform(-0.01, 0.01)
        target_position = utils.apply(zone_pose, (-0.02 + dx, -0.02 + dy, 0.04))
        target_pose = (target_position, zone_pose[1])
        zone_size = (0.1, 0.1, 0.1)
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
```''')


class RegionLargeJitterOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        if _is_spec_prompt(prompt):
            return _spec_response(prompt)
        class_name = _prompt_value(prompt, "Class name")
        return _response(f'''```python
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class {class_name}(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move Steve to the green land"
        self.task_completed_desc = "done moving Steve."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        obj_id = env.asset_ids_dict["rigid"]["minecraft/steve.urdf"]
        target_id = env.asset_ids_dict["fixed"]["minecraft/green_land.urdf"]
        zone_pose = env.get_object_pose(target_id)
        dx = np.random.uniform(-0.05, 0.05)
        dy = np.random.uniform(-0.05, 0.05)
        target_position = utils.apply(zone_pose, (-0.02 + dx, -0.02 + dy, 0.04))
        target_pose = (target_position, zone_pose[1])
        zone_size = (0.26, 0.28, 0.10)
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
```''')


class RegionFixedSlotOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        if _is_spec_prompt(prompt):
            return _spec_response(prompt)
        class_name = _prompt_value(prompt, "Class name")
        return _response(f'''```python
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class {class_name}(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move Steve to the green land"
        self.task_completed_desc = "done moving Steve."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        obj_id = env.asset_ids_dict["rigid"]["minecraft/steve.urdf"]
        target_id = env.asset_ids_dict["fixed"]["minecraft/green_land.urdf"]
        zone_pose = env.get_object_pose(target_id)
        dx = np.random.uniform(-0.035, 0.035)
        dy = np.random.uniform(-0.050, 0.050)
        target_position = utils.apply(zone_pose, (-0.02 + dx, -0.02 + dy, 0.04))
        target_pose = (target_position, zone_pose[1])
        zone_size = (0.26, 0.28, 0.10)
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
```''')


class RegionRawTargetPoseOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        if _is_spec_prompt(prompt):
            return _spec_response(prompt)
        class_name = _prompt_value(prompt, "Class name")
        return _response(f'''```python
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class {class_name}(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move Steve to the boat"
        self.task_completed_desc = "done moving Steve."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        obj_id = env.asset_ids_dict["rigid"]["minecraft/steve.urdf"]
        target_id = env.asset_ids_dict["fixed"]["minecraft/boat.urdf"]
        zone_pose = env.get_object_pose(target_id)
        dx = np.random.uniform(-0.01, 0.01)
        dy = np.random.uniform(-0.01, 0.01)
        target_pose = utils.apply(zone_pose, (-0.02 + dx, -0.02 + dy, 0.04))
        zone_size = (0.18, 0.20, 0.10)
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
```''')


class RegionRandomTargetPositionOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        if _is_spec_prompt(prompt):
            return _spec_response(prompt)
        class_name = _prompt_value(prompt, "Class name")
        return _response(f'''```python
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class {class_name}(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move Steve to the boat"
        self.task_completed_desc = "done moving Steve."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        obj_id = env.asset_ids_dict["rigid"]["minecraft/steve.urdf"]
        target_id = env.asset_ids_dict["fixed"]["minecraft/boat.urdf"]
        zone_pose = env.get_object_pose(target_id)
        target_position = np.random.uniform(
            low=np.array(zone_pose[0]) - np.array((0.18, 0.20, 0.10)) / 2,
            high=np.array(zone_pose[0]) + np.array((0.18, 0.20, 0.10)) / 2,
        )
        target_pose = (target_position, zone_pose[1])
        zone_size = (0.18, 0.20, 0.10)
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
```''')


def _prompt_value(prompt, key):
    prefix = f"{key}: "
    return next(line.strip()[len(prefix):] for line in prompt.splitlines() if line.strip().startswith(prefix))


def _code_prompt_task_name(prompt):
    prefix = '"task-name": '
    line = next(line.strip() for line in prompt.splitlines() if line.strip().startswith(prefix))
    return line[len(prefix):].strip().strip('",')


def _is_spec_prompt(prompt):
    return "Return only JSON" in prompt


def _spec_response(prompt):
    task_name = _prompt_value(prompt, "Task name")
    return _response(json.dumps({
        "task-name": task_name,
        "task-description": "Move the gray hanoi ring to the blue hanoi stand.",
        "urdf-to-use-fixed": ["hanoi/stand_blue.urdf"],
        "urdf-to-use-rigid": ["hanoi/disk_gray.urdf"],
        "urdf-to-inspect": [],
    }))


def _invalid_chess_code(class_name, task_name):
    x, y, z = _chess_offset(task_name)
    return f'''```python
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class {class_name}(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move the chess piece"
        self.task_completed_desc = "done moving the chess piece."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        obj_id = env.asset_ids_dict["rigid"]["chess/circle_chess.urdf"]
        board_id = env.asset_ids_dict["fixed"]["chess/board.urdf"]
        board_pose = env.get_object_pose(board_id)
        zone_position = utils.apply(board_pose, ({x:.3f}, {y:.3f}, {z:.3f}))
        zone_pose = (zone_position, board_pose[1])
        target_pose = (zone_position, zone_pose[1])
        zone_size = (0.150, 0.150, 0.080)
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
```'''


def _valid_chess_code(class_name, task_name):
    x, y, z = _chess_offset(task_name)
    return f'''```python
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class {class_name}(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.lang_template = "move the chess piece"
        self.task_completed_desc = "done moving the chess piece."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        obj_id = env.asset_ids_dict["rigid"]["chess/circle_chess.urdf"]
        board_id = env.asset_ids_dict["fixed"]["chess/board.urdf"]
        board_pose = env.get_object_pose(board_id)
        zone_position = utils.apply(board_pose, ({x:.3f}, {y:.3f}, {z:.3f}))
        zone_pose = (zone_position, board_pose[1])
        dx = np.random.uniform(-0.020, 0.020)
        dy = np.random.uniform(-0.020, 0.020)
        target_position = utils.apply(zone_pose, (dx, dy, 0.0))
        target_pose = (target_position, zone_pose[1])
        zone_size = (0.150, 0.150, 0.080)
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
```'''


def _chess_offset(task_name):
    label = task_name.split("-chess-to-block-", 1)[1]
    if label.startswith("uppercase-"):
        label = label[len("uppercase-"):]
    if label.startswith("lowercase-"):
        label = label[len("lowercase-"):]
    return {
        "0": (-0.15, -0.10, 0.04),
        "1": (0.15, -0.10, 0.04),
        "a": (-0.15, 0.00, 0.04),
        "c": (0.05, 0.00, 0.04),
        "d": (0.15, 0.00, 0.04),
        "A": (-0.15, 0.10, 0.04),
        "B": (-0.05, 0.10, 0.04),
        "C": (0.05, 0.10, 0.04),
        "D": (0.15, 0.10, 0.04),
    }[label]


def _response(content):
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


if __name__ == "__main__":
    unittest.main()

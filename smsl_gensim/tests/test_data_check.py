import pickle
import tempfile
import unittest

from smsl_gensim import build_collection_plan, check_dataset


class DatasetCheckTests(unittest.TestCase):
    def test_dataset_requires_every_transition_demo(self):
        with tempfile.TemporaryDirectory() as tmp:
            check = check_dataset("hanoi", data_dir=tmp)
        self.assertFalse(check.ok)
        self.assertEqual(check.plan.total_demos, 78)
        self.assertEqual(check.saved_transition_dirs, 0)
        self.assertEqual(check.saved_episodes, 0)
        self.assertEqual(len(check.missing_demos), 78)

    def test_complete_dataset_with_scene_metadata_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_collection_plan("hanoi", data_dir=tmp)
            for job in plan.jobs:
                _write_episode(job)
            check = check_dataset("hanoi", data_dir=tmp)
        self.assertTrue(check.ok)
        self.assertEqual(check.saved_transition_dirs, 78)
        self.assertEqual(check.saved_episodes, 78)

    def test_wrong_scene_metadata_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_collection_plan("hanoi", data_dir=tmp)
            for job in plan.jobs:
                _write_episode(job)
            first = plan.jobs[0]
            _write_episode(first, scene_overrides={"to_state": "State_bad"})
            check = check_dataset("hanoi", data_dir=tmp)
        self.assertFalse(check.ok)
        self.assertEqual(check.scene_mismatches[0].key, "to_state")


def _write_episode(job, episode_id=0, seed=2, scene=True, scene_overrides=None):
    name = f"{episode_id:06d}-{seed}.pkl"
    for field in ("action", "color", "depth", "reward", "info"):
        path = job.output_dir / field
        path.mkdir(parents=True, exist_ok=True)
        with (path / name).open("wb") as handle:
            pickle.dump([], handle)
    if scene:
        _write_scene(job, name, scene_overrides or {})


def _write_scene(job, name, overrides):
    path = job.output_dir / "scene"
    path.mkdir(parents=True, exist_ok=True)
    data = {
        "scenario": job.scenario,
        "from_state": job.from_state,
        "operation": job.operation,
        "to_state": job.to_state,
        "asset_poses": {},
        "asset_ids": {},
    }
    data.update(overrides)
    with (path / name).open("wb") as handle:
        pickle.dump(data, handle)


if __name__ == "__main__":
    unittest.main()

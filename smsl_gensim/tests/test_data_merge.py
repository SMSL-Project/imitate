import json
import pickle
import tempfile
import unittest
from pathlib import Path

from smsl_gensim import build_collection_plan, merge_operation_data


class DataMergeTests(unittest.TestCase):
    def test_merge_operation_data_groups_same_task_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "source"
            output_dir = Path(tmp) / "merged"
            jobs = _same_task_jobs(data_dir)
            for job in jobs:
                _write_episode(job)

            result = merge_operation_data("hanoi", data_dir=str(data_dir), output_dir=str(output_dir))

            task_dir = output_dir / "hanoi" / "train" / jobs[0].task_name
            index = json.loads((task_dir / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(result.episode_count, 2)
            self.assertEqual(len(result.operations), 1)
            self.assertEqual(len(index["episodes"]), 2)
            self.assertEqual(index["episodes"][0]["operation"], jobs[0].operation)
            self.assertNotEqual(index["episodes"][0]["from_state"], index["episodes"][1]["from_state"])
            self.assertTrue((task_dir / "action" / "000000-2.pkl").exists())
            self.assertTrue((task_dir / "action" / "000001-2.pkl").exists())
            self.assertTrue((task_dir / "scene" / "000001-2.pkl").exists())

    def test_merge_operation_data_can_remove_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "source"
            output_dir = Path(tmp) / "merged"
            source_root = data_dir / "hanoi" / "train"
            _write_episode(_same_task_jobs(data_dir)[0])

            result = merge_operation_data("hanoi", data_dir=str(data_dir), output_dir=str(output_dir), remove_source=True)

            self.assertTrue(result.source_removed)
            self.assertFalse(source_root.exists())
            self.assertTrue((output_dir / "hanoi" / "train" / "index.json").exists())

    def test_merge_operation_data_requires_complete_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "source"
            output_dir = Path(tmp) / "merged"
            job = _same_task_jobs(data_dir)[0]
            _write_episode(job, fields=("action",))

            with self.assertRaises(FileNotFoundError):
                merge_operation_data("hanoi", data_dir=str(data_dir), output_dir=str(output_dir))


def _same_task_jobs(data_dir: Path):
    jobs = build_collection_plan("hanoi", data_dir=str(data_dir)).jobs
    by_task = {}
    for job in jobs:
        by_task.setdefault(job.task_name, []).append(job)
    return next(group[:2] for group in by_task.values() if len(group) > 1)


def _write_episode(job, fields=("action", "color", "depth", "reward", "info", "scene")):
    name = "000000-2.pkl"
    for field in fields:
        path = job.output_dir / field
        path.mkdir(parents=True, exist_ok=True)
        with (path / name).open("wb") as handle:
            pickle.dump({"field": field}, handle)


if __name__ == "__main__":
    unittest.main()

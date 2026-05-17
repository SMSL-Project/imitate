import pickle
import tempfile
import unittest
from pathlib import Path

from smsl_gensim import build_collection_plan, check_scenario_status
from smsl_gensim.data_merge import MERGE_FIELDS


TASK = "move-gray-hanoi-ring-to-the-center-of-blue-hanoi-stand"


class StatusTests(unittest.TestCase):
    def test_status_counts_partial_pipeline_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transition_data = root / "smsl"
            operation_data = root / "operations"
            exp_dir = root / "exps"
            eval_dir = root / "evals"
            _write_transition_dataset(transition_data)
            _write_merged_operation(operation_data, TASK, episodes=2)
            _write_checkpoint(exp_dir, TASK, demos=2)
            _write_eval(eval_dir, TASK)
            _write_summary(eval_dir)

            status = check_scenario_status(
                "hanoi",
                transition_data_dir=str(transition_data),
                operation_data_dir=str(operation_data),
                exp_dir=str(exp_dir),
                eval_dir=str(eval_dir),
            )

        self.assertTrue(status.transition_check.ok)
        self.assertEqual(status.merged_count, 1)
        self.assertEqual(status.trained_count, 1)
        self.assertEqual(status.evaluated_count, 1)
        self.assertTrue(status.summary_exists)


def _write_transition_dataset(data_dir: Path) -> None:
    for job in build_collection_plan("hanoi", data_dir=str(data_dir)).jobs:
        _write_transition_episode(job)


def _write_transition_episode(job) -> None:
    name = "000000-2.pkl"
    for field in ("action", "color", "depth", "reward", "info"):
        path = job.output_dir / field
        path.mkdir(parents=True, exist_ok=True)
        with (path / name).open("wb") as handle:
            pickle.dump([], handle)
    path = job.output_dir / "scene"
    path.mkdir(parents=True, exist_ok=True)
    with (path / name).open("wb") as handle:
        pickle.dump(
            {
                "scenario": job.scenario,
                "from_state": job.from_state,
                "operation": job.operation,
                "to_state": job.to_state,
            },
            handle,
        )


def _write_merged_operation(data_dir: Path, task_name: str, episodes: int) -> None:
    for index in range(episodes):
        name = f"{index:06d}-{index}.pkl"
        for field in MERGE_FIELDS:
            path = data_dir / "hanoi" / "train" / task_name / field
            path.mkdir(parents=True, exist_ok=True)
            with (path / name).open("wb") as handle:
                pickle.dump({"field": field}, handle)


def _write_checkpoint(exp_dir: Path, task_name: str, demos: int) -> None:
    path = exp_dir / f"smsl-hanoi-{task_name}-cliport-n{demos}-train" / "checkpoints"
    path.mkdir(parents=True)
    (path / "last.ckpt").write_bytes(b"checkpoint")


def _write_eval(eval_dir: Path, task_name: str) -> None:
    path = eval_dir / "hanoi" / "train"
    path.mkdir(parents=True)
    (path / f"{task_name}.json").write_text("{}\n", encoding="utf-8")


def _write_summary(eval_dir: Path) -> None:
    path = eval_dir / "hanoi" / "train"
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text("{}\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

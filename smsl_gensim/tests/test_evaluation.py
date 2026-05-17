import pickle
import tempfile
import unittest
from pathlib import Path

from smsl_gensim import build_operation_evaluation, build_scenario_evaluation
from smsl_gensim.data_check import DATA_FIELDS
from smsl_gensim.scenarios import load_task_list


TASK = "move-gray-hanoi-ring-to-the-center-of-blue-hanoi-stand"


class EvaluationTests(unittest.TestCase):
    def test_build_operation_evaluation_resolves_model_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "merged"
            exp_dir = root / "exps"
            output_dir = root / "evals"
            _write_merged_operation(data_dir, episodes=2)
            _write_run(exp_dir, train_demos=2)

            plan = build_operation_evaluation(
                "hanoi",
                TASK,
                data_dir=str(data_dir),
                exp_dir=str(exp_dir),
                output_dir=str(output_dir),
            )

            self.assertEqual(plan.task_name, TASK)
            self.assertEqual(plan.dataset.episodes, 2)
            self.assertEqual(plan.train_demos, 2)
            self.assertEqual(len(plan.transitions), 18)
            self.assertEqual(plan.rollout_count, 18)
            self.assertEqual(plan.checkpoint_path.name, "last.ckpt")
            self.assertTrue(plan.train_config_path.exists())
            self.assertEqual(plan.output_path, (output_dir / "hanoi" / "train" / f"{TASK}.json").resolve())

    def test_build_operation_evaluation_requires_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "merged"
            exp_dir = root / "exps"
            _write_merged_operation(data_dir, episodes=1)

            with self.assertRaises(FileNotFoundError):
                build_operation_evaluation("hanoi", TASK, data_dir=str(data_dir), exp_dir=str(exp_dir))

    def test_build_scenario_evaluation_resolves_all_operations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "merged"
            exp_dir = root / "exps"
            output_dir = root / "evals"
            _write_scenario(data_dir, exp_dir, episodes=1)

            plan = build_scenario_evaluation(
                "hanoi",
                data_dir=str(data_dir),
                exp_dir=str(exp_dir),
                output_dir=str(output_dir),
            )

            self.assertEqual(len(plan.operations), len(load_task_list("hanoi")))
            self.assertEqual(plan.rollout_count, 78)
            self.assertEqual(plan.output_path, (output_dir / "hanoi" / "train" / "summary.json").resolve())


def _write_merged_operation(data_dir: Path, episodes: int, task_name: str = TASK) -> None:
    for index in range(episodes):
        name = f"{index:06d}-{index}.pkl"
        for field in DATA_FIELDS:
            path = data_dir / "hanoi" / "train" / task_name / field
            path.mkdir(parents=True, exist_ok=True)
            with (path / name).open("wb") as handle:
                pickle.dump({"field": field}, handle)


def _write_run(exp_dir: Path, train_demos: int, task_name: str = TASK) -> None:
    run_dir = exp_dir / f"smsl-hanoi-{task_name}-cliport-n{train_demos}-train"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / ".hydra").mkdir()
    (run_dir / "checkpoints" / "last.ckpt").write_bytes(b"checkpoint")
    (run_dir / ".hydra" / "config.yaml").write_text("train:\n  task: hanoi\n", encoding="utf-8")


def _write_scenario(data_dir: Path, exp_dir: Path, episodes: int) -> None:
    for task_name in load_task_list("hanoi"):
        _write_merged_operation(data_dir, episodes, task_name)
        _write_run(exp_dir, episodes, task_name)


if __name__ == "__main__":
    unittest.main()

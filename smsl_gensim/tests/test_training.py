import pickle
import tempfile
import unittest
from pathlib import Path

from smsl_gensim import build_operation_training, build_scenario_training
from smsl_gensim.data_check import DATA_FIELDS
from smsl_gensim.scenarios import load_task_list


TASK = "move-gray-hanoi-ring-to-the-center-of-blue-hanoi-stand"


class TrainingTests(unittest.TestCase):
    def test_transporter_uses_lightning_2_epoch_hooks(self):
        path = Path(__file__).resolve().parents[1] / "cliport" / "agents" / "transporter.py"
        text = path.read_text(encoding="utf-8")

        self.assertIn("def on_train_epoch_end", text)
        self.assertIn("def on_validation_epoch_end", text)
        self.assertNotIn("def training_epoch_end", text)
        self.assertNotIn("def validation_epoch_end", text)

    def test_build_operation_training_uses_merged_dataset_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "merged"
            exp_dir = Path(tmp) / "exps"
            _write_merged_operation(data_dir, "train", episodes=2)

            training = build_operation_training("hanoi", TASK, data_dir=str(data_dir), exp_dir=str(exp_dir))

            self.assertEqual(training.n_demos, 2)
            self.assertEqual(training.n_val, 1)
            self.assertEqual(training.train_dataset.episodes, 2)
            self.assertEqual(training.val_dataset.episodes, 2)
            self.assertEqual(training.checkpoint_path, (exp_dir / f"smsl-hanoi-{TASK}-cliport-n2-train" / "checkpoints" / "last.ckpt").resolve())
            self.assertIn(f"train.task={TASK}", training.command)
            self.assertIn(f"train.train_dir={(exp_dir / f'smsl-hanoi-{TASK}-cliport-n2-train').resolve()}", training.command)
            self.assertIn(f"train.train_data_path={training.train_dataset.path}", training.command)
            self.assertIn(f"train.val_data_path={training.val_dataset.path}", training.command)

    def test_build_operation_training_accepts_vram_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "merged"
            _write_merged_operation(data_dir, "train", episodes=2)

            training = build_operation_training(
                "hanoi",
                TASK,
                data_dir=str(data_dir),
                batch_size=1,
                n_rotations=12,
            )

            self.assertIn("train.batch_size=1", training.command)
            self.assertIn("train.n_rotations=12", training.command)

    def test_build_operation_training_rejects_missing_demos(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "merged"
            _write_merged_operation(data_dir, "train", episodes=1)

            with self.assertRaises(ValueError):
                build_operation_training("hanoi", TASK, data_dir=str(data_dir), n_demos=2)

    def test_build_scenario_training_resolves_all_operations(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "merged"
            _write_scenario(data_dir, "train", episodes=2)

            training = build_scenario_training(
                "hanoi",
                data_dir=str(data_dir),
                batch_size=1,
                n_rotations=12,
            )

            self.assertEqual(training.operation_count, len(load_task_list("hanoi")))
            self.assertEqual(training.demo_count, 18)
            self.assertTrue(all("train.batch_size=1" in op.command for op in training.operations))
            self.assertTrue(all("train.n_rotations=12" in op.command for op in training.operations))


def _write_merged_operation(data_dir: Path, split: str, episodes: int, task_name: str = TASK) -> None:
    for index in range(episodes):
        name = f"{index:06d}-{index}.pkl"
        for field in DATA_FIELDS:
            path = data_dir / "hanoi" / split / task_name / field
            path.mkdir(parents=True, exist_ok=True)
            with (path / name).open("wb") as handle:
                pickle.dump({"field": field}, handle)


def _write_scenario(data_dir: Path, split: str, episodes: int) -> None:
    for task_name in load_task_list("hanoi"):
        _write_merged_operation(data_dir, split, episodes, task_name)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from smsl_gensim import clean_dataset, dataset_path


class DataCleanTests(unittest.TestCase):
    def test_clean_dataset_removes_only_selected_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            train = Path(tmp) / "hanoi" / "train"
            val = Path(tmp) / "hanoi" / "val"
            train.mkdir(parents=True)
            val.mkdir(parents=True)
            (train / "demo.txt").write_text("train", encoding="utf-8")
            (val / "demo.txt").write_text("val", encoding="utf-8")

            result = clean_dataset("hanoi", data_dir=tmp)

            self.assertTrue(result.removed)
            self.assertFalse(train.exists())
            self.assertTrue(val.exists())

    def test_clean_dataset_rejects_broad_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                dataset_path("", "train", tmp)
            with self.assertRaises(ValueError):
                dataset_path("hanoi", "all", tmp)


if __name__ == "__main__":
    unittest.main()

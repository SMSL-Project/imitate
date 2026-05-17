from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class CleanResult:
    scenario: str
    split: str
    path: Path
    existed: bool
    removed: bool


def dataset_path(scenario: str, split: str = "train", data_dir: str = "data/smsl") -> Path:
    path = Path(data_dir) / scenario / split
    if not scenario or split not in SPLITS or path.name != split or path.parent.name != scenario:
        raise ValueError("clean-data must target one scenario split")
    return path


def clean_dataset(
    scenario: str,
    split: str = "train",
    data_dir: str = "data/smsl",
    dry_run: bool = False,
) -> CleanResult:
    path = dataset_path(scenario, split, data_dir)
    existed = path.exists()
    if existed and not dry_run:
        shutil.rmtree(path)
    return CleanResult(scenario, split, path, existed, existed and not dry_run)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shutil

from smsl_gensim.collection import CollectionJob, build_collection_plan
from smsl_gensim.data_check import DATA_FIELDS


MERGE_FIELDS = DATA_FIELDS + ("scene",)


@dataclass(frozen=True)
class OperationMerge:
    task_name: str
    output_dir: Path
    episodes: int


@dataclass(frozen=True)
class MergeResult:
    scenario: str
    split: str
    source_root: Path
    output_root: Path
    operations: tuple[OperationMerge, ...]
    source_removed: bool = False

    @property
    def episode_count(self) -> int:
        return sum(operation.episodes for operation in self.operations)


def merge_operation_data(
    scenario: str,
    split: str = "train",
    data_dir: str = "data/smsl",
    output_dir: str = "data/smsl_operations",
    dry_run: bool = False,
    remove_source: bool = False,
) -> MergeResult:
    plan = build_collection_plan(scenario, demos_per_transition=1, split=split, data_dir=data_dir)
    output_root = Path(output_dir) / scenario / split
    entries_by_task = _entries_by_task(plan.jobs)

    operations = tuple(
        OperationMerge(task_name, output_root / task_name, len(entries))
        for task_name, entries in sorted(entries_by_task.items())
    )
    result = MergeResult(scenario, split, plan.output_root, output_root, operations)
    if dry_run:
        return result

    if output_root.exists():
        shutil.rmtree(output_root)
    for task_name, entries in entries_by_task.items():
        _write_operation(output_root / task_name, task_name, scenario, split, entries)
    _write_root_index(result)
    if remove_source and plan.output_root.exists():
        shutil.rmtree(plan.output_root)
        result = MergeResult(scenario, split, plan.output_root, output_root, operations, True)
    return result


def _entries_by_task(jobs: tuple[CollectionJob, ...]) -> dict[str, list[dict]]:
    entries_by_task = {}
    seen_dirs = set()
    for job in jobs:
        if job.output_dir in seen_dirs:
            continue
        seen_dirs.add(job.output_dir)
        for source_name in _episode_names(job):
            entries_by_task.setdefault(job.task_name, []).append(_entry(job, source_name))
    return entries_by_task


def _episode_names(job: CollectionJob) -> tuple[str, ...]:
    action_dir = job.output_dir / "action"
    if not action_dir.exists():
        return ()
    names = tuple(path.name for path in sorted(action_dir.glob("*.pkl")))
    for name in names:
        missing = [field for field in MERGE_FIELDS if not (job.output_dir / field / name).exists()]
        if missing:
            raise FileNotFoundError(f"{job.output_dir}: {name} missing {', '.join(missing)}")
    return names


def _entry(job: CollectionJob, source_name: str) -> dict:
    return {
        "from_state": job.from_state,
        "operation": job.operation,
        "to_state": job.to_state,
        "task_name": job.task_name,
        "source_dir": str(job.output_dir),
        "source_file": source_name,
        "seed": _seed(source_name),
    }


def _write_operation(path: Path, task_name: str, scenario: str, split: str, entries: list[dict]) -> None:
    for index, entry in enumerate(entries):
        target_name = f"{index:06d}-{entry['seed']}.pkl"
        entry["episode"] = index
        entry["file"] = target_name
        for field in MERGE_FIELDS:
            target_dir = path / field
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(entry["source_dir"]) / field / entry["source_file"], target_dir / target_name)
    _write_json(
        path / "index.json",
        {"scenario": scenario, "split": split, "task_name": task_name, "episodes": entries},
    )


def _write_root_index(result: MergeResult) -> None:
    _write_json(
        result.output_root / "index.json",
        {
            "scenario": result.scenario,
            "split": result.split,
            "source_root": str(result.source_root),
            "operations": [
                {
                    "task_name": operation.task_name,
                    "episodes": operation.episodes,
                    "path": str(operation.output_dir),
                }
                for operation in result.operations
            ],
        },
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _seed(name: str) -> str:
    return name.rsplit("-", 1)[1].replace(".pkl", "")

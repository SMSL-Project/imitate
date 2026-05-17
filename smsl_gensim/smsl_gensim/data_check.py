from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import pickle

from smsl_gensim.collection import CollectionJob, CollectionPlan, build_collection_plan


DATA_FIELDS = ("action", "color", "depth", "reward", "info")


@dataclass(frozen=True)
class MissingDemo:
    job: CollectionJob
    expected: int
    saved: int


@dataclass(frozen=True)
class SceneMismatch:
    path: Path
    key: str
    expected: str
    actual: object


@dataclass(frozen=True)
class DatasetCheck:
    plan: CollectionPlan
    saved_transition_dirs: int
    saved_episodes: int
    missing_demos: Tuple[MissingDemo, ...]
    missing_episode_files: Tuple[Path, ...]
    missing_scene_files: Tuple[Path, ...]
    scene_mismatches: Tuple[SceneMismatch, ...]

    @property
    def ok(self) -> bool:
        return not (
            self.missing_demos
            or self.missing_episode_files
            or self.missing_scene_files
            or self.scene_mismatches
        )


def check_dataset(
    scenario: str,
    demos_per_transition: int = 1,
    split: str = "train",
    data_dir: str = "data/smsl",
) -> DatasetCheck:
    plan = build_collection_plan(scenario, demos_per_transition, split, data_dir)
    jobs = _transition_jobs(plan)
    missing_demos = []
    missing_episode_files = []
    missing_scene_files = []
    scene_mismatches = []
    saved_transition_dirs = 0
    saved_episodes = 0

    for job in jobs:
        action_files = _episode_files(job.output_dir)
        saved = len(action_files)
        saved_episodes += saved
        saved_transition_dirs += int(saved >= plan.demos_per_transition)
        if saved < plan.demos_per_transition:
            missing_demos.append(MissingDemo(job, plan.demos_per_transition, saved))
        for path in action_files:
            missing_episode_files.extend(_missing_episode_files(job.output_dir, path.name))
            scene_path = job.output_dir / "scene" / path.name
            if scene_path.exists():
                scene_mismatches.extend(_scene_mismatches(scene_path, job))
            else:
                missing_scene_files.append(scene_path)

    return DatasetCheck(
        plan=plan,
        saved_transition_dirs=saved_transition_dirs,
        saved_episodes=saved_episodes,
        missing_demos=tuple(missing_demos),
        missing_episode_files=tuple(missing_episode_files),
        missing_scene_files=tuple(missing_scene_files),
        scene_mismatches=tuple(scene_mismatches),
    )


def _transition_jobs(plan: CollectionPlan) -> Tuple[CollectionJob, ...]:
    jobs = {}
    for job in plan.jobs:
        jobs.setdefault(job.output_dir, job)
    return tuple(jobs.values())


def _episode_files(output_dir: Path) -> Tuple[Path, ...]:
    action_dir = output_dir / "action"
    if not action_dir.exists():
        return ()
    return tuple(sorted(action_dir.glob("*.pkl")))


def _missing_episode_files(output_dir: Path, name: str) -> Tuple[Path, ...]:
    return tuple(
        output_dir / field / name
        for field in DATA_FIELDS
        if not (output_dir / field / name).exists()
    )


def _scene_mismatches(path: Path, job: CollectionJob) -> Tuple[SceneMismatch, ...]:
    with path.open("rb") as handle:
        scene = pickle.load(handle)
    expected = {
        "scenario": job.scenario,
        "from_state": job.from_state,
        "operation": job.operation,
        "to_state": job.to_state,
    }
    return tuple(
        SceneMismatch(path, key, value, scene.get(key))
        for key, value in expected.items()
        if scene.get(key) != value
    )

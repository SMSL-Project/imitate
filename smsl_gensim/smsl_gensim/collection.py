from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Tuple
import pickle
import random

from smsl_gensim.demo_plan import Transition, check_demo_plan
from smsl_gensim.scenes import reset_scene
from smsl_gensim.task_registry import ROOT_DIR


@dataclass(frozen=True)
class CollectionJob:
    scenario: str
    split: str
    demo_index: int
    from_state: str
    operation: str
    to_state: str
    task_name: str
    output_dir: Path


@dataclass(frozen=True)
class CollectionPlan:
    scenario: str
    split: str
    demos_per_transition: int
    output_root: Path
    transitions: Tuple[Transition, ...]
    jobs: Tuple[CollectionJob, ...]

    @property
    def transition_count(self) -> int:
        return len(self.transitions)

    @property
    def total_demos(self) -> int:
        return len(self.jobs)


@dataclass(frozen=True)
class CollectionResult:
    job: CollectionJob
    seed: int
    total_reward: float
    saved: bool
    failure_reason: str = ""


@dataclass(frozen=True)
class CollectionRun:
    plan: CollectionPlan
    results: Tuple[CollectionResult, ...]

    @property
    def saved_count(self) -> int:
        return sum(result.saved for result in self.results)

    @property
    def unsaved_count(self) -> int:
        return len(self.results) - self.saved_count


def build_collection_plan(
    scenario: str,
    demos_per_transition: int = 1,
    split: str = "train",
    data_dir: str = "data/smsl",
) -> CollectionPlan:
    if demos_per_transition < 1:
        raise ValueError("demos_per_transition must be at least 1")

    demo_check = check_demo_plan(scenario)
    if not demo_check.ok:
        raise ValueError(f"{scenario} demo plan is incomplete")

    output_root = Path(data_dir) / scenario / split
    jobs = tuple(
        CollectionJob(
            scenario=scenario,
            split=split,
            demo_index=demo_index,
            from_state=transition.from_state,
            operation=transition.operation,
            to_state=transition.to_state,
            task_name=transition.task_name,
            output_dir=output_root / _transition_dir_name(transition),
        )
        for transition in demo_check.transitions
        for demo_index in range(demos_per_transition)
    )
    return CollectionPlan(
        scenario=scenario,
        split=split,
        demos_per_transition=demos_per_transition,
        output_root=output_root,
        transitions=demo_check.transitions,
        jobs=jobs,
    )


def build_collection_job(
    scenario: str,
    from_state: str,
    operation: str,
    split: str = "train",
    data_dir: str = "data/smsl",
    demo_index: int = 0,
) -> CollectionJob:
    transition = find_transition(scenario, from_state, operation)
    return CollectionJob(
        scenario=scenario,
        split=split,
        demo_index=demo_index,
        from_state=transition.from_state,
        operation=transition.operation,
        to_state=transition.to_state,
        task_name=transition.task_name,
        output_dir=Path(data_dir) / scenario / split / _transition_dir_name(transition),
    )


def find_transition(scenario: str, from_state: str, operation: str) -> Transition:
    for transition in check_demo_plan(scenario).transitions:
        if transition.from_state == from_state and transition.operation == operation:
            return transition
    raise ValueError(f"{scenario} has no transition for {from_state} with {operation}")


def collect_one_demo(
    job: CollectionJob,
    assets_root: str = "",
    disp: bool = False,
    shared_memory: bool = False,
    save_data: bool = True,
) -> CollectionResult:
    import numpy as np
    import pybullet as p
    from cliport import tasks
    from cliport.dataset import RavensDataset
    from cliport.environments.environment import Environment

    assets = assets_root or str(ROOT_DIR / "cliport" / "environments" / "assets")
    task = tasks.names[job.task_name]()
    task.mode = job.split
    cfg = _dataset_cfg(job, assets)
    dataset = RavensDataset(str(job.output_dir), cfg, n_demos=0, augment=False)
    seed = _next_seed(dataset.max_seed, job.split)
    np.random.seed(seed)
    random.seed(seed)

    env = Environment(
        assets,
        disp=disp,
        shared_memory=shared_memory,
        hz=480,
        record_cfg=cfg["record"],
    )
    try:
        env.seed(seed)
        env.set_task(task)
        reset_scene(env, job.scenario, job.from_state)
        task.reset(env)
        agent = task.oracle(env)

        episode = []
        reward = 0
        total_reward = 0.0
        failure_reason = ""
        info = env.info
        obs, _, _, info = env.step()
        for _ in range(task.max_steps):
            act = agent.act(obs, info)
            if act is None:
                failure_reason = "oracle did not find a visible unmatched object to pick"
                break
            episode.append((obs, act, reward, info))
            obs, reward, done, info = env.step(act)
            total_reward += reward
            if done:
                break

        episode.append((obs, None, reward, info))
        saved = bool(save_data and total_reward > 0.99)
        if not saved and not failure_reason:
            failure_reason = _failure_reason(save_data, total_reward)
        if saved:
            dataset.add(seed, episode)
            _save_scene(job, dataset.n_episodes - 1, seed, env.asset_poses_dict[job.from_state], env.asset_ids_dict)
        return CollectionResult(job, seed, total_reward, saved, failure_reason)
    finally:
        p.disconnect()


Collector = Callable[..., CollectionResult]


def iter_collect_demos(
    plan: CollectionPlan,
    assets_root: str = "",
    disp: bool = False,
    collector: Collector = collect_one_demo,
) -> Iterable[CollectionResult]:
    for job in plan.jobs:
        yield collector(job, assets_root=assets_root, disp=disp)


def collect_demos(
    plan: CollectionPlan,
    assets_root: str = "",
    disp: bool = False,
    collector: Collector = collect_one_demo,
) -> CollectionRun:
    return CollectionRun(plan, tuple(iter_collect_demos(plan, assets_root, disp, collector)))


def _transition_dir_name(transition: Transition) -> str:
    return "__".join(
        (
            _path_part(transition.from_state),
            _path_part(transition.task_name),
            _path_part(transition.to_state),
        )
    )


def _path_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "-" for char in value)


def _dataset_cfg(job: CollectionJob, assets_root: str) -> dict:
    return {
        "assets_root": assets_root,
        "data_dir": str(job.output_dir.parent),
        "disp": False,
        "shared_memory": False,
        "task": job.task_name,
        "mode": job.split,
        "n": 1,
        "save_data": True,
        "dataset": {
            "type": "single",
            "images": True,
            "cache": True,
            "augment": {"theta_sigma": 60},
        },
        "record": {
            "save_video": False,
            "save_video_path": str(job.output_dir / "videos"),
            "add_text": False,
            "add_task_text": True,
            "fps": 20,
            "video_height": 640,
            "video_width": 720,
        },
        "train": {"data_augmentation": False},
    }


def _next_seed(max_seed: int, split: str) -> int:
    if max_seed >= 0:
        return max_seed + 2
    return {"train": -2, "val": -1, "test": 9999}[split] + 2


def _failure_reason(save_data: bool, total_reward: float) -> str:
    if not save_data:
        return "save_data is disabled"
    return f"total reward {total_reward:.3f} did not reach success threshold 0.990"


def _save_scene(job: CollectionJob, episode_id: int, seed: int, asset_poses: dict, asset_ids: dict) -> None:
    scene_dir = job.output_dir / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    with (scene_dir / f"{episode_id:06d}-{seed}.pkl").open("wb") as handle:
        pickle.dump(
            {
                "scenario": job.scenario,
                "from_state": job.from_state,
                "operation": job.operation,
                "to_state": job.to_state,
                "asset_poses": asset_poses,
                "asset_ids": asset_ids,
            },
            handle,
        )

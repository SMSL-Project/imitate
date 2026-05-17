from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from smsl_gensim.demo_plan import Transition, iter_transitions
from smsl_gensim.training import (
    DEFAULT_AGENT,
    ROOT_DIR,
    OperationDataset,
    operation_task_name,
    operation_dataset,
)
from smsl_gensim.scenarios import load_task_list


@dataclass(frozen=True)
class OperationEvaluation:
    scenario: str
    task_name: str
    mode: str
    dataset: OperationDataset
    transitions: tuple[Transition, ...]
    checkpoint_path: Path
    train_config_path: Path
    output_path: Path
    agent: str
    train_demos: int
    rollouts_per_transition: int
    assets_root: Path
    disp: bool
    save_video: bool

    @property
    def rollout_count(self) -> int:
        return len(self.transitions) * self.rollouts_per_transition


@dataclass(frozen=True)
class EvalEpisode:
    from_state: str
    operation: str
    to_state: str
    seed: int
    reward: float
    success: bool


@dataclass(frozen=True)
class EvalResult:
    plan: OperationEvaluation
    episodes: tuple[EvalEpisode, ...]
    saved: bool

    @property
    def success_count(self) -> int:
        return sum(episode.success for episode in self.episodes)

    @property
    def success_rate(self) -> float:
        return self.success_count / len(self.episodes) if self.episodes else 0.0

    @property
    def mean_reward(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(episode.reward for episode in self.episodes) / len(self.episodes)


@dataclass(frozen=True)
class ScenarioEvaluation:
    scenario: str
    mode: str
    operations: tuple[OperationEvaluation, ...]
    output_path: Path

    @property
    def rollout_count(self) -> int:
        return sum(operation.rollout_count for operation in self.operations)


@dataclass(frozen=True)
class ScenarioEvalResult:
    plan: ScenarioEvaluation
    results: tuple[EvalResult, ...]
    saved: bool

    @property
    def success_count(self) -> int:
        return sum(result.success_count for result in self.results)

    @property
    def rollout_count(self) -> int:
        return sum(len(result.episodes) for result in self.results)

    @property
    def success_rate(self) -> float:
        if not self.rollout_count:
            return 0.0
        return self.success_count / self.rollout_count

    @property
    def mean_reward(self) -> float:
        episodes = [episode for result in self.results for episode in result.episodes]
        if not episodes:
            return 0.0
        return sum(episode.reward for episode in episodes) / len(episodes)


def build_operation_evaluation(
    scenario: str,
    operation: str,
    mode: str = "train",
    data_dir: str = "data/smsl_operations",
    exp_dir: str = "exps",
    output_dir: str = "evals/smsl_operations",
    agent: str = DEFAULT_AGENT,
    train_demos: int | None = None,
    checkpoint: str = "last.ckpt",
    rollouts_per_transition: int = 1,
    assets_root: str = "",
    disp: bool = False,
    save_video: bool = False,
    exp_name: str = "",
    require_checkpoint: bool = True,
) -> OperationEvaluation:
    task_name = operation_task_name(scenario, operation)
    dataset = operation_dataset(scenario, task_name, mode, data_dir)
    train_demos = train_demos or dataset.episodes
    transitions = tuple(t for t in iter_transitions(scenario) if t.task_name == task_name)
    if not transitions:
        raise ValueError(f"{scenario} has no transitions for {task_name}")
    if rollouts_per_transition < 1:
        raise ValueError("rollouts_per_transition must be at least 1")

    model_task = exp_name or f"smsl-{scenario}-{task_name}"
    run_dir = (Path(exp_dir) / f"{model_task}-{agent}-n{train_demos}-train").resolve()
    checkpoint_path = _checkpoint_path(run_dir, checkpoint)
    train_config_path = run_dir / ".hydra" / "config.yaml"
    if require_checkpoint and not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if require_checkpoint and not train_config_path.exists():
        raise FileNotFoundError(f"training config not found: {train_config_path}")

    return OperationEvaluation(
        scenario=scenario,
        task_name=task_name,
        mode=mode,
        dataset=dataset,
        transitions=transitions,
        checkpoint_path=checkpoint_path,
        train_config_path=train_config_path,
        output_path=(Path(output_dir) / scenario / mode / f"{task_name}.json").resolve(),
        agent=agent,
        train_demos=train_demos,
        rollouts_per_transition=rollouts_per_transition,
        assets_root=Path(assets_root or ROOT_DIR / "cliport" / "environments" / "assets").resolve(),
        disp=disp,
        save_video=save_video,
    )


def build_scenario_evaluation(
    scenario: str,
    mode: str = "train",
    data_dir: str = "data/smsl_operations",
    exp_dir: str = "exps",
    output_dir: str = "evals/smsl_operations",
    agent: str = DEFAULT_AGENT,
    checkpoint: str = "last.ckpt",
    rollouts_per_transition: int = 1,
    assets_root: str = "",
    disp: bool = False,
    save_video: bool = False,
    require_checkpoint: bool = True,
) -> ScenarioEvaluation:
    operations = tuple(
        build_operation_evaluation(
            scenario=scenario,
            operation=task_name,
            mode=mode,
            data_dir=data_dir,
            exp_dir=exp_dir,
            output_dir=output_dir,
            agent=agent,
            checkpoint=checkpoint,
            rollouts_per_transition=rollouts_per_transition,
            assets_root=assets_root,
            disp=disp,
            save_video=save_video,
            require_checkpoint=require_checkpoint,
        )
        for task_name in load_task_list(scenario)
    )
    return ScenarioEvaluation(
        scenario=scenario,
        mode=mode,
        operations=operations,
        output_path=(Path(output_dir) / scenario / mode / "summary.json").resolve(),
    )


def run_operation_evaluation(plan: OperationEvaluation, save: bool = True) -> EvalResult:
    import numpy as np
    import pybullet as p
    import torch
    from cliport import agents, tasks
    from cliport.dataset import RavensDataset
    from cliport.environments.environment import Environment
    from cliport.utils import utils
    from smsl_gensim.scenes import reset_scene
    from torch.utils.data import DataLoader

    cfg = utils.load_hydra_config(str(plan.train_config_path))
    ds = RavensDataset(str(plan.dataset.path), cfg, n_demos=plan.dataset.episodes, augment=False)
    loader = DataLoader(ds, shuffle=False, pin_memory=False, num_workers=1)
    name = f"{plan.task_name}-{plan.agent}-n{plan.train_demos}"
    agent = agents.names[plan.agent](name, cfg, loader, loader)
    agent.load(str(plan.checkpoint_path))
    agent.eval()

    env = Environment(
        str(plan.assets_root),
        disp=plan.disp,
        shared_memory=False,
        hz=480,
        record_cfg=_record_cfg(plan),
    )
    episodes = []
    try:
        for index, transition in enumerate(
            _rollouts(plan.transitions, plan.rollouts_per_transition)
        ):
            seed = index
            np.random.seed(seed)
            task = tasks.names[plan.task_name]()
            task.mode = plan.mode
            env.seed(seed)
            env.set_task(task)
            reset_scene(env, plan.scenario, transition.from_state)
            task.reset(env)
            obs, reward, done, info = env.step()
            total_reward = 0.0

            if plan.save_video:
                env.start_rec(f"{plan.task_name}-{index:06d}")

            for _ in range(task.max_steps):
                with torch.no_grad():
                    act = agent.act(obs, info, None)
                obs, reward, done, info = env.step(act)
                total_reward += reward
                if done:
                    break

            if plan.save_video:
                env.end_rec()

            episodes.append(
                EvalEpisode(
                    transition.from_state,
                    transition.operation,
                    transition.to_state,
                    seed,
                    float(total_reward),
                    total_reward > 0.99,
                )
            )
    finally:
        p.disconnect()

    result = EvalResult(plan, tuple(episodes), save)
    if save:
        _write_result(result)
    return result


def run_scenario_evaluation(plan: ScenarioEvaluation, save: bool = True) -> ScenarioEvalResult:
    result = ScenarioEvalResult(
        plan=plan,
        results=tuple(run_operation_evaluation(operation, save) for operation in plan.operations),
        saved=save,
    )
    if save:
        _write_scenario_result(result)
    return result


def summarize_scenario_results(plan: ScenarioEvaluation, save: bool = True) -> dict:
    operation_results = [_load_operation_summary(operation) for operation in plan.operations]
    rollouts = sum(item["rollouts"] for item in operation_results)
    success_count = sum(item["success_count"] for item in operation_results)
    reward_sum = sum(item["mean_reward"] * item["rollouts"] for item in operation_results)
    payload = {
        "scenario": plan.scenario,
        "mode": plan.mode,
        "operations": len(operation_results),
        "rollouts": rollouts,
        "success_count": success_count,
        "success_rate": success_count / rollouts if rollouts else 0.0,
        "mean_reward": reward_sum / rollouts if rollouts else 0.0,
        "operation_results": operation_results,
    }
    if save:
        plan.output_path.parent.mkdir(parents=True, exist_ok=True)
        plan.output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _checkpoint_path(run_dir: Path, checkpoint: str) -> Path:
    path = Path(checkpoint)
    if path.exists() or path.parent != Path("."):
        return path.resolve()
    name = checkpoint if checkpoint.endswith(".ckpt") else f"{checkpoint}.ckpt"
    return run_dir / "checkpoints" / name


def _rollouts(transitions: tuple[Transition, ...], repeats: int):
    for _ in range(repeats):
        yield from transitions


def _record_cfg(plan: OperationEvaluation) -> dict:
    video_dir = plan.output_path.parent / "videos" / plan.task_name
    return {
        "save_video": plan.save_video,
        "save_video_path": str(video_dir),
        "add_text": True,
        "add_task_text": True,
        "fps": 20,
        "video_height": 640,
        "video_width": 720,
    }


def _write_result(result: EvalResult) -> None:
    payload = {
        "scenario": result.plan.scenario,
        "task_name": result.plan.task_name,
        "mode": result.plan.mode,
        "checkpoint": str(result.plan.checkpoint_path),
        "train_config": str(result.plan.train_config_path),
        "train_demos": result.plan.train_demos,
        "rollouts": len(result.episodes),
        "success_count": result.success_count,
        "success_rate": result.success_rate,
        "mean_reward": result.mean_reward,
        "episodes": [episode.__dict__ for episode in result.episodes],
    }
    result.plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    result.plan.output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_scenario_result(result: ScenarioEvalResult) -> None:
    payload = {
        "scenario": result.plan.scenario,
        "mode": result.plan.mode,
        "operations": len(result.results),
        "rollouts": result.rollout_count,
        "success_count": result.success_count,
        "success_rate": result.success_rate,
        "mean_reward": result.mean_reward,
        "operation_results": [
            {
                "task_name": operation.plan.task_name,
                "rollouts": len(operation.episodes),
                "success_count": operation.success_count,
                "success_rate": operation.success_rate,
                "mean_reward": operation.mean_reward,
                "path": str(operation.plan.output_path),
            }
            for operation in result.results
        ],
    }
    result.plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    result.plan.output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_operation_summary(plan: OperationEvaluation) -> dict:
    payload = json.loads(plan.output_path.read_text(encoding="utf-8"))
    return {
        "task_name": plan.task_name,
        "rollouts": payload["rollouts"],
        "success_count": payload["success_count"],
        "success_rate": payload["success_rate"],
        "mean_reward": payload["mean_reward"],
        "path": str(plan.output_path),
    }

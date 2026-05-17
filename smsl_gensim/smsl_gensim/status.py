from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from smsl_gensim.data_check import DatasetCheck, check_dataset
from smsl_gensim.data_merge import MERGE_FIELDS
from smsl_gensim.scenarios import load_task_list
from smsl_gensim.training import DEFAULT_AGENT


@dataclass(frozen=True)
class OperationStatus:
    task_name: str
    merged_path: Path
    merged_episodes: int
    merged_ok: bool
    checkpoint_path: Path
    trained: bool
    eval_path: Path
    evaluated: bool


@dataclass(frozen=True)
class ScenarioStatus:
    scenario: str
    split: str
    transition_check: DatasetCheck
    operations: tuple[OperationStatus, ...]
    summary_path: Path
    summary_exists: bool

    @property
    def operation_count(self) -> int:
        return len(self.operations)

    @property
    def merged_count(self) -> int:
        return sum(operation.merged_ok for operation in self.operations)

    @property
    def trained_count(self) -> int:
        return sum(operation.trained for operation in self.operations)

    @property
    def evaluated_count(self) -> int:
        return sum(operation.evaluated for operation in self.operations)


def check_scenario_status(
    scenario: str,
    split: str = "train",
    transition_data_dir: str = "data/smsl",
    operation_data_dir: str = "data/smsl_operations",
    exp_dir: str = "exps",
    eval_dir: str = "evals/smsl_operations",
    agent: str = DEFAULT_AGENT,
    transition_demos: int = 1,
    train_demos: int | None = None,
    checkpoint: str = "last.ckpt",
) -> ScenarioStatus:
    transition_check = check_dataset(scenario, transition_demos, split, transition_data_dir)
    operations = tuple(
        _operation_status(
            scenario,
            split,
            task_name,
            operation_data_dir,
            exp_dir,
            eval_dir,
            agent,
            train_demos,
            checkpoint,
        )
        for task_name in load_task_list(scenario)
    )
    summary_path = (Path(eval_dir) / scenario / split / "summary.json").resolve()
    return ScenarioStatus(
        scenario=scenario,
        split=split,
        transition_check=transition_check,
        operations=operations,
        summary_path=summary_path,
        summary_exists=summary_path.exists(),
    )


def _operation_status(
    scenario: str,
    split: str,
    task_name: str,
    operation_data_dir: str,
    exp_dir: str,
    eval_dir: str,
    agent: str,
    train_demos: int | None,
    checkpoint: str,
) -> OperationStatus:
    merged_path = (Path(operation_data_dir) / scenario / split / task_name).resolve()
    merged_episodes, merged_ok = _merged_episode_count(merged_path)
    demos = train_demos if train_demos is not None else merged_episodes
    checkpoint_path = _checkpoint_path(exp_dir, scenario, task_name, agent, demos, checkpoint)
    eval_path = (Path(eval_dir) / scenario / split / f"{task_name}.json").resolve()
    return OperationStatus(
        task_name=task_name,
        merged_path=merged_path,
        merged_episodes=merged_episodes,
        merged_ok=merged_ok,
        checkpoint_path=checkpoint_path,
        trained=checkpoint_path.exists(),
        eval_path=eval_path,
        evaluated=eval_path.exists(),
    )


def _merged_episode_count(path: Path) -> tuple[int, bool]:
    names_by_field = {
        field: {file.name for file in (path / field).glob("*.pkl")}
        for field in MERGE_FIELDS
    }
    action_names = names_by_field["action"]
    return len(action_names), bool(action_names and all(names == action_names for names in names_by_field.values()))


def _checkpoint_path(
    exp_dir: str,
    scenario: str,
    task_name: str,
    agent: str,
    train_demos: int,
    checkpoint: str,
) -> Path:
    path = Path(checkpoint)
    if path.exists() or path.parent != Path("."):
        return path.resolve()
    name = checkpoint if checkpoint.endswith(".ckpt") else f"{checkpoint}.ckpt"
    run_dir = f"smsl-{scenario}-{task_name}-{agent}-n{train_demos}-train"
    return (Path(exp_dir) / run_dir / "checkpoints" / name).resolve()

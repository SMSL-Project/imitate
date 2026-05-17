from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shlex
import subprocess
import sys

from smsl_gensim.data_check import DATA_FIELDS
from smsl_gensim.scenarios import load_task_list


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_AGENT = "cliport"


@dataclass(frozen=True)
class OperationDataset:
    scenario: str
    split: str
    task_name: str
    path: Path
    episodes: int


@dataclass(frozen=True)
class OperationTraining:
    scenario: str
    task_name: str
    model_task: str
    agent: str
    train_dataset: OperationDataset
    val_dataset: OperationDataset
    n_demos: int
    n_val: int
    checkpoint_path: Path
    command: tuple[str, ...]
    cwd: Path


@dataclass(frozen=True)
class ScenarioTraining:
    scenario: str
    split: str
    val_split: str
    operations: tuple[OperationTraining, ...]

    @property
    def operation_count(self) -> int:
        return len(self.operations)

    @property
    def demo_count(self) -> int:
        return sum(operation.n_demos for operation in self.operations)


def operation_task_name(scenario: str, operation: str) -> str:
    task_names = set(load_task_list(scenario))
    for candidate in (operation, operation.replace("_", "-")):
        if candidate in task_names:
            return candidate
    raise ValueError(f"{operation} is not a primitive task for {scenario}")


def operation_dataset(
    scenario: str,
    operation: str,
    split: str = "train",
    data_dir: str = "data/smsl_operations",
) -> OperationDataset:
    task_name = operation_task_name(scenario, operation)
    path = (Path(data_dir) / scenario / split / task_name).resolve()
    return OperationDataset(scenario, split, task_name, path, _episode_count(path))


def build_operation_training(
    scenario: str,
    operation: str,
    split: str = "train",
    val_split: str = "train",
    data_dir: str = "data/smsl_operations",
    exp_dir: str = "exps",
    n_demos: int | None = None,
    n_val: int = 1,
    agent: str = DEFAULT_AGENT,
    n_steps: int | None = None,
    training_step_scale: int | None = None,
    batch_size: int | None = None,
    n_rotations: int | None = None,
    gpu: int = -1,
    log: bool = False,
    debug: bool = False,
    cache: bool = True,
    exp_name: str = "",
) -> OperationTraining:
    train_dataset = operation_dataset(scenario, operation, split, data_dir)
    val_dataset = operation_dataset(scenario, operation, val_split, data_dir)
    n_demos = n_demos or train_dataset.episodes
    _check_demo_count("train", n_demos, train_dataset.episodes)
    _check_demo_count("val", n_val, val_dataset.episodes)

    model_task = exp_name or f"smsl-{scenario}-{train_dataset.task_name}"
    run_dir = (Path(exp_dir) / f"{model_task}-{agent}-n{n_demos}-train").resolve()
    checkpoint_path = run_dir / "checkpoints" / "last.ckpt"
    command = [
        sys.executable,
        "cliport/train.py",
        f"train.task={train_dataset.task_name}",
        f"train.model_task={model_task}",
        f"train.train_dir={run_dir}",
        f"train.agent={agent}",
        f"train.n_demos={n_demos}",
        f"train.n_val={n_val}",
        f"train.gpu={gpu}",
        f"train.log={_hydra_bool(log)}",
        f"debug={_hydra_bool(debug)}",
        f"dataset.cache={_hydra_bool(cache)}",
        f"train.data_dir={(Path(data_dir) / scenario).resolve()}",
        f"train.train_data_path={train_dataset.path}",
        f"train.val_data_path={val_dataset.path}",
    ]
    if n_steps is not None:
        command.append(f"train.n_steps={n_steps}")
    if training_step_scale is not None:
        command.append(f"train.training_step_scale={training_step_scale}")
    if batch_size is not None:
        command.append(f"train.batch_size={batch_size}")
    if n_rotations is not None:
        command.append(f"train.n_rotations={n_rotations}")

    return OperationTraining(
        scenario=scenario,
        task_name=train_dataset.task_name,
        model_task=model_task,
        agent=agent,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_demos=n_demos,
        n_val=n_val,
        checkpoint_path=checkpoint_path,
        command=tuple(command),
        cwd=ROOT_DIR,
    )


def build_scenario_training(
    scenario: str,
    split: str = "train",
    val_split: str = "train",
    data_dir: str = "data/smsl_operations",
    exp_dir: str = "exps",
    n_demos: int | None = None,
    n_val: int = 1,
    agent: str = DEFAULT_AGENT,
    n_steps: int | None = None,
    training_step_scale: int | None = None,
    batch_size: int | None = None,
    n_rotations: int | None = None,
    gpu: int = -1,
    log: bool = False,
    debug: bool = False,
    cache: bool = True,
) -> ScenarioTraining:
    operations = tuple(
        build_operation_training(
            scenario=scenario,
            operation=task_name,
            split=split,
            val_split=val_split,
            data_dir=data_dir,
            exp_dir=exp_dir,
            n_demos=n_demos,
            n_val=n_val,
            agent=agent,
            n_steps=n_steps,
            training_step_scale=training_step_scale,
            batch_size=batch_size,
            n_rotations=n_rotations,
            gpu=gpu,
            log=log,
            debug=debug,
            cache=cache,
        )
        for task_name in load_task_list(scenario)
    )
    return ScenarioTraining(scenario, split, val_split, operations)


def run_operation_training(training: OperationTraining) -> None:
    env = os.environ.copy()
    env.setdefault("GENSIM_ROOT", str(ROOT_DIR))
    subprocess.run(training.command, cwd=training.cwd, env=env, check=True)


def run_scenario_training(training: ScenarioTraining) -> None:
    for operation in training.operations:
        run_operation_training(operation)


def format_command(command: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _episode_count(path: Path) -> int:
    names_by_field = {
        field: {file.name for file in (path / field).glob("*.pkl")}
        for field in DATA_FIELDS
    }
    action_names = names_by_field["action"]
    if not action_names:
        raise FileNotFoundError(f"{path} has no merged episodes")
    for field, names in names_by_field.items():
        if names != action_names:
            raise FileNotFoundError(f"{path} has incomplete {field} episode files")
    return len(action_names)


def _check_demo_count(label: str, requested: int, available: int) -> None:
    if requested > available:
        raise ValueError(f"requested {requested} {label} demos, but {available} are available")


def _hydra_bool(value: bool) -> str:
    return "true" if value else "false"

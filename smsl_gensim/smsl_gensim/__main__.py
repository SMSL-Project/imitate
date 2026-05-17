from __future__ import annotations

import argparse
from dataclasses import replace

from tqdm import tqdm

from smsl_gensim.checks import check_scenario_tasks
from smsl_gensim.collection import (
    build_collection_job,
    build_collection_plan,
    collect_one_demo,
    iter_collect_demos,
)
from smsl_gensim.data_clean import clean_dataset, dataset_path
from smsl_gensim.data_check import check_dataset
from smsl_gensim.data_merge import merge_operation_data
from smsl_gensim.demo_plan import check_demo_plan
from smsl_gensim.evaluation import (
    build_operation_evaluation,
    build_scenario_evaluation,
    run_operation_evaluation,
    run_scenario_evaluation,
    summarize_scenario_results,
)
from smsl_gensim.scenarios import available_scenarios, load_task_list, scenario_summary, state_names
from smsl_gensim.scenes import check_scene
from smsl_gensim.status import check_scenario_status
from smsl_gensim.task_generation import generate_scenario_tasks
from smsl_gensim.training import (
    build_operation_training,
    build_scenario_training,
    format_command,
    run_operation_training,
)


def main(argv: list[str] | None = None) -> None:
    scenarios = available_scenarios()
    parser = argparse.ArgumentParser(prog="python -m smsl_gensim")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available SMSL scenarios.")

    show_parser = subparsers.add_parser("show", help="Show a scenario summary.")
    show_parser.add_argument("scenario", choices=scenarios)

    tasks_parser = subparsers.add_parser("tasks", help="List primitive tasks for a scenario.")
    tasks_parser.add_argument("scenario", choices=scenarios)

    states_parser = subparsers.add_parser("states", help="List SMSL states for a scenario.")
    states_parser.add_argument("scenario", choices=scenarios)

    check_parser = subparsers.add_parser("check-tasks", help="Check primitive task files and assets.")
    check_parser.add_argument("scenario", choices=scenarios)

    check_scene_parser = subparsers.add_parser("check-scene", help="Check one constrained scene.")
    check_scene_parser.add_argument("scenario", choices=scenarios)
    check_scene_parser.add_argument("state")

    check_scenes_parser = subparsers.add_parser("check-scenes", help="Check every constrained scene.")
    check_scenes_parser.add_argument("scenario", choices=scenarios)

    demos_plan_parser = subparsers.add_parser("demos-plan", help="Check SMSL transition demo targets.")
    demos_plan_parser.add_argument("scenario", choices=scenarios)

    generate_parser = subparsers.add_parser("generate-tasks", help="Generate primitive CLIPort task code from SMSL.")
    generate_parser.add_argument("scenario", choices=scenarios)
    generate_parser.add_argument("--backend", choices=("llm", "template"), default="llm")
    generate_parser.add_argument("--model", default="gpt-4o")
    generate_parser.add_argument("--temperature", type=float, default=0.0)
    generate_parser.add_argument("--output-dir", default=None)
    generate_parser.add_argument("--task-list-dir", default=None)
    generate_parser.add_argument("--overwrite", action="store_true")

    status_parser = subparsers.add_parser("status", help="Show the SMSL pipeline status for one scenario.")
    status_parser.add_argument("scenario", choices=scenarios)
    status_parser.add_argument("--mode", choices=("train", "val", "test"), default="train")
    status_parser.add_argument("--n", type=_positive_int, default=1)
    status_parser.add_argument("--train-demos", type=_positive_int, default=None)
    status_parser.add_argument("--transition-data-dir", default="data/smsl")
    status_parser.add_argument("--operation-data-dir", default="data/smsl_operations")
    status_parser.add_argument("--exp-dir", default="exps")
    status_parser.add_argument("--eval-dir", default="evals/smsl_operations")
    status_parser.add_argument("--agent", default="cliport")
    status_parser.add_argument("--checkpoint", default="last.ckpt")

    collect_parser = subparsers.add_parser("collect-demos", help="Collect SMSL transition demos.")
    collect_parser.add_argument("scenario", choices=scenarios)
    collect_parser.add_argument("--n", type=_positive_int, default=1)
    collect_parser.add_argument("--mode", choices=("train", "val", "test"), default="train")
    collect_parser.add_argument("--data-dir", default="data/smsl")
    collect_parser.add_argument("--assets-root", default="")
    collect_parser.add_argument("--disp", action="store_true")
    collect_parser.add_argument("--dry-run", action="store_true")

    collect_missing_parser = subparsers.add_parser("collect-missing", help="Collect missing SMSL transition demos.")
    collect_missing_parser.add_argument("scenario", choices=scenarios)
    collect_missing_parser.add_argument("--n", type=_positive_int, default=1)
    collect_missing_parser.add_argument("--mode", choices=("train", "val", "test"), default="train")
    collect_missing_parser.add_argument("--data-dir", default="data/smsl")
    collect_missing_parser.add_argument("--assets-root", default="")
    collect_missing_parser.add_argument("--disp", action="store_true")
    collect_missing_parser.add_argument("--dry-run", action="store_true")

    collect_one_parser = subparsers.add_parser("collect-one", help="Collect or preview one SMSL transition.")
    collect_one_parser.add_argument("scenario", choices=scenarios)
    collect_one_parser.add_argument("--from-state", required=True)
    collect_one_parser.add_argument("--operation", required=True)
    collect_one_parser.add_argument("--mode", choices=("train", "val", "test"), default="train")
    collect_one_parser.add_argument("--data-dir", default="data/smsl")
    collect_one_parser.add_argument("--assets-root", default="")
    collect_one_parser.add_argument("--disp", action="store_true")
    collect_one_parser.add_argument("--dry-run", action="store_true")

    check_data_parser = subparsers.add_parser("check-data", help="Check collected SMSL demos.")
    check_data_parser.add_argument("scenario", choices=scenarios)
    check_data_parser.add_argument("--n", type=_positive_int, default=1)
    check_data_parser.add_argument("--mode", choices=("train", "val", "test"), default="train")
    check_data_parser.add_argument("--data-dir", default="data/smsl")

    clean_data_parser = subparsers.add_parser("clean-data", help="Remove collected SMSL demos for one scenario split.")
    clean_data_parser.add_argument("scenario", choices=scenarios)
    clean_data_parser.add_argument("--mode", choices=("train", "val", "test"), default="train")
    clean_data_parser.add_argument("--data-dir", default="data/smsl")
    clean_data_parser.add_argument("--dry-run", action="store_true")

    merge_data_parser = subparsers.add_parser("merge-data", help="Merge transition demos into operation datasets.")
    merge_data_parser.add_argument("scenario", choices=scenarios)
    merge_data_parser.add_argument("--mode", choices=("train", "val", "test"), default="train")
    merge_data_parser.add_argument("--data-dir", default="data/smsl")
    merge_data_parser.add_argument("--output-dir", default="data/smsl_operations")
    merge_data_parser.add_argument("--dry-run", action="store_true")
    merge_data_parser.add_argument("--remove-source", action="store_true")

    train_operation_parser = subparsers.add_parser("train-operation", help="Train CLIPort on one merged operation dataset.")
    train_operation_parser.add_argument("scenario", choices=scenarios)
    train_operation_parser.add_argument("operation")
    train_operation_parser.add_argument("--n", type=_positive_int, default=None)
    train_operation_parser.add_argument("--mode", choices=("train", "val", "test"), default="train")
    train_operation_parser.add_argument("--val-mode", choices=("train", "val", "test"), default="train")
    train_operation_parser.add_argument("--data-dir", default="data/smsl_operations")
    train_operation_parser.add_argument("--exp-dir", default="exps")
    train_operation_parser.add_argument("--agent", default="cliport")
    train_operation_parser.add_argument("--n-val", type=_positive_int, default=1)
    train_operation_parser.add_argument("--n-steps", type=_positive_int, default=None)
    train_operation_parser.add_argument("--training-step-scale", type=int, default=None)
    train_operation_parser.add_argument("--batch-size", type=_positive_int, default=None)
    train_operation_parser.add_argument("--n-rotations", type=_positive_int, default=None)
    train_operation_parser.add_argument("--gpu", type=int, default=-1)
    train_operation_parser.add_argument("--log", action="store_true")
    train_operation_parser.add_argument("--debug", action="store_true")
    train_operation_parser.add_argument("--no-cache", action="store_true")
    train_operation_parser.add_argument("--exp-name", default="")
    train_operation_parser.add_argument("--dry-run", action="store_true")

    train_scenario_parser = subparsers.add_parser("train-scenario", help="Train every operation policy for a scenario.")
    train_scenario_parser.add_argument("scenario", choices=scenarios)
    train_scenario_parser.add_argument("--n", type=_positive_int, default=None)
    train_scenario_parser.add_argument("--mode", choices=("train", "val", "test"), default="train")
    train_scenario_parser.add_argument("--val-mode", choices=("train", "val", "test"), default="train")
    train_scenario_parser.add_argument("--data-dir", default="data/smsl_operations")
    train_scenario_parser.add_argument("--exp-dir", default="exps")
    train_scenario_parser.add_argument("--agent", default="cliport")
    train_scenario_parser.add_argument("--n-val", type=_positive_int, default=1)
    train_scenario_parser.add_argument("--n-steps", type=_positive_int, default=None)
    train_scenario_parser.add_argument("--training-step-scale", type=int, default=None)
    train_scenario_parser.add_argument("--batch-size", type=_positive_int, default=None)
    train_scenario_parser.add_argument("--n-rotations", type=_positive_int, default=None)
    train_scenario_parser.add_argument("--gpu", type=int, default=-1)
    train_scenario_parser.add_argument("--log", action="store_true")
    train_scenario_parser.add_argument("--debug", action="store_true")
    train_scenario_parser.add_argument("--no-cache", action="store_true")
    train_scenario_parser.add_argument("--skip-existing", action="store_true")
    train_scenario_parser.add_argument("--dry-run", action="store_true")

    eval_operation_parser = subparsers.add_parser("eval-operation", help="Evaluate one operation policy on SMSL scenes.")
    eval_operation_parser.add_argument("scenario", choices=scenarios)
    eval_operation_parser.add_argument("operation")
    eval_operation_parser.add_argument("--mode", choices=("train", "val", "test"), default="train")
    eval_operation_parser.add_argument("--data-dir", default="data/smsl_operations")
    eval_operation_parser.add_argument("--exp-dir", default="exps")
    eval_operation_parser.add_argument("--output-dir", default="evals/smsl_operations")
    eval_operation_parser.add_argument("--agent", default="cliport")
    eval_operation_parser.add_argument("--train-demos", type=_positive_int, default=None)
    eval_operation_parser.add_argument("--checkpoint", default="last.ckpt")
    eval_operation_parser.add_argument("--n", type=_positive_int, default=1)
    eval_operation_parser.add_argument("--assets-root", default="")
    eval_operation_parser.add_argument("--disp", action="store_true")
    eval_operation_parser.add_argument("--save-video", action="store_true")
    eval_operation_parser.add_argument("--exp-name", default="")
    eval_operation_parser.add_argument("--no-save", action="store_true")
    eval_operation_parser.add_argument("--dry-run", action="store_true")

    eval_scenario_parser = subparsers.add_parser("eval-scenario", help="Evaluate every operation policy for a scenario.")
    eval_scenario_parser.add_argument("scenario", choices=scenarios)
    eval_scenario_parser.add_argument("--mode", choices=("train", "val", "test"), default="train")
    eval_scenario_parser.add_argument("--data-dir", default="data/smsl_operations")
    eval_scenario_parser.add_argument("--exp-dir", default="exps")
    eval_scenario_parser.add_argument("--output-dir", default="evals/smsl_operations")
    eval_scenario_parser.add_argument("--agent", default="cliport")
    eval_scenario_parser.add_argument("--checkpoint", default="last.ckpt")
    eval_scenario_parser.add_argument("--n", type=_positive_int, default=1)
    eval_scenario_parser.add_argument("--assets-root", default="")
    eval_scenario_parser.add_argument("--disp", action="store_true")
    eval_scenario_parser.add_argument("--save-video", action="store_true")
    eval_scenario_parser.add_argument("--skip-existing", action="store_true")
    eval_scenario_parser.add_argument("--no-save", action="store_true")
    eval_scenario_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "list":
        print("\n".join(scenarios))
    elif args.command == "show":
        summary = scenario_summary(args.scenario)
        print(f"name: {summary.name}")
        print(f"graph: {summary.graph_name}")
        print(f"initial_state: {summary.initial_state}")
        print(f"states: {summary.state_count}")
        print(f"transitions: {summary.transition_count}")
        print(f"operations: {summary.operation_count}")
        print(f"tasks: {summary.task_count}")
    elif args.command == "tasks":
        print("\n".join(load_task_list(args.scenario)))
    elif args.command == "states":
        print("\n".join(state_names(args.scenario)))
    elif args.command == "check-tasks":
        check = check_scenario_tasks(args.scenario)
        print(f"{check.scenario}: {check.task_count} tasks")
        _print_status("task files", check.missing_task_files)
        _print_status("registry names", check.missing_registry_names)
        _print_status("task code", check.invalid_task_files)
        _print_status("assets", check.missing_assets)
        if not check.ok:
            raise SystemExit(1)
    elif args.command == "check-scene":
        check = check_scene(args.scenario, args.state)
        _print_scene_check(check)
        if not check.ok:
            raise SystemExit(1)
    elif args.command == "check-scenes":
        checks = tuple(
            check_scene(args.scenario, state)
            for state in state_names(args.scenario)
        )
        missing = tuple(path for check in checks for path in check.missing_assets)
        print(f"{args.scenario}: {len(checks)} scenes")
        _print_status("assets", missing)
        if missing:
            raise SystemExit(1)
    elif args.command == "demos-plan":
        check = check_demo_plan(args.scenario)
        print(f"{check.scenario}: {check.transition_count} transitions")
        _print_status("tasks", check.missing_task_names)
        _print_status("scenes", check.missing_scene_states + check.missing_scene_assets)
        if not check.ok:
            raise SystemExit(1)
    elif args.command == "generate-tasks":
        result = generate_scenario_tasks(
            args.scenario,
            backend=args.backend,
            model=args.model,
            temperature=args.temperature,
            overwrite=args.overwrite,
            progress=lambda items: tqdm(items, desc=f"{args.scenario} tasks", unit="task"),
            **_task_generation_paths(args.output_dir, args.task_list_dir),
        )
        _print_task_generation(result)
    elif args.command == "status":
        status = check_scenario_status(
            args.scenario,
            args.mode,
            args.transition_data_dir,
            args.operation_data_dir,
            args.exp_dir,
            args.eval_dir,
            args.agent,
            args.n,
            args.train_demos,
            args.checkpoint,
        )
        _print_scenario_status(status)
    elif args.command == "collect-demos":
        plan = build_collection_plan(
            args.scenario,
            demos_per_transition=args.n,
            split=args.mode,
            data_dir=args.data_dir,
        )
        _print_collection_plan(plan)
        if args.dry_run:
            print("dry run: no files written")
        else:
            saved = 0
            with tqdm(total=plan.total_demos, desc=f"{plan.scenario} {plan.split}", unit="demo") as progress:
                for index, result in enumerate(
                    iter_collect_demos(plan, assets_root=args.assets_root, disp=args.disp),
                    1,
                ):
                    saved += int(result.saved)
                    progress.write(_collection_result_line(index, plan.total_demos, result))
                    progress.set_postfix(saved=f"{saved}/{index}", reward=f"{result.total_reward:.3f}")
                    progress.update(1)
            print(f"saved demos: {saved}/{plan.total_demos}")
            if saved != plan.total_demos:
                raise SystemExit(1)
    elif args.command == "collect-missing":
        check = check_dataset(
            args.scenario,
            demos_per_transition=args.n,
            split=args.mode,
            data_dir=args.data_dir,
        )
        jobs = _missing_collection_jobs(check)
        print(f"{check.plan.scenario} {check.plan.split}")
        print(f"missing demos: {len(jobs)}")
        print(f"output root: {check.plan.output_root}")
        if args.dry_run:
            print("dry run: no files written")
        elif jobs:
            saved = 0
            with tqdm(total=len(jobs), desc=f"{check.plan.scenario} {check.plan.split} missing", unit="demo") as progress:
                for index, job in enumerate(jobs, 1):
                    result = collect_one_demo(job, assets_root=args.assets_root, disp=args.disp)
                    saved += int(result.saved)
                    progress.write(_collection_result_line(index, len(jobs), result))
                    progress.set_postfix(saved=f"{saved}/{index}", reward=f"{result.total_reward:.3f}")
                    progress.update(1)
            print(f"saved missing demos: {saved}/{len(jobs)}")
            if saved != len(jobs):
                raise SystemExit(1)
        else:
            print("nothing to collect")
    elif args.command == "collect-one":
        job = build_collection_job(
            args.scenario,
            from_state=args.from_state,
            operation=args.operation,
            split=args.mode,
            data_dir=args.data_dir,
        )
        _print_collection_job(job)
        if args.dry_run:
            print("dry run: no files written")
        else:
            result = collect_one_demo(job, assets_root=args.assets_root, disp=args.disp)
            print(f"seed: {result.seed}")
            print(f"total reward: {result.total_reward:.3f}")
            print(f"saved: {result.saved}")
            if result.failure_reason:
                print(f"failure reason: {result.failure_reason}")
            if not result.saved:
                raise SystemExit(1)
    elif args.command == "check-data":
        check = check_dataset(
            args.scenario,
            demos_per_transition=args.n,
            split=args.mode,
            data_dir=args.data_dir,
        )
        _print_dataset_check(check)
        if not check.ok:
            raise SystemExit(1)
    elif args.command == "clean-data":
        path = dataset_path(args.scenario, args.mode, args.data_dir)
        print(f"{args.scenario} {args.mode}")
        print(f"path: {path}")
        result = clean_dataset(args.scenario, args.mode, args.data_dir, args.dry_run)
        if args.dry_run:
            print(f"status: {'exists' if result.existed else 'missing'}")
            print("dry run: no files removed")
        else:
            print(f"removed: {'yes' if result.removed else 'no'}")
    elif args.command == "merge-data":
        result = merge_operation_data(
            args.scenario,
            args.mode,
            args.data_dir,
            args.output_dir,
            args.dry_run,
            args.remove_source,
        )
        print(f"{result.scenario} {result.split}")
        print(f"source root: {result.source_root}")
        print(f"output root: {result.output_root}")
        print(f"operations: {len(result.operations)}")
        print(f"episodes: {result.episode_count}")
        for operation in result.operations:
            print(f"  {operation.task_name}: {operation.episodes}")
        if args.dry_run:
            print("dry run: no files written")
        elif args.remove_source:
            print(f"source removed: {'yes' if result.source_removed else 'no'}")
    elif args.command == "train-operation":
        training = build_operation_training(
            args.scenario,
            args.operation,
            args.mode,
            args.val_mode,
            args.data_dir,
            args.exp_dir,
            args.n,
            args.n_val,
            args.agent,
            args.n_steps,
            args.training_step_scale,
            args.batch_size,
            args.n_rotations,
            args.gpu,
            args.log,
            args.debug,
            not args.no_cache,
            args.exp_name,
        )
        _print_operation_training(training)
        if args.dry_run:
            print("dry run: training not started")
        else:
            run_operation_training(training)
    elif args.command == "train-scenario":
        training = build_scenario_training(
            args.scenario,
            args.mode,
            args.val_mode,
            args.data_dir,
            args.exp_dir,
            args.n,
            args.n_val,
            args.agent,
            args.n_steps,
            args.training_step_scale,
            args.batch_size,
            args.n_rotations,
            args.gpu,
            args.log,
            args.debug,
            not args.no_cache,
        )
        _print_scenario_training(training)
        jobs = _pending_training_operations(training, args.skip_existing)
        print(f"training jobs: {len(jobs)}/{training.operation_count}")
        if args.dry_run:
            print("dry run: training not started")
        else:
            for index, operation in enumerate(jobs, 1):
                print(f"[{index}/{len(jobs)}] {operation.task_name}")
                run_operation_training(operation)
    elif args.command == "eval-operation":
        evaluation = build_operation_evaluation(
            args.scenario,
            args.operation,
            args.mode,
            args.data_dir,
            args.exp_dir,
            args.output_dir,
            args.agent,
            args.train_demos,
            args.checkpoint,
            args.n,
            args.assets_root,
            args.disp,
            args.save_video,
            args.exp_name,
            not args.dry_run,
        )
        _print_operation_evaluation(evaluation)
        if args.dry_run:
            print("dry run: evaluation not started")
        else:
            result = run_operation_evaluation(evaluation, save=not args.no_save)
            print(f"success: {result.success_count}/{len(result.episodes)}")
            print(f"success rate: {result.success_rate:.3f}")
            print(f"mean reward: {result.mean_reward:.3f}")
            if result.saved:
                print(f"saved: {evaluation.output_path}")
    elif args.command == "eval-scenario":
        evaluation = build_scenario_evaluation(
            args.scenario,
            args.mode,
            args.data_dir,
            args.exp_dir,
            args.output_dir,
            args.agent,
            args.checkpoint,
            args.n,
            args.assets_root,
            args.disp,
            args.save_video,
            not (args.dry_run or args.skip_existing),
        )
        _print_scenario_evaluation(evaluation)
        jobs = _pending_evaluations(evaluation, args.skip_existing)
        print(f"evaluation jobs: {len(jobs)}/{len(evaluation.operations)}")
        if args.dry_run:
            print("dry run: evaluation not started")
        else:
            if args.skip_existing:
                _check_evaluation_checkpoints(jobs)
                _run_pending_evaluations(evaluation, jobs, save=not args.no_save)
            else:
                result = run_scenario_evaluation(evaluation, save=not args.no_save)
                print(f"success: {result.success_count}/{result.rollout_count}")
                print(f"success rate: {result.success_rate:.3f}")
                print(f"mean reward: {result.mean_reward:.3f}")
                if result.saved:
                    print(f"saved: {evaluation.output_path}")


def _print_status(label: str, missing: tuple[object, ...]) -> None:
    print(f"{label}: {'missing' if missing else 'ok'}")
    for item in missing:
        print(f"  {item}")


def _print_scene_check(check) -> None:
    print(f"{check.scenario}: {check.state}")
    print(f"fixed objects: {check.fixed_count}")
    print(f"rigid objects: {check.rigid_count}")
    _print_status("assets", check.missing_assets)


def _print_collection_job(job) -> None:
    print(f"{job.scenario}: {job.from_state} -> {job.to_state}")
    print(f"operation: {job.operation}")
    print(f"task: {job.task_name}")
    print(f"mode: {job.split}")
    print(f"output dir: {job.output_dir}")


def _task_generation_paths(output_dir: str | None, task_list_dir: str | None) -> dict:
    paths = {}
    if output_dir:
        paths["output_dir"] = output_dir
    if task_list_dir:
        paths["task_list_dir"] = task_list_dir
    return paths


def _print_task_generation(result) -> None:
    print(f"{result.scenario}: {len(result.tasks)} tasks")
    print(f"backend: {result.backend}")
    if result.model:
        print(f"model: {result.model}")
    print(f"output: {result.output_dir}")
    print(f"task list: {result.task_list_path}")
    print(f"backup: {result.backup_dir}")
    print(f"written: {result.written_count}/{len(result.tasks)}")
    for task in result.tasks:
        status = "wrote" if task.written else "kept"
        print(f"  {status}: {task.path.name}")


def _print_collection_plan(plan) -> None:
    print(f"{plan.scenario}: {plan.transition_count} transitions")
    print(f"mode: {plan.split}")
    print(f"demos per transition: {plan.demos_per_transition}")
    print(f"total planned demos: {plan.total_demos}")
    print(f"output root: {plan.output_root}")


def _print_collection_result(index: int, total: int, result) -> None:
    print(_collection_result_line(index, total, result))


def _collection_result_line(index: int, total: int, result) -> str:
    job = result.job
    line = (
        f"[{index}/{total}] {job.from_state} --{job.operation}-> {job.to_state} "
        f"seed={result.seed} reward={result.total_reward:.3f} saved={result.saved}"
    )
    if result.failure_reason:
        line += f" reason={result.failure_reason}"
    return line


def _missing_collection_jobs(check) -> tuple:
    return tuple(
        replace(item.job, demo_index=item.saved + offset)
        for item in check.missing_demos
        for offset in range(item.expected - item.saved)
    )


def _print_dataset_check(check) -> None:
    print(f"{check.plan.scenario} {check.plan.split}")
    print(f"expected transition demos: {check.plan.total_demos}")
    print(f"saved transition dirs: {check.saved_transition_dirs}")
    print(f"saved episodes: {check.saved_episodes}")
    _print_missing_demos(check.missing_demos)
    _print_status("episode files", check.missing_episode_files)
    _print_scene_metadata_status(check)


def _print_missing_demos(missing) -> None:
    print(f"missing demos: {'missing' if missing else 'ok'}")
    for item in missing:
        job = item.job
        print(
            f"  {job.from_state} --{job.operation}-> {job.to_state} "
            f"saved {item.saved}/{item.expected}"
        )


def _print_scene_metadata_status(check) -> None:
    missing = check.missing_scene_files + check.scene_mismatches
    print(f"scene metadata: {'missing' if missing else 'ok'}")
    for path in check.missing_scene_files:
        print(f"  {path}")
    for item in check.scene_mismatches:
        print(f"  {item.path}: {item.key} expected {item.expected}, got {item.actual}")


def _print_operation_training(training) -> None:
    print(f"{training.scenario}")
    print(f"operation: {training.task_name}")
    print(f"train data: {training.train_dataset.path}")
    print(f"train episodes: {training.train_dataset.episodes}")
    print(f"train demos: {training.n_demos}")
    print(f"val data: {training.val_dataset.path}")
    print(f"val episodes: {training.val_dataset.episodes}")
    print(f"val demos: {training.n_val}")
    print(f"command: {format_command(training.command)}")


def _print_scenario_training(training) -> None:
    print(f"{training.scenario}")
    print(f"mode: {training.split}")
    print(f"val mode: {training.val_split}")
    print(f"operations: {training.operation_count}")
    print(f"train demos: {training.demo_count}")
    for operation in training.operations:
        print(
            f"  {operation.task_name}: "
            f"{operation.n_demos} train, "
            f"{operation.n_val} val"
        )


def _pending_training_operations(training, skip_existing: bool) -> tuple:
    if not skip_existing:
        return training.operations
    jobs = []
    for operation in training.operations:
        if operation.checkpoint_path.exists():
            print(f"skip existing: {operation.task_name} ({operation.checkpoint_path})")
        else:
            jobs.append(operation)
    return tuple(jobs)


def _print_operation_evaluation(evaluation) -> None:
    print(f"{evaluation.scenario}")
    print(f"operation: {evaluation.task_name}")
    print(f"mode: {evaluation.mode}")
    print(f"transitions: {len(evaluation.transitions)}")
    print(f"rollouts: {evaluation.rollout_count}")
    print(f"dataset: {evaluation.dataset.path}")
    print(f"checkpoint: {evaluation.checkpoint_path}")
    print(f"output: {evaluation.output_path}")


def _print_scenario_evaluation(evaluation) -> None:
    print(f"{evaluation.scenario}")
    print(f"mode: {evaluation.mode}")
    print(f"operations: {len(evaluation.operations)}")
    print(f"rollouts: {evaluation.rollout_count}")
    print(f"output: {evaluation.output_path}")
    for operation in evaluation.operations:
        print(
            f"  {operation.task_name}: "
            f"{len(operation.transitions)} transitions, "
            f"{operation.dataset.episodes} demos"
        )


def _pending_evaluations(evaluation, skip_existing: bool) -> tuple:
    if not skip_existing:
        return evaluation.operations
    jobs = []
    for operation in evaluation.operations:
        if operation.output_path.exists():
            print(f"skip existing: {operation.task_name} ({operation.output_path})")
        else:
            jobs.append(operation)
    return tuple(jobs)


def _run_pending_evaluations(evaluation, jobs: tuple, save: bool) -> None:
    for index, operation in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {operation.task_name}")
        result = run_operation_evaluation(operation, save=save)
        print(f"success: {result.success_count}/{len(result.episodes)}")
        print(f"success rate: {result.success_rate:.3f}")
        print(f"mean reward: {result.mean_reward:.3f}")
    if save:
        summary = summarize_scenario_results(evaluation)
        print(f"success: {summary['success_count']}/{summary['rollouts']}")
        print(f"success rate: {summary['success_rate']:.3f}")
        print(f"mean reward: {summary['mean_reward']:.3f}")
        print(f"saved: {evaluation.output_path}")


def _check_evaluation_checkpoints(jobs: tuple) -> None:
    missing = [
        path
        for operation in jobs
        for path in (operation.checkpoint_path, operation.train_config_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("\n".join(str(path) for path in missing))


def _print_scenario_status(status) -> None:
    transition = status.transition_check
    print(f"{status.scenario} {status.split}")
    print(
        "transition demos: "
        f"{'ok' if transition.ok else 'missing'} "
        f"({transition.saved_transition_dirs}/{transition.plan.transition_count} transitions, "
        f"{transition.saved_episodes} episodes)"
    )
    print(f"merged operation data: {status.merged_count}/{status.operation_count}")
    print(f"trained checkpoints: {status.trained_count}/{status.operation_count}")
    print(f"evaluation results: {status.evaluated_count}/{status.operation_count}")
    print(f"scenario summary: {'ok' if status.summary_exists else 'missing'}")
    _print_missing_status_items(
        "missing merged operation data",
        tuple(op for op in status.operations if not op.merged_ok),
        lambda op: f"{op.task_name} ({op.merged_episodes} episodes)",
    )
    _print_missing_status_items(
        "missing checkpoints",
        tuple(op for op in status.operations if not op.trained),
        lambda op: f"{op.task_name}: {op.checkpoint_path}",
    )
    _print_missing_status_items(
        "missing evaluation results",
        tuple(op for op in status.operations if not op.evaluated),
        lambda op: f"{op.task_name}: {op.eval_path}",
    )
    if not status.summary_exists:
        print(f"missing scenario summary: {status.summary_path}")


def _print_missing_status_items(label: str, items: tuple, line) -> None:
    if items:
        print(f"{label}:")
        for item in items:
            print(f"  {line(item)}")


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


if __name__ == "__main__":
    main()

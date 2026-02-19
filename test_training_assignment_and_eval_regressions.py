#!/usr/bin/env python3
"""Regression tests for training-side assignment and evaluation behavior."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.append(str(SRC))

from trainers.base_trainer import TrainerLoggingMixin  # noqa: E402


def _get_function_default(
    file_path: Path,
    class_name: str,
    function_name: str,
    arg_name: str,
):
    tree = ast.parse(file_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == function_name:
                    args = item.args.args
                    defaults = item.args.defaults
                    default_start = len(args) - len(defaults)
                    for idx, arg in enumerate(args):
                        if arg.arg == arg_name and idx >= default_start:
                            default_node = defaults[idx - default_start]
                            return ast.literal_eval(default_node)
    raise AssertionError(
        f"Could not find default value for {class_name}.{function_name}({arg_name})"
    )


def _get_add_argument_default(file_path: Path, flag_name: str):
    tree = ast.parse(file_path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        if not node.args:
            continue

        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and first_arg.value == flag_name:
            for kw in node.keywords:
                if kw.arg == "default":
                    return ast.literal_eval(kw.value)
    raise AssertionError(f"Could not find add_argument default for {flag_name}")


def test_trainer_default_eval_cadence() -> None:
    assert (
        _get_function_default(
            SRC / "MAPPO" / "trainer.py",
            "MAPPOTrainer",
            "__init__",
            "eval_freq_episodes",
        )
        == 5
    )
    assert (
        _get_function_default(
            SRC / "MAPPO" / "trainer.py",
            "MAPPOTrainer",
            "__init__",
            "eval_episodes",
        )
        == 1
    )
    assert (
        _get_function_default(
            SRC / "MAPPO" / "online_trainer.py",
            "MAPPOOnlineTrainer",
            "__init__",
            "eval_freq_episodes",
        )
        == 5
    )
    assert (
        _get_function_default(
            SRC / "QMIX" / "trainer.py",
            "QMIXTrainer",
            "__init__",
            "eval_freq_episodes",
        )
        == 5
    )


def test_train_cli_eval_flags_defaults() -> None:
    main_file = SRC / "main.py"
    assert _get_add_argument_default(main_file, "--train-eval-freq") == 5
    assert _get_add_argument_default(main_file, "--train-eval-episodes") == 1


def test_cooperative_reward_helper_reads_single_scalar() -> None:
    rewards = {0: -3.5, 1: -3.5, 2: -3.5}
    assert TrainerLoggingMixin.get_cooperative_step_reward(rewards) == -3.5
    assert TrainerLoggingMixin.get_cooperative_step_reward({}) == 0.0


def test_assignment_info_helper_reads_metadata() -> None:
    infos = {
        0: {
            "assigned_agent_id": 2,
            "assigned_case_id": 17,
            "assigned_task_id": 4,
        },
        1: {
            "assigned_agent_id": 2,
            "assigned_case_id": 17,
            "assigned_task_id": 4,
        },
    }
    assert TrainerLoggingMixin.get_assigned_agent_id(infos) == 2
    assert TrainerLoggingMixin.get_assigned_agent_id({0: {"assigned_agent_id": None}}) is None


def test_random_agent_has_reproducible_seed_path_and_random_sampling() -> None:
    baselines_file = SRC / "baselines.py"
    tree = ast.parse(baselines_file.read_text())

    random_class = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "RandomAgent":
            random_class = node
            break
    assert random_class is not None

    set_seed_method = None
    select_actions_method = None
    for item in random_class.body:
        if isinstance(item, ast.FunctionDef) and item.name == "set_seed":
            set_seed_method = item
        if isinstance(item, ast.FunctionDef) and item.name == "select_actions":
            select_actions_method = item

    assert set_seed_method is not None, "RandomAgent should expose seed reset"
    assert (
        select_actions_method is not None
    ), "RandomAgent should expose random action sampling"

    # Enforce strict-random semantics via RNG sampling call.
    select_calls = [n for n in ast.walk(select_actions_method) if isinstance(n, ast.Call)]
    has_randint = any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "randint"
        for call in select_calls
    )
    assert has_randint, "RandomAgent.select_actions should sample with RNG"


def main() -> None:
    test_trainer_default_eval_cadence()
    test_train_cli_eval_flags_defaults()
    test_cooperative_reward_helper_reads_single_scalar()
    test_assignment_info_helper_reads_metadata()
    test_random_agent_has_reproducible_seed_path_and_random_sampling()
    print("All training assignment/eval regression tests passed.")


if __name__ == "__main__":
    main()

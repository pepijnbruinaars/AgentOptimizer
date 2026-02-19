#!/usr/bin/env python3
"""Unit tests for environment parallelism, assignment, and logging behavior."""

from __future__ import annotations

import importlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.append(str(SRC))

try:
    from CustomEnvironment.custom_environment.env.custom_environment import (  # noqa: E402
        AgentOptimizerEnvironment,
        SimulationParameters,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - dependency-gated path
    AgentOptimizerEnvironment = None
    SimulationParameters = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class _ConstantDurationDistribution:
    """Simple deterministic duration distribution for tests."""

    def __init__(self, duration_seconds: float) -> None:
        self.duration_seconds = float(duration_seconds)

    def generate_sample(self, size: int) -> np.ndarray:
        return np.full(size, self.duration_seconds, dtype=float)


def _require_environment_dependencies() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "Environment dependencies are missing. Install requirements first."
        ) from _IMPORT_ERROR


def _build_pre_fitted_distributions(
    data: pd.DataFrame,
    default_duration_seconds: float = 120.0,
    incapable_pairs: set[tuple[str, str]] | None = None,
):
    incapable_pairs = incapable_pairs or set()

    resources = sorted(set(data["resource"]))
    activities = sorted(set(data["activity_name"]))

    activity_durations: dict[str, dict[str, _ConstantDurationDistribution | None]] = {}
    stats_dict: dict[str, dict[str, dict[str, float]]] = {}
    global_activity_medians: dict[str, float] = {
        activity: float(default_duration_seconds) for activity in activities
    }

    for resource in resources:
        resource_activity_durations: dict[str, _ConstantDurationDistribution | None] = {}
        resource_stats: dict[str, dict[str, float]] = {}
        for activity in activities:
            if (resource, activity) in incapable_pairs:
                resource_activity_durations[activity] = None
                resource_stats[activity] = {"mean": -1.0, "median": -1.0, "std": -1.0}
            else:
                resource_activity_durations[activity] = _ConstantDurationDistribution(
                    default_duration_seconds
                )
                resource_stats[activity] = {
                    "mean": float(default_duration_seconds),
                    "median": float(default_duration_seconds),
                    "std": 0.0,
                }
        activity_durations[resource] = resource_activity_durations
        stats_dict[resource] = resource_stats

    return activity_durations, stats_dict, global_activity_medians


def _make_env(
    data: pd.DataFrame,
    tmp_path: Path,
    default_duration_seconds: float = 120.0,
    incapable_pairs: set[tuple[str, str]] | None = None,
    start_timestamp: pd.Timestamp | None = None,
) -> AgentOptimizerEnvironment:
    pre_fitted_distributions = _build_pre_fitted_distributions(
        data=data,
        default_duration_seconds=default_duration_seconds,
        incapable_pairs=incapable_pairs,
    )
    start = start_timestamp if start_timestamp is not None else data["start_timestamp"].min()
    return AgentOptimizerEnvironment(
        data=data,
        simulation_parameters=SimulationParameters({"start_timestamp": start}),
        experiment_dir=str(tmp_path),
        pre_fitted_distributions=pre_fitted_distributions,
    )


def _run_until_done(
    env: AgentOptimizerEnvironment, observations: dict[int, dict], max_steps: int = 200
) -> None:
    for _ in range(max_steps):
        actions = {agent_id: 1 for agent_id in observations.keys()}
        observations, _, terminations, truncations, _ = env.step(actions)
        if any(terminations.values()) or any(truncations.values()):
            return
    raise AssertionError("Environment did not terminate within max_steps")


def _single_step_info(infos: dict[int, dict]) -> dict:
    return infos[next(iter(infos))]


def test_parallel_assignments_start_cases_at_the_same_timestamp(tmp_path: Path) -> None:
    _require_environment_dependencies()
    t0 = pd.Timestamp("2026-02-01 09:00:00")

    data = pd.DataFrame(
        [
            {
                "case_id": 1,
                "activity_name": "A",
                "resource": "r0",
                "start_timestamp": t0,
                "end_timestamp": t0 + pd.Timedelta(seconds=120),
            },
            {
                "case_id": 2,
                "activity_name": "A",
                "resource": "r1",
                "start_timestamp": t0,
                "end_timestamp": t0 + pd.Timedelta(seconds=120),
            },
        ]
    )

    env = _make_env(data=data, tmp_path=tmp_path, default_duration_seconds=120.0)

    observations, _ = env.reset()
    all_volunteer = {agent_id: 1 for agent_id in observations.keys()}
    observations, _, _, _, _ = env.step(all_volunteer)
    assert env.current_time == t0
    assert env.upcoming_case is not None

    first_agent_id, second_agent_id = sorted(observations.keys())

    actions_case_1 = {agent_id: 0 for agent_id in observations.keys()}
    actions_case_1[first_agent_id] = 1
    observations, _, _, _, infos = env.step(actions_case_1)
    first_step_info = _single_step_info(infos)
    assert first_step_info["assigned_agent_id"] == first_agent_id
    assert first_step_info["assigned_case_id"] in {1, 2}
    assert env.current_time == t0

    actions_case_2 = {agent_id: 0 for agent_id in observations.keys()}
    actions_case_2[second_agent_id] = 1
    observations, _, _, _, infos = env.step(actions_case_2)
    second_step_info = _single_step_info(infos)
    assert second_step_info["assigned_agent_id"] == second_agent_id

    active_case_ids = sorted(
        agent.current_case.id
        for agent in env.agents
        if agent.current_case is not None
    )
    assert active_case_ids == [1, 2]

    _run_until_done(env, observations)
    log = pd.read_csv(env.log_file)

    assert len(log) == 2
    assigned_times = pd.to_datetime(log["task_assigned_time"])
    started_times = pd.to_datetime(log["task_started_time"])
    assert (assigned_times == t0).all()
    assert (started_times == t0).all()


def test_assignment_falls_back_to_capable_agent_when_no_one_volunteers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_environment_dependencies()
    t0 = pd.Timestamp("2026-02-01 10:00:00")

    data = pd.DataFrame(
        [
            {
                "case_id": 1,
                "activity_name": "A",
                "resource": "r0",
                "start_timestamp": t0,
                "end_timestamp": t0 + pd.Timedelta(seconds=120),
            },
            {
                "case_id": 2,
                "activity_name": "A",
                "resource": "r1",
                "start_timestamp": t0 + pd.Timedelta(hours=1),
                "end_timestamp": t0 + pd.Timedelta(hours=1, seconds=120),
            },
        ]
    )

    env = _make_env(data=data, tmp_path=tmp_path, default_duration_seconds=120.0)
    observations, _ = env.reset()
    all_zero = {agent_id: 0 for agent_id in observations.keys()}

    observations, _, _, _, _ = env.step(all_zero)
    assert env.upcoming_case is not None

    selected_agent_id = max(observations.keys())
    monkeypatch.setattr(np.random, "choice", lambda candidates: candidates[-1])

    _, _, _, _, infos = env.step(all_zero)
    step_info = _single_step_info(infos)

    assert step_info["assigned_agent_id"] == selected_agent_id
    assert step_info["assigned_case_id"] == 1
    assert env.agents[selected_agent_id].current_case is not None
    assert env.agents[selected_agent_id].current_case.id == 1


def test_assignment_raises_when_no_capable_agent_exists(tmp_path: Path) -> None:
    _require_environment_dependencies()
    t0 = pd.Timestamp("2026-02-01 11:00:00")

    data = pd.DataFrame(
        [
            {
                "case_id": 1,
                "activity_name": "A",
                "resource": "r0",
                "start_timestamp": t0,
                "end_timestamp": t0 + pd.Timedelta(seconds=120),
            }
        ]
    )

    env = _make_env(
        data=data,
        tmp_path=tmp_path,
        default_duration_seconds=120.0,
        incapable_pairs={("r0", "A")},
    )
    observations, _ = env.reset()
    all_zero = {agent_id: 0 for agent_id in observations.keys()}
    env.step(all_zero)

    with pytest.raises(ValueError, match="No capable agents found"):
        env.step(all_zero)


def test_flush_csv_buffer_writes_once_and_preserves_single_header(tmp_path: Path) -> None:
    _require_environment_dependencies()
    t0 = pd.Timestamp("2026-02-01 12:00:00")

    data = pd.DataFrame(
        [
            {
                "case_id": 1,
                "activity_name": "A",
                "resource": "r0",
                "start_timestamp": t0,
                "end_timestamp": t0 + pd.Timedelta(seconds=120),
            }
        ]
    )

    env = _make_env(data=data, tmp_path=tmp_path, default_duration_seconds=120.0)
    log_path = Path(env.log_file)

    env.csv_buffer.append(
        [1, 1, t0, t0 + pd.Timedelta(seconds=120), 0, t0, t0, t0 + pd.Timedelta(seconds=120), 0]
    )
    env._flush_csv_buffer(force=False)
    assert not log_path.exists(), "Buffer should not flush before threshold without force"

    env._flush_csv_buffer(force=True)
    assert log_path.exists()
    assert env.csv_header_written is True
    assert env.csv_buffer == []

    env.csv_buffer.append(
        [
            2,
            1,
            t0 + pd.Timedelta(minutes=5),
            t0 + pd.Timedelta(minutes=7),
            0,
            t0 + pd.Timedelta(minutes=5),
            t0 + pd.Timedelta(minutes=5),
            t0 + pd.Timedelta(minutes=7),
            0,
        ]
    )
    env._flush_csv_buffer(force=True)

    lines = log_path.read_text().splitlines()
    assert sum(1 for line in lines if line.startswith("case_id,")) == 1

    log = pd.read_csv(log_path)
    assert log["case_id"].tolist() == [1, 2]


def test_next_time_uses_immediate_work_then_events_and_fallback(tmp_path: Path) -> None:
    _require_environment_dependencies()
    t0 = pd.Timestamp("2026-02-01 13:00:00")

    data = pd.DataFrame(
        [
            {
                "case_id": 1,
                "activity_name": "A",
                "resource": "r0",
                "start_timestamp": t0 + pd.Timedelta(minutes=10),
                "end_timestamp": t0 + pd.Timedelta(minutes=12),
            }
        ]
    )

    env = _make_env(
        data=data,
        tmp_path=tmp_path,
        default_duration_seconds=120.0,
        start_timestamp=t0,
    )

    assert env._get_next_time() == t0 + pd.Timedelta(minutes=10)

    env.agents[0].busy_until = t0 + pd.Timedelta(minutes=5)
    assert env._get_next_time() == t0 + pd.Timedelta(minutes=5)

    env.agents[0].busy_until = None
    env.future_cases = []
    assert env._get_next_time() == t0 + pd.Timedelta(seconds=1)


def test_trainer_logging_mixin_routes_metrics_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer_module = importlib.import_module("trainers.base_trainer")

    class FakeProgressManager:
        def __init__(self, total_episodes: int, disable: bool = False) -> None:
            self.total_episodes = total_episodes
            self.disable = disable
            self.training_started = 0
            self.training_finished = 0
            self.started_episodes: list[tuple[int, int | None]] = []
            self.timestep_updates: list[tuple[float | None, float | None]] = []
            self.episode_updates: list[dict[str, float | None]] = []
            self.episodes_finished = 0

        def start_training(self) -> None:
            self.training_started += 1

        def start_episode(self, episode_num: int, max_steps: int | None = None) -> None:
            self.started_episodes.append((episode_num, max_steps))

        def update_timestep(
            self, step_reward: float | None = None, cumulative_reward: float | None = None
        ) -> None:
            self.timestep_updates.append((step_reward, cumulative_reward))

        def update_episode(self, metrics: dict[str, float | None]) -> None:
            self.episode_updates.append(metrics)

        def finish_episode(self) -> None:
            self.episodes_finished += 1

        def finish_training(self) -> None:
            self.training_finished += 1

    class FakeTensorBoardLogger:
        def __init__(self, log_dir: str, enabled: bool = True) -> None:
            self.log_dir = log_dir
            self.enabled = enabled
            self.episode_calls: list[tuple[int, dict]] = []
            self.training_calls: list[tuple[int, dict]] = []
            self.eval_calls: list[tuple[int, dict]] = []
            self.closed = 0

        def log_episode_metrics(self, episode: int, metrics: dict) -> None:
            self.episode_calls.append((episode, metrics))

        def log_training_metrics(self, step: int, metrics: dict) -> None:
            self.training_calls.append((step, metrics))

        def log_evaluation_metrics(self, episode: int, metrics: dict) -> None:
            self.eval_calls.append((episode, metrics))

        def close(self) -> None:
            self.closed += 1

    monkeypatch.setattr(trainer_module, "ProgressBarManager", FakeProgressManager)
    monkeypatch.setattr(trainer_module, "TensorBoardLogger", FakeTensorBoardLogger)

    class DummyTrainer(trainer_module.TrainerLoggingMixin):
        def __init__(self) -> None:
            self.total_training_episodes = 7

    trainer = DummyTrainer()
    trainer.setup_logging(str(tmp_path), enable_tensorboard=False, disable_progress=True)

    assert trainer.progress_manager.total_episodes == 7
    assert trainer.progress_manager.disable is True
    assert trainer.progress_manager.training_started == 1
    assert trainer.tb_logger.log_dir == os.path.join(str(tmp_path), "tensorboard")
    assert trainer.tb_logger.enabled is False

    trainer.log_episode_start(3, max_steps=25)
    trainer.log_timestep(4, step_reward=1.5, cumulative_reward=5.0)

    metrics = {
        "reward": 6.0,
        "length": 12,
        "avg_reward": 4.5,
        "avg_length": 9.0,
        "cumulative_reward": 18.0,
        "time": 1.2,
        "actor_loss": 0.1,
        "critic_loss": 0.2,
        "loss": 0.3,
        "epsilon": 0.4,
        "buffer_size": 5,
        "target_updates": 6,
        "ignored_metric": 999,
    }
    trainer.log_episode_end(3, metrics)
    trainer.log_evaluation(3, {"reward": 7.5, "cumulative_reward": 21.0})
    trainer.cleanup_logging()

    assert trainer.progress_manager.started_episodes == [(3, 25)]
    assert trainer.progress_manager.timestep_updates == [(1.5, 5.0)]
    assert trainer.progress_manager.episode_updates == [
        {"avg_reward": 4.5, "avg_length": 9.0, "time_elapsed": 1.2}
    ]
    assert trainer.progress_manager.episodes_finished == 1
    assert trainer.progress_manager.training_finished == 1

    assert trainer.tb_logger.episode_calls == [
        (
            3,
            {
                "reward": 6.0,
                "length": 12,
                "avg_reward": 4.5,
                "avg_length": 9.0,
                "cumulative_reward": 18.0,
                "time": 1.2,
            },
        )
    ]
    assert trainer.tb_logger.training_calls == [
        (
            3,
            {
                "actor_loss": 0.1,
                "critic_loss": 0.2,
                "loss": 0.3,
                "epsilon": 0.4,
                "buffer_size": 5,
                "target_updates": 6,
            },
        )
    ]
    assert trainer.tb_logger.eval_calls == [(3, {"reward": 7.5, "cumulative_reward": 21.0})]
    assert trainer.tb_logger.closed == 1


def test_trainer_logging_helpers_handle_edge_payloads() -> None:
    trainer_module = importlib.import_module("trainers.base_trainer")
    mixin = trainer_module.TrainerLoggingMixin

    assert mixin.get_cooperative_step_reward({}) == 0.0
    assert mixin.get_cooperative_step_reward({0: -2.25, 1: -2.25}) == -2.25

    assert mixin.get_assigned_agent_id({}) is None
    assert mixin.get_assigned_agent_id({0: "invalid"}) is None
    assert mixin.get_assigned_agent_id({0: {"assigned_agent_id": None}}) is None
    assert mixin.get_assigned_agent_id({0: {"assigned_agent_id": "4"}}) == 4


def test_qmix_replay_ring_buffer_keeps_fixed_size_latest_transitions() -> None:
    from QMIX.trainer import QMIXTrainer

    trainer = object.__new__(QMIXTrainer)
    trainer.buffer_size = 3
    trainer.buffer = []
    trainer.buffer_pos = 0

    for value in range(5):
        trainer._append_transition((value,))

    assert len(trainer.buffer) == 3
    assert {item[0] for item in trainer.buffer} == {2, 3, 4}
    assert trainer.buffer_pos == 2


def test_qmix_terminal_replay_transition_is_shape_safe_for_learning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_environment_dependencies()
    from QMIX.agent import QMIXAgent
    from QMIX.trainer import QMIXTrainer

    t0 = pd.Timestamp("2026-02-01 14:00:00")
    data = pd.DataFrame(
        [
            {
                "case_id": 1,
                "activity_name": "A",
                "resource": "r0",
                "start_timestamp": t0,
                "end_timestamp": t0 + pd.Timedelta(seconds=1),
            }
        ]
    )

    env = _make_env(data=data, tmp_path=tmp_path, default_duration_seconds=1.0)
    agent = QMIXAgent(env, device="cpu")
    trainer = QMIXTrainer(
        env=env,
        agent=agent,
        total_training_episodes=1,
        batch_size=1,
        buffer_size=16,
        target_update_interval=1000,
        eval_freq_episodes=1000,
        save_freq_episodes=1000,
        log_freq_episodes=1000,
        should_eval=False,
        experiment_dir=str(tmp_path / "qmix_replay_regression"),
        enable_tensorboard=False,
        disable_progress=True,
    )

    trainer.train()

    terminal_transitions = [transition for transition in trainer.buffer if transition[4]]
    assert terminal_transitions, "Expected at least one terminal transition in replay"

    terminal_transition = terminal_transitions[0]
    obs = terminal_transition[0]
    next_obs = terminal_transition[3]
    assert next_obs is not None, "Terminal transition next_obs should be shape-compatible"

    if isinstance(obs, np.ndarray):
        assert isinstance(next_obs, np.ndarray)
        assert next_obs.shape == obs.shape
        assert np.array_equal(next_obs, obs)
    else:
        assert set(next_obs.keys()) == set(obs.keys())
        for agent_id in obs:
            assert set(next_obs[agent_id].keys()) == set(obs[agent_id].keys())
            for key in obs[agent_id]:
                assert np.array_equal(
                    np.asarray(next_obs[agent_id][key]),
                    np.asarray(obs[agent_id][key]),
                )

    monkeypatch.setattr(random, "sample", lambda seq, k: [terminal_transition])
    loss = trainer.learn()
    assert loss is not None

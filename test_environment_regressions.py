#!/usr/bin/env python3
"""
Regression tests for environment scheduling and timestamp semantics.
"""

import os
import sys
import tempfile

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

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


def _require_environment_dependencies() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "Environment dependencies are missing. Install requirements first."
        ) from _IMPORT_ERROR


def _run_until_done(env: AgentOptimizerEnvironment, max_steps: int = 2000) -> None:
    """Step environment with all agents volunteering until termination."""
    observations, _ = env.reset()
    for _ in range(max_steps):
        actions = {agent_id: 1 for agent_id in observations.keys()}
        observations, _, terminations, truncations, _ = env.step(actions)
        if any(terminations.values()) or any(truncations.values()):
            return
    raise AssertionError("Environment did not terminate within max_steps")


def test_case_open_time_is_not_overwritten_by_reassignment() -> None:
    """
    `case_open_time` in logs must remain the case's initial open timestamp,
    even when later tasks are assigned at newer timestamps.
    """
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
                "case_id": 1,
                "activity_name": "B",
                "resource": "r0",
                "start_timestamp": t0 + pd.Timedelta(seconds=600),
                "end_timestamp": t0 + pd.Timedelta(seconds=720),
            },
        ]
    )

    _require_environment_dependencies()

    with tempfile.TemporaryDirectory(prefix="case_open_time_regression_") as tmp_dir:
        env = AgentOptimizerEnvironment(
            data=data,
            simulation_parameters=SimulationParameters(
                {"start_timestamp": data["start_timestamp"].min()}
            ),
            experiment_dir=tmp_dir,
        )

        _run_until_done(env)

        log = pd.read_csv(env.log_file)
        assert not log.empty, "Expected completed-case log rows"

        logged_open_times = pd.to_datetime(log["case_open_time"])
        assert (logged_open_times == t0).all(), (
            "case_open_time should be the initial case open timestamp for all tasks"
        )


def test_same_timestamp_arrivals_are_assignable_without_time_advance() -> None:
    """
    If another case has already arrived at current_time, the simulation should not
    advance time before allowing that assignment. This enables parallel starts.
    """
    t0 = pd.Timestamp("2026-02-01 10:00:00")

    data = pd.DataFrame(
        [
            {
                "case_id": 1,
                "activity_name": "A",
                "resource": "r0",
                "start_timestamp": t0,
                "end_timestamp": t0 + pd.Timedelta(seconds=300),
            },
            {
                "case_id": 2,
                "activity_name": "A",
                "resource": "r1",
                "start_timestamp": t0,
                "end_timestamp": t0 + pd.Timedelta(seconds=300),
            },
        ]
    )

    _require_environment_dependencies()

    with tempfile.TemporaryDirectory(prefix="parallel_time_regression_") as tmp_dir:
        env = AgentOptimizerEnvironment(
            data=data,
            simulation_parameters=SimulationParameters(
                {"start_timestamp": data["start_timestamp"].min()}
            ),
            experiment_dir=tmp_dir,
        )

        observations, _ = env.reset()
        all_actions = {agent_id: 1 for agent_id in observations.keys()}
        sorted_agent_ids = sorted(observations.keys())
        assert len(sorted_agent_ids) == 2
        first_agent_id, second_agent_id = sorted_agent_ids

        # First step should discover immediate work and stay at t0.
        observations, _, _, _, _ = env.step(all_actions)
        assert env.current_time == t0
        assert env.upcoming_case is not None

        # Choose one specific agent for first assignment.
        actions_case_1 = {agent_id: 0 for agent_id in observations.keys()}
        actions_case_1[first_agent_id] = 1

        # Assign first case; second case already arrived at t0, so still no time advance.
        observations, _, _, _, _ = env.step(actions_case_1)
        assert env.current_time == t0
        assert env.upcoming_case is not None

        # Choose the other agent for second assignment.
        actions_case_2 = {agent_id: 0 for agent_id in observations.keys()}
        actions_case_2[second_agent_id] = 1

        # Assign second case at the same timestamp; both agents should now work in parallel.
        observations, _, _, _, _ = env.step(actions_case_2)
        active_case_ids = [
            agent.current_case.id
            for agent in env.agents
            if agent.current_case is not None
        ]
        assert len(active_case_ids) == 2
        assert len(set(active_case_ids)) == 2


def main() -> None:
    """Run regressions as a plain Python test script."""
    test_case_open_time_is_not_overwritten_by_reassignment()
    test_same_timestamp_arrivals_are_assignable_without_time_advance()
    print("All environment regression tests passed.")


if __name__ == "__main__":
    main()

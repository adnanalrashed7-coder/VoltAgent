import csv

import numpy as np

from voltagent.data_logging import FIELDS, log_simulated_experience
from voltagent.simulation_env import GeneratorInverterEnv, SimulationConfig


def test_reset_returns_valid_observation_and_repeats_with_seed():
    env = GeneratorInverterEnv()
    obs1, info1 = env.reset(seed=123)
    obs2, info2 = env.reset(seed=123)
    assert obs1.shape == (7,)
    assert env.observation_space.contains(obs1)
    assert np.allclose(obs1, obs2)
    assert info1["simulation_only"] is True
    assert info2["simulation_only"] is True
    env.close()


def test_step_is_finite_and_limits_applied_current_delta():
    env = GeneratorInverterEnv()
    obs, _ = env.reset(seed=10)
    next_obs, reward, terminated, truncated, info = env.step(np.array([5.0], dtype=np.float32))
    assert env.observation_space.contains(next_obs)
    assert np.isfinite(reward)
    assert abs(info["action_applied_delta_a"]) <= env.config.max_delta_current_a_per_step + 1e-6
    assert abs(float(next_obs[2]) - float(obs[2])) <= env.config.max_delta_current_a_per_step + 1e-6
    assert info["simulation_only"] is True
    assert not (terminated and truncated)
    env.close()


def test_research_guard_forces_current_reduction_when_frequency_is_low():
    env = GeneratorInverterEnv()
    obs, _ = env.reset(seed=1, options={
        "frequency_hz": 48.8,
        "charge_current_a": 10.0,
        "target_charge_current_a": 20.0,
        "battery_temp_c": 25.0,
        "house_load_kw": 0.5,
    })
    _, _, _, _, info = env.step(np.array([5.0], dtype=np.float32))
    assert info["guard_active"] is True
    assert info["action_applied_delta_a"] < 0.0
    env.close()


def test_episode_truncates_at_configured_length():
    env = GeneratorInverterEnv(config=SimulationConfig(episode_steps=1))
    env.reset(seed=3, options={
        "frequency_hz": 50.0,
        "charge_current_a": 5.0,
        "target_charge_current_a": 5.0,
        "battery_temp_c": 25.0,
        "house_load_kw": 0.5,
    })
    _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
    assert truncated is True
    assert terminated is False
    env.close()


def test_simulation_logger_explicitly_labels_source(tmp_path):
    env = GeneratorInverterEnv()
    obs, _ = env.reset(seed=5)
    next_obs, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
    path = tmp_path / "simulation.csv"
    log_simulated_experience(
        path,
        run_id="unit-test-run",
        step=1,
        obs=obs,
        next_obs=next_obs,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
    )
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["source_type"] == "simulation"
    assert rows[0]["run_id"] == "unit-test-run"
    assert list(rows[0].keys()) == FIELDS
    env.close()

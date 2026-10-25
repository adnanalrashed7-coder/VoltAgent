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


def test_research_guard_sheds_battery_charging_when_frequency_is_low():
    env = GeneratorInverterEnv()
    obs, _ = env.reset(seed=1, options={
        "frequency_hz": 48.8,
        "battery_current_a": -1.0,
        "battery_soc": 0.6,
        "battery_temp_c": 25.0,
        "house_load_kw": 0.5,
    })
    _, _, _, _, info = env.step(np.array([-1.0], dtype=np.float32))
    assert info["guard_active"] is True
    assert info["action_applied_delta_a"] > 0.0
    env.close()


def test_episode_truncates_at_configured_length():
    env = GeneratorInverterEnv(config=SimulationConfig(episode_steps=1))
    env.reset(seed=3, options={
        "frequency_hz": 50.0,
        "battery_current_a": 0.0,
        "battery_soc": 0.6,
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


def test_emergency_charge_shed_is_exposed_at_top_level_info():
    env = GeneratorInverterEnv()
    try:
        env.reset(seed=7, options={
            "frequency_hz": 50.0,
            "battery_current_a": 0.0,
            "battery_soc": 0.6,
            "battery_temp_c": 25.0,
            "house_load_kw": 0.5,
        })

        _, _, _, _, info = env.step(
            np.array([0.0], dtype=np.float32)
        )

        assert "emergency_charge_shed" in info
        assert isinstance(info["emergency_charge_shed"], bool)
        assert "emergency_charge_shed" not in info["reward_components"]
    finally:
        env.close()

def test_emergency_overload_guard_sheds_charging_load():
    config = SimulationConfig(
        generator_rated_power_kw=4.0,
        dc_charge_voltage_v=50.0,
        inverter_efficiency=0.9,
        guard_sheds_charging=True,
    )
    env = GeneratorInverterEnv(config=config)

    try:
        env.reset(
            seed=11,
            options={
                "frequency_hz": 50.0,
                "battery_current_a": -20.0,
                "battery_soc": 0.6,
                "battery_temp_c": 25.0,
                "house_load_kw": 3.5,
            },
        )

        _, _, _, _, info = env.step(
            np.array([0.0], dtype=np.float32)
        )

        assert info["emergency_charge_shed"] is True
        assert env.battery_current_a == 0.0
        assert info["guard_active"] is True

    finally:
        env.close()       

def test_battery_heating_scales_with_squared_current():
    config = SimulationConfig(
        generator_rated_power_kw=4.0,
        battery_internal_resistance_ohm=0.02,
        battery_thermal_capacity_j_per_c=2000.0,
        battery_cooling_w_per_c=4.0,
        ambient_temp_c=25.0,
    )

    def run_one_step(current_a):
        env = GeneratorInverterEnv(config=config)

        try:
            env.reset(
                seed=17,
                options={
                    "frequency_hz": 50.0,
                    "battery_current_a": current_a,
                    "battery_soc": 0.6,
                    "battery_temp_c": 25.0,
                    "house_load_kw": 3.5,
                },
            )

            env.step(np.array([0.0], dtype=np.float32))

            assert env.battery_current_a == current_a
            return env.battery_temp_c - 25.0

        finally:
            env.close()

    rise_at_20a = run_one_step(20.0)
    rise_at_40a = run_one_step(40.0)

    assert np.isclose(rise_at_20a, 0.0008, atol=1e-9)
    assert np.isclose(rise_at_40a, 0.0032, atol=1e-9)
    assert np.isclose(rise_at_40a / rise_at_20a, 4.0)


def test_battery_support_reduces_generator_load_and_updates_soc():
    config = SimulationConfig(action_request_limit_a=0.5, max_delta_current_a_per_step=0.5)

    def run_step(action):
        env = GeneratorInverterEnv(config=config)
        try:
            env.reset(seed=31, options={
                "frequency_hz": 50.0,
                "battery_current_a": 0.0,
                "battery_soc": 0.6,
                "battery_temp_c": 25.0,
                "house_load_kw": 1.0,
            })
            return env.step(np.array([action], dtype=np.float32)), env
        except Exception:
            env.close()
            raise

    (support_result, support_env) = run_step(1.0)
    (charge_result, charge_env) = run_step(-1.0)
    try:
        support_info = support_result[4]
        charge_info = charge_result[4]
        assert support_info["battery_current_a"] > 0.0
        assert support_info["total_load_kw"] < 1.0
        assert support_info["battery_soc"] < 0.6
        assert charge_info["battery_current_a"] < 0.0
        assert charge_info["total_load_kw"] > 1.0
        assert charge_info["battery_soc"] > 0.6
    finally:
        support_env.close()
        charge_env.close()


def test_soc_bounds_block_battery_dispatch_beyond_reserve():
    config = SimulationConfig()
    for soc, action in (
        (config.battery_soc_min, 1.0),
        (config.battery_soc_max, -1.0),
    ):
        env = GeneratorInverterEnv(config=config)
        try:
            env.reset(seed=32, options={
                "frequency_hz": 50.0,
                "battery_current_a": 0.0,
                "battery_soc": soc,
                "battery_temp_c": 25.0,
                "house_load_kw": 3.5,
            })
            env.step(np.array([action], dtype=np.float32))
            assert env.battery_current_a == 0.0
        finally:
            env.close()


def test_high_temperature_derates_battery_discharge_current():
    env = GeneratorInverterEnv()
    try:
        env.reset(seed=33, options={
            "frequency_hz": 50.0,
            "battery_current_a": 10.0,
            "battery_soc": 0.6,
            "battery_temp_c": 59.0,
            "house_load_kw": 3.5,
        })

        env.step(np.array([0.0], dtype=np.float32))

        assert 0.0 < env.battery_current_a < 10.0
        assert env.battery_current_a <= 50.0 / 15.0
    finally:
        env.close()
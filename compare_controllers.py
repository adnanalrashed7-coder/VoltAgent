"""Compare PPO with simple baselines on identical seeded simulator scenarios.

The metrics are simulator-specific research metrics, not real-world validation.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from voltagent import MODEL_SCHEMA_VERSION
from voltagent.simulation_env import GeneratorInverterEnv, SimulationConfig


def controller_action(name: str, obs: np.ndarray, model: PPO | None) -> np.ndarray:
    freq, df_dt, current, temp, _load, soc, _mechanical_power = map(float, obs)
    if name == "zero_battery":
        if current > 0.25:
            return np.array([-1.0], dtype=np.float32)
        if current < -0.25:
            return np.array([1.0], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)
    if name in ("frequency_aware", "thermal_aware"):
        if name == "thermal_aware" and (temp > 40.0 or soc < 0.30 or soc > 0.85):
            if current > 0.25:
                return np.array([-1.0], dtype=np.float32)
            if current < -0.25:
                return np.array([1.0], dtype=np.float32)
            return np.array([0.0], dtype=np.float32)
        if freq < 49.7 or df_dt < -0.5:
            return np.array([1.0], dtype=np.float32)
        if freq > 50.3 or df_dt > 0.5:
            return np.array([-1.0], dtype=np.float32)
        if current > 0.25:
            return np.array([-1.0], dtype=np.float32)
        if current < -0.25:
            return np.array([1.0], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)
    if name == "ppo":
        if model is None:
            raise ValueError("PPO policy requested but no model was loaded")
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)
    raise ValueError(f"Unknown controller: {name}")


def evaluate_controller(name: str, seeds: list[int], model: PPO | None, config: SimulationConfig) -> dict:
    all_freq_errors = []
    all_frequencies = []
    all_temps = []
    all_battery_currents = []
    all_battery_socs = []
    all_rewards = []
    all_guard_flags = []
    all_overloads = []
    episode_trips = 0
    episode_steps = []
    episode_final_socs = []

    env = GeneratorInverterEnv(config=config)
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated):
                action = controller_action(name, obs, model)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                all_frequencies.append(float(obs[0]))
                all_freq_errors.append(abs(float(obs[0]) - config.nominal_frequency_hz))
                all_temps.append(float(obs[3]))
                all_battery_currents.append(abs(float(obs[2])))
                all_battery_socs.append(float(obs[5]))
                all_rewards.append(float(reward))
                all_guard_flags.append(bool(info.get("guard_active", False)))
                all_overloads.append(float(info.get("overload_kw", 0.0)))
            episode_steps.append(steps)
            episode_final_socs.append(float(obs[5]))
            episode_trips += int(terminated)
    finally:
        env.close()

    freq_errors = np.asarray(all_freq_errors, dtype=float)
    frequencies = np.asarray(all_frequencies, dtype=float)
    temps = np.asarray(all_temps, dtype=float)
    return {
        "episodes": len(seeds),
        "total_steps": len(all_rewards),
        "mean_reward_per_step": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "mean_absolute_frequency_error_hz": float(np.mean(freq_errors)) if len(freq_errors) else 0.0,
        "p95_absolute_frequency_error_hz": float(np.percentile(freq_errors, 95)) if len(freq_errors) else 0.0,
        "max_absolute_frequency_error_hz": float(np.max(freq_errors)) if len(freq_errors) else 0.0,
        "minimum_frequency_hz": float(np.min(frequencies)) if len(frequencies) else 0.0,
        "maximum_frequency_hz": float(np.max(frequencies)) if len(frequencies) else 0.0,
        "maximum_battery_temperature_c": float(np.max(temps)) if len(temps) else 0.0,
        "mean_absolute_battery_current_a": float(np.mean(all_battery_currents)) if all_battery_currents else 0.0,
        "battery_throughput_ah": float(np.sum(all_battery_currents) * config.dt_s / 3600.0),
        "minimum_battery_soc": float(np.min(all_battery_socs)) if all_battery_socs else 0.0,
        "mean_final_battery_soc": float(np.mean(episode_final_socs)) if episode_final_socs else 0.0,
        "simulated_trip_count": episode_trips,
        "simulated_trip_rate": float(episode_trips / max(1, len(seeds))),
        "guard_activation_fraction": float(np.mean(all_guard_flags)) if all_guard_flags else 0.0,
        "overload_step_fraction": float(np.mean(np.asarray(all_overloads) > 1e-9)) if all_overloads else 0.0,
        "mean_episode_steps": float(np.mean(episode_steps)) if episode_steps else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline controllers against PPO on identical synthetic scenarios.")
    parser.add_argument("--model-path", default="inverter_agent_ppo", help="Trained model path with or without .zip.")
    parser.add_argument("--episodes", type=int, default=30, help="Number of seeded scenarios (default 30).")
    parser.add_argument("--seed", type=int, default=5000, help="First scenario seed.")
    parser.add_argument("--output", default="reports/controller_comparison.json", help="Output JSON report path.")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")

    base = Path(args.model_path)
    if base.suffix == ".zip":
        base = base.with_suffix("")
    model_file = Path(str(base) + ".zip")
    metadata_file = Path(str(base) + ".metadata.json")
    if not model_file.is_file() or not metadata_file.is_file():
        raise SystemExit("A current model and metadata are required. Run `python train_agent.py` first.")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("model_schema_version") != MODEL_SCHEMA_VERSION or metadata.get("hardware_validated") is not False:
        raise SystemExit("Model schema/metadata is not compatible with this research comparison.")
    try:
        config = SimulationConfig(**metadata["config"])
        config.validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Model metadata has an invalid simulator config: {exc}") from exc
    model = PPO.load(str(base))

    seeds = list(range(args.seed, args.seed + args.episodes))
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "simulation_only": True,
        "hardware_validated": False,
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in config.__dict__.items()},
        "scenario_seeds": seeds,
        "controllers": {},
        "note": "Research comparison on a simplified synthetic simulator; not proof of hardware safety or real-world superiority.",
    }
    for name, controller_model in (
        ("zero_battery", None),
        ("frequency_aware", None),
        ("thermal_aware", None),
        ("ppo", model),
    ):
        result = evaluate_controller(name, seeds, controller_model, config)
        report["controllers"][name] = result
        print(
            f"{name:18s} | mean |f-50|={result['mean_absolute_frequency_error_hz']:.3f} Hz "
            f"| p95={result['p95_absolute_frequency_error_hz']:.3f} Hz "
            f"| trips={result['simulated_trip_count']}/{args.episodes} "
            f"| max T={result['maximum_battery_temperature_c']:.2f} C "
            f"| mean |Ibat|={result['mean_absolute_battery_current_a']:.2f} A"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Comparison report: {output}")
    print("Results apply only to this simulation and its placeholder parameters.")


if __name__ == "__main__":
    main()

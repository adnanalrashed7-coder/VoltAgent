"""Train a new VoltAgent research model using synthetic simulation only.

This script intentionally does NOT fine-tune from CSV logs. In particular, it
will never treat previously logged simulation transitions as real-world data.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from voltagent import MODEL_SCHEMA_VERSION, __version__
from voltagent.simulation_env import GeneratorInverterEnv, SimulationConfig


def model_base_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.suffix == ".zip":
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_model(model: PPO, config: SimulationConfig, episodes: int, seed: int) -> dict:
    """Evaluate deterministically and report descriptive simulation metrics."""
    records = []
    env = GeneratorInverterEnv(config=config)
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            rewards = []
            frequencies = []
            temperatures = []
            currents = []
            guards = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(reward))
                frequencies.append(float(obs[0]))
                temperatures.append(float(obs[3]))
                currents.append(float(obs[2]))
                guards += int(info.get("guard_active", False))
            records.append({
                "episode": episode + 1,
                "steps": len(rewards),
                "total_reward": float(np.sum(rewards)),
                "mean_absolute_frequency_error_hz": float(np.mean(np.abs(np.asarray(frequencies) - config.nominal_frequency_hz))),
                "max_absolute_frequency_error_hz": float(np.max(np.abs(np.asarray(frequencies) - config.nominal_frequency_hz))),
                "max_battery_temperature_c": float(np.max(temperatures)),
                "final_charge_current_a": float(currents[-1]),
                "guard_activation_fraction": float(guards / max(1, len(rewards))),
                "terminated_by_simulated_trip": bool(terminated),
                "truncated_by_episode_length": bool(truncated),
            })
    finally:
        env.close()

    if not records:
        return {"episodes": 0, "results": []}
    return {
        "simulation_only": True,
        "episodes": episodes,
        "mean_total_reward": float(np.mean([r["total_reward"] for r in records])),
        "mean_absolute_frequency_error_hz": float(np.mean([r["mean_absolute_frequency_error_hz"] for r in records])),
        "worst_max_absolute_frequency_error_hz": float(np.max([r["max_absolute_frequency_error_hz"] for r in records])),
        "max_battery_temperature_c": float(np.max([r["max_battery_temperature_c"] for r in records])),
        "simulated_trip_count": int(sum(r["terminated_by_simulated_trip"] for r in records)),
        "results": records,
        "note": "Descriptive results from this simplified synthetic model; not evidence of real-world safety or performance.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VoltAgent PPO against a research-only simulator.")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Training timesteps (default: 50000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--model-path", default="inverter_agent_ppo", help="Output model path, with or without .zip.")
    parser.add_argument("--eval-episodes", type=int, default=8, help="Number of deterministic simulation evaluation episodes.")
    parser.add_argument("--device", default="auto", help="Stable-Baselines3 device, e.g. auto, cpu, cuda.")
    args = parser.parse_args()

    if args.timesteps < 1 or args.eval_episodes < 1:
        parser.error("--timesteps and --eval-episodes must be >= 1")

    base_path = model_base_path(args.model_path)
    config = SimulationConfig()
    env = GeneratorInverterEnv(config=config)

    # Fail early if the environment violates Gymnasium expectations.
    check_env(env, warn=True)
    print("[RESEARCH MODE] Training with synthetic simulation only.")
    print("[WARNING] Default electrical and thermal parameters are placeholders, not hardware specifications.")
    print(f"[INFO] Training for {args.timesteps:,} timesteps; random seed={args.seed}.")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(str(base_path))

    metadata = {
        "project": "VoltAgent",
        "project_version": __version__,
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_source": "synthetic_simulation_only",
        "hardware_validated": False,
        "observation_fields": [
            "frequency_hz", "df_dt_hz_s", "charge_current_a", "battery_temp_c",
            "house_load_kw", "target_charge_current_a", "mechanical_power_kw",
        ],
        "action": "normalized scalar [-1, 1], mapped to a request of +/- action_request_limit_a amperes per step; simulator rate-limits the applied change",
        "timesteps_requested": args.timesteps,
        "timesteps_completed": int(model.num_timesteps),
        "seed": args.seed,
        "config": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in config.__dict__.items()
        },
        "warnings": [
            "Research-only simplified simulator; not a validated digital twin.",
            "This model must not be used to control physical electrical equipment.",
            "Re-train and re-validate if the observation/action schema changes.",
        ],
    }
    metadata_path = Path(str(base_path) + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    report = evaluate_model(model, config, args.eval_episodes, args.seed + 10_000)
    report_path = Path("reports") / f"simulation_evaluation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] New research model saved: {base_path}.zip")
    print(f"[OK] Metadata saved: {metadata_path}")
    print(f"[OK] Simulation evaluation saved: {report_path}")
    print(f"[SUMMARY] Mean absolute frequency error: {report.get('mean_absolute_frequency_error_hz', float('nan')):.3f} Hz")
    print(f"[SUMMARY] Simulated trip episodes: {report.get('simulated_trip_count', 'n/a')}/{args.eval_episodes}")
    print("[LIMITATION] These metrics describe this simulator only; they do not demonstrate hardware safety.")
    env.close()


if __name__ == "__main__":
    main()

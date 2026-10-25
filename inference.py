"""Run a trained VoltAgent model in the synthetic simulator, never on hardware."""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from voltagent import MODEL_SCHEMA_VERSION
from voltagent.data_logging import log_simulated_experience
from voltagent.simulation_env import GeneratorInverterEnv


def _base_path(path_text: str) -> Path:
    path = Path(path_text)
    return path.with_suffix("") if path.suffix == ".zip" else path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VoltAgent inside the research simulator.")
    parser.add_argument("--model-path", default="inverter_agent_ppo", help="Model path, with or without .zip.")
    parser.add_argument("--steps", type=int, default=100, help="Maximum simulation steps.")
    parser.add_argument("--seed", type=int, default=7, help="Environment random seed.")
    parser.add_argument("--log-path", default="data/simulation_inference.csv", help="Simulation log CSV path.")
    parser.add_argument("--no-log", action="store_true", help="Do not write a CSV log.")
    parser.add_argument("--render", action="store_true", help="Print each simulator step.")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be >= 1")

    base_path = _base_path(args.model_path)
    zip_path = Path(str(base_path) + ".zip")
    metadata_path = Path(str(base_path) + ".metadata.json")
    if not zip_path.is_file() or not metadata_path.is_file():
        raise SystemExit(
            f"Model or metadata not found for '{base_path}'. Run `python train_agent.py` first. "
            "The bundled legacy model was trained with an incompatible, unvalidated schema."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_schema_version") != MODEL_SCHEMA_VERSION:
        raise SystemExit("Model schema mismatch. Re-train using the current train_agent.py.")
    if metadata.get("training_source") != "synthetic_simulation_only" or metadata.get("hardware_validated") is not False:
        raise SystemExit("The model metadata does not identify a supported simulation-only model.")

    print("[RESEARCH MODE] This run is a simulation, not a connection to a generator, inverter, or battery.")
    print("[WARNING] Simulator limits are illustrative and must not be treated as equipment protection limits.")
    model = PPO.load(str(base_path))
    env = GeneratorInverterEnv(render_mode="human" if args.render else None)
    obs, _ = env.reset(seed=args.seed)
    run_id = str(uuid.uuid4())
    total_reward = 0.0
    frequencies = []
    temps = []
    log_path = Path(args.log_path)

    print("step | frequency Hz | battery A | SOC % | temp C | house kW | guard")
    print("-" * 75)
    try:
        for step in range(1, args.steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            frequencies.append(float(next_obs[0]))
            temps.append(float(next_obs[3]))
            if not args.no_log:
                log_simulated_experience(
                    log_path,
                    run_id=run_id,
                    step=step,
                    obs=obs,
                    next_obs=next_obs,
                    reward=float(reward),
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
            print(
                f"{step:4d} | {next_obs[0]:12.3f} | {next_obs[2]:9.2f} | "
                f"{next_obs[5] * 100:5.1f} | {next_obs[3]:6.2f} | {next_obs[4]:8.2f} | "
                f"{str(info.get('guard_active', False)):>5s}"
            )
            obs = next_obs
            if terminated:
                print("[SIMULATED TRIP] A research episode stop condition was reached.")
                break
            if truncated:
                break
    finally:
        env.close()

    print("-" * 75)
    print(f"Simulation steps: {len(frequencies)}")
    print(f"Total reward: {total_reward:.3f}")
    if frequencies:
        print(
            "Mean absolute frequency error: "
            f"{np.mean(np.abs(np.asarray(frequencies) - env.config.nominal_frequency_hz)):.3f} Hz"
        )
        print(f"Maximum simulated battery temperature: {max(temps):.2f} C")
    if not args.no_log:
        print(f"Simulation-only log: {log_path}")
    print("No physical equipment was controlled. Results do not establish real-world safety.")


if __name__ == "__main__":
    main()

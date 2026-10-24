"""Research-only, simplified generator/inverter/battery simulation.

IMPORTANT:
    This is a teaching/research model, not a validated digital twin or protection
    relay. All default equipment-related values are placeholders, not ratings for
    any real generator, inverter, or battery. Do not use its output to control
    physical equipment. Replace/calibrate parameters after hardware is identified.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class SimulationConfig:
    """Placeholder parameters for reproducible research experiments."""

    dt_s: float = 0.2
    episode_steps: int = 250

    # Placeholder electrical ratings: not tied to any real equipment.
    nominal_frequency_hz: float = 50.0
    generator_rated_power_kw: float = 4.0
    dc_charge_voltage_v: float = 50.0
    inverter_efficiency: float = 0.90
    max_charge_current_a: float = 50.0

    # Simple frequency/governor dynamics. Must be calibrated for a real set.
    inertia_constant_s: float = 2.5
    governor_time_constant_s: float = 0.8
    droop_hz_at_rated_power: float = 2.0
    damping_pu_per_pu: float = 1.0

    # Simple one-node thermal model; illustrative only.
    battery_internal_resistance_ohm: float = 0.02
    battery_thermal_capacity_j_per_c: float = 2000.0
    battery_cooling_w_per_c: float = 4.0
    ambient_temp_c: float = 25.0

    # Broad research episode stop conditions; NOT safety limits for hardware.
    stop_frequency_low_hz: float = 45.0
    stop_frequency_high_hz: float = 55.0
    stop_battery_temp_c: float = 60.0

    # Simulator action guard / soft limits; NOT a real protective system.
    action_request_limit_a: float = 0.5
    max_delta_current_a_per_step: float = 0.5
    guard_frequency_hz: float = 49.0
    guard_df_dt_hz_s: float = -1.25
    guard_loads_a_downward_step: bool = True

    # Synthetic household-load scenarios, in kW.
    load_choices_kw: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 3.5)
    load_change_min_steps: int = 10
    load_change_max_steps: int = 25

    # Randomized initial condition and charge target for training diversity.
    initial_current_low_a: float = 2.0
    initial_current_high_a: float = 10.0
    target_current_low_a: float = 15.0
    target_current_high_a: float = 35.0
    initial_temp_low_c: float = 20.0
    initial_temp_high_c: float = 35.0
    initial_load_low_kw: float = 0.25
    initial_load_high_kw: float = 1.5

    def validate(self) -> None:
        if self.dt_s <= 0 or self.episode_steps < 1:
            raise ValueError("dt_s must be positive and episode_steps >= 1")
        if self.generator_rated_power_kw <= 0:
            raise ValueError("generator_rated_power_kw must be positive")
        if not 0 < self.inverter_efficiency <= 1:
            raise ValueError("inverter_efficiency must be in (0, 1]")
        if self.max_charge_current_a <= 0:
            raise ValueError("max_charge_current_a must be positive")
        if self.action_request_limit_a <= 0 or self.max_delta_current_a_per_step <= 0:
            raise ValueError("action and slew limits must be positive")
        if not self.stop_frequency_low_hz < self.nominal_frequency_hz < self.stop_frequency_high_hz:
            raise ValueError("stop frequency bounds must enclose nominal frequency")
        if self.inertia_constant_s <= 0 or self.governor_time_constant_s <= 0:
            raise ValueError("inertia and governor time constant must be positive")
        if self.droop_hz_at_rated_power <= 0:
            raise ValueError("droop_hz_at_rated_power must be positive")
        if self.battery_thermal_capacity_j_per_c <= 0:
            raise ValueError("battery thermal capacity must be positive")
        if not self.load_choices_kw or any(x < 0 for x in self.load_choices_kw):
            raise ValueError("load_choices_kw must contain non-negative values")
        if self.load_change_min_steps < 1 or self.load_change_max_steps < self.load_change_min_steps:
            raise ValueError("invalid load-change step range")


class GeneratorInverterEnv(gym.Env):
    """A synthetic Gymnasium environment; not a hardware controller.

    Observation order:
      [frequency_hz, df_dt_hz_s, charge_current_a, battery_temp_c,
       house_load_kw, target_charge_current_a, mechanical_power_kw]

    Action:
      One normalized scalar in [-1, 1], mapped to a small requested current
      change in amperes per simulation step. The simulator applies a rate limit
      and a simple research-only guard.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: SimulationConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SimulationConfig()
        self.config.validate()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        self.render_mode = render_mode

        # PPO-friendly normalized action. The simulator maps it to an amp request.
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        max_house_load_kw = max(5.0, max(self.config.load_choices_kw), self.config.initial_load_high_kw)
        self.observation_space = spaces.Box(
            low=np.array([30.0, -25.0, 0.0, -10.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([70.0, 25.0, self.config.max_charge_current_a, 90.0, max_house_load_kw,
                           self.config.max_charge_current_a,
                           self.config.generator_rated_power_kw], dtype=np.float32),
            dtype=np.float32,
        )

        self.step_count = 0
        self.freq_hz = self.config.nominal_frequency_hz
        self.df_dt_hz_s = 0.0
        self.charge_current_a = 5.0
        self.battery_temp_c = self.config.ambient_temp_c
        self.house_load_kw = 0.5
        self.target_charge_current_a = 20.0
        self.mechanical_power_kw = 0.5
        self.mechanical_power_setpoint_kw = 0.5
        self.next_load_change_step = self.config.load_change_max_steps
        self.last_total_load_kw = 0.0
        self.last_overload_kw = 0.0
        self.last_guard_active = False
        self.last_action_requested_a = 0.0
        self.last_action_applied_a = 0.0

    def _electrical_load_kw(self) -> float:
        dc_charge_kw = self.config.dc_charge_voltage_v * self.charge_current_a / 1000.0
        ac_charge_kw = dc_charge_kw / self.config.inverter_efficiency
        return float(self.house_load_kw + ac_charge_kw)

    def _observation(self) -> np.ndarray:
        return np.asarray(
            [self.freq_hz, self.df_dt_hz_s, self.charge_current_a,
             self.battery_temp_c, self.house_load_kw,
             self.target_charge_current_a, self.mechanical_power_kw],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        opts = options or {}
        cfg = self.config
        self.step_count = 0
        self.freq_hz = float(opts["frequency_hz"] if "frequency_hz" in opts else
                             cfg.nominal_frequency_hz + self.np_random.uniform(-0.1, 0.1))
        self.df_dt_hz_s = 0.0
        self.charge_current_a = float(opts["charge_current_a"] if "charge_current_a" in opts else
                                      self.np_random.uniform(cfg.initial_current_low_a, cfg.initial_current_high_a))
        self.battery_temp_c = float(opts["battery_temp_c"] if "battery_temp_c" in opts else
                                    self.np_random.uniform(cfg.initial_temp_low_c, cfg.initial_temp_high_c))
        self.house_load_kw = float(opts["house_load_kw"] if "house_load_kw" in opts else
                                   self.np_random.uniform(cfg.initial_load_low_kw, cfg.initial_load_high_kw))
        self.target_charge_current_a = float(opts["target_charge_current_a"] if "target_charge_current_a" in opts else
                                              self.np_random.uniform(cfg.target_current_low_a, cfg.target_current_high_a))
        self.charge_current_a = float(np.clip(self.charge_current_a, 0.0, cfg.max_charge_current_a))
        self.target_charge_current_a = float(np.clip(self.target_charge_current_a, 0.0, cfg.max_charge_current_a))
        self.battery_temp_c = max(-10.0, self.battery_temp_c)
        self.house_load_kw = max(0.0, self.house_load_kw)

        initial_load = self._electrical_load_kw()
        self.mechanical_power_kw = float(np.clip(initial_load, 0.0, cfg.generator_rated_power_kw))
        self.mechanical_power_setpoint_kw = self.mechanical_power_kw
        self.next_load_change_step = int(self.np_random.integers(
            cfg.load_change_min_steps, cfg.load_change_max_steps + 1))
        self.last_total_load_kw = initial_load
        self.last_overload_kw = max(0.0, initial_load - cfg.generator_rated_power_kw)
        self.last_guard_active = False
        self.last_action_requested_a = 0.0
        self.last_action_applied_a = 0.0

        return self._observation(), {"simulation_only": True, "config": asdict(cfg)}

    def step(self, action):
        cfg = self.config
        self.step_count += 1

        raw = np.asarray(action, dtype=np.float64).reshape(-1)
        if raw.size != 1 or not np.all(np.isfinite(raw)):
            requested = 0.0
        else:
            normalized_action = float(np.clip(raw[0], self.action_space.low[0], self.action_space.high[0]))
            requested = normalized_action * cfg.action_request_limit_a
        delta = float(np.clip(requested, -cfg.max_delta_current_a_per_step,
                              cfg.max_delta_current_a_per_step))

        # Research-only supervisory guard. This is not a certified safety function.
        guard_active = (self.freq_hz < cfg.guard_frequency_hz or
                        self.df_dt_hz_s < cfg.guard_df_dt_hz_s)
        if guard_active and cfg.guard_loads_a_downward_step:
            delta = min(delta, -cfg.max_delta_current_a_per_step)

        previous_current = self.charge_current_a
        self.charge_current_a = float(np.clip(
            self.charge_current_a + delta, 0.0, cfg.max_charge_current_a))
        applied_delta = self.charge_current_a - previous_current

        if self.step_count >= self.next_load_change_step:
            choices = np.asarray(cfg.load_choices_kw, dtype=np.float64)
            self.house_load_kw = float(self.np_random.choice(choices))
            self.next_load_change_step = self.step_count + int(self.np_random.integers(
                cfg.load_change_min_steps, cfg.load_change_max_steps + 1))

        total_load_kw = self._electrical_load_kw()
        overload_kw = max(0.0, total_load_kw - cfg.generator_rated_power_kw)
                # Research-only overload override:
        # shed simulated charging load when total demand exceeds rating.
        emergency_charge_shed = False

        if cfg.guard_loads_a_downward_step and overload_kw > 1e-9:
            self.charge_current_a = 0.0
            applied_delta = self.charge_current_a - previous_current

            total_load_kw = self._electrical_load_kw()
            overload_kw = max(
                0.0,
                total_load_kw - cfg.generator_rated_power_kw,
            )

            guard_active = True
            emergency_charge_shed = True
        # First-order governor response with a simple droop law.
        governor_gain_kw_per_hz = cfg.generator_rated_power_kw / cfg.droop_hz_at_rated_power
        governor_target_kw = self.mechanical_power_setpoint_kw + governor_gain_kw_per_hz * (
            cfg.nominal_frequency_hz - self.freq_hz)
        governor_target_kw = float(np.clip(governor_target_kw, 0.0, cfg.generator_rated_power_kw))
        governor_alpha = min(1.0, cfg.dt_s / cfg.governor_time_constant_s)
        self.mechanical_power_kw += governor_alpha * (governor_target_kw - self.mechanical_power_kw)
        self.mechanical_power_kw = float(np.clip(self.mechanical_power_kw, 0.0, cfg.generator_rated_power_kw))

        # Simplified swing-equation-like frequency dynamics in per-unit power.
        power_imbalance_pu = (self.mechanical_power_kw - total_load_kw) / cfg.generator_rated_power_kw
        speed_deviation_pu = (self.freq_hz - cfg.nominal_frequency_hz) / cfg.nominal_frequency_hz
        damping_pu = cfg.damping_pu_per_pu * speed_deviation_pu
        df_dt = cfg.nominal_frequency_hz / (2.0 * cfg.inertia_constant_s) * (power_imbalance_pu - damping_pu)
        old_freq = self.freq_hz
        self.freq_hz = float(self.freq_hz + df_dt * cfg.dt_s)
        self.df_dt_hz_s = float((self.freq_hz - old_freq) / cfg.dt_s)

        # One-node thermal approximation: I^2 R heating minus ambient cooling.
        joule_heat_w = self.charge_current_a ** 2 * cfg.battery_internal_resistance_ohm
        cooling_w = cfg.battery_cooling_w_per_c * (self.battery_temp_c - cfg.ambient_temp_c)
        dtemp_dt = (joule_heat_w - cooling_w) / cfg.battery_thermal_capacity_j_per_c
        self.battery_temp_c = float(self.battery_temp_c + dtemp_dt * cfg.dt_s)

        freq_error = self.freq_hz - cfg.nominal_frequency_hz
        current_error = self.charge_current_a - self.target_charge_current_a
        slew_penalty = 0.02 * (applied_delta / cfg.dt_s) ** 2
        frequency_penalty = 3.0 * freq_error ** 2
        target_penalty = 0.025 * current_error ** 2
        overload_penalty = 5.0 * overload_kw ** 2
        # 40 C is a simulation penalty threshold, not a universal battery limit.
        thermal_penalty = 0.2 * max(0.0, self.battery_temp_c - 40.0) ** 2
        reward = -(frequency_penalty + target_penalty + slew_penalty + overload_penalty + thermal_penalty)

        terminated = bool(
            self.freq_hz < cfg.stop_frequency_low_hz
            or self.freq_hz > cfg.stop_frequency_high_hz
            or self.battery_temp_c > cfg.stop_battery_temp_c
        )
        if terminated:
            reward -= 100.0
        truncated = bool(self.step_count >= cfg.episode_steps and not terminated)

        self.last_total_load_kw = total_load_kw
        self.last_overload_kw = overload_kw
        self.last_guard_active = guard_active
        self.last_action_requested_a = requested
        self.last_action_applied_a = applied_delta

        obs = self._observation()
        info = {
            "simulation_only": True,
            "action_requested_a": requested,
            "action_applied_delta_a": applied_delta,
            "guard_active": guard_active,
            "total_load_kw": total_load_kw,
            "overload_kw": overload_kw,
            "mechanical_power_kw": self.mechanical_power_kw,
            "emergency_charge_shed": emergency_charge_shed,
            "reward_components": {
                "frequency_penalty": frequency_penalty,
                "target_penalty": target_penalty,
                "slew_penalty": slew_penalty,
                "overload_penalty": overload_penalty,
                "thermal_penalty": thermal_penalty,
            },
        }
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        print(
            f"t={self.step_count:04d} f={self.freq_hz:6.2f} Hz "
            f"I={self.charge_current_a:5.2f} A "
            f"T={self.battery_temp_c:5.2f} C "
            f"load={self.house_load_kw:4.2f} kW "
            f"P_mech={self.mechanical_power_kw:4.2f} kW"
        )

    def close(self) -> None:
        return None

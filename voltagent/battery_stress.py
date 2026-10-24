"""Research-only battery current and thermal stress model.

This is an illustrative model, not a battery safety controller.
All parameters require validation against real battery data.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BatteryStressConfig:
    dt_s: float = 0.2
    dc_voltage_v: float = 50.0
    internal_resistance_ohm: float = 0.02
    thermal_capacity_j_per_c: float = 2000.0
    cooling_w_per_c: float = 4.0
    ambient_temp_c: float = 25.0
    max_abs_current_a: float = 50.0

    def validate(self) -> None:
        values = (
            self.dt_s,
            self.dc_voltage_v,
            self.internal_resistance_ohm,
            self.thermal_capacity_j_per_c,
            self.cooling_w_per_c,
            self.ambient_temp_c,
            self.max_abs_current_a,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Configuration values must be finite")

        if self.dt_s <= 0 or self.dc_voltage_v <= 0:
            raise ValueError("Time step and voltage must be positive")

        if self.internal_resistance_ohm < 0 or self.cooling_w_per_c < 0:
            raise ValueError("Resistance and cooling cannot be negative")

        if self.thermal_capacity_j_per_c <= 0:
            raise ValueError("Thermal capacity must be positive")

        if self.max_abs_current_a <= 0:
            raise ValueError("Maximum current must be positive")


def simulate_battery_current_profile(
    requested_current_a: Sequence[float],
    config: BatteryStressConfig,
    *,
    initial_temp_c: float | None = None,
    max_slew_a_per_s: float | None = None,
) -> dict:
    """Simulate signed battery current and a one-node thermal response.

    Sign convention:
        Positive current: battery discharge/support.
        Negative current: battery charging.

    max_slew_a_per_s=None means immediate current tracking.
    A finite positive value limits the current change per time step.
    """

    config.validate()

    if not requested_current_a:
        raise ValueError("Current profile must not be empty")

    if max_slew_a_per_s is not None:
        if not math.isfinite(max_slew_a_per_s) or max_slew_a_per_s <= 0:
            raise ValueError("Current slew limit must be positive and finite")

    temperature = (
        config.ambient_temp_c
        if initial_temp_c is None
        else float(initial_temp_c)
    )
    if not math.isfinite(temperature):
        raise ValueError("Initial temperature must be finite")

    current = 0.0
    current_history = []
    temperature_history = []
    resistive_heat_j = 0.0
    peak_current_tracking_error_a = 0.0

    for requested in requested_current_a:
        requested = float(requested)
        if not math.isfinite(requested):
            raise ValueError("Requested currents must be finite")

        target = max(
            -config.max_abs_current_a,
            min(config.max_abs_current_a, requested),
        )

        if max_slew_a_per_s is None:
            current = target
        else:
            max_delta = max_slew_a_per_s * config.dt_s
            delta = max(-max_delta, min(max_delta, target - current))
            current += delta

        # Resistive heating depends on current magnitude, not direction.
        heating_w = current**2 * config.internal_resistance_ohm
        cooling_w = config.cooling_w_per_c * (
            temperature - config.ambient_temp_c
        )
        temperature += (
            (heating_w - cooling_w)
            / config.thermal_capacity_j_per_c
            * config.dt_s
        )

        resistive_heat_j += heating_w * config.dt_s

        # Difference between requested and applied current.
        current_tracking_error_a = abs(requested - current)
        peak_current_tracking_error_a = max(
            peak_current_tracking_error_a,
            current_tracking_error_a,
        )

        current_history.append(current)
        temperature_history.append(temperature)

    return {
        "current_profile_a": current_history,
        "temperature_profile_c": temperature_history,
        "peak_abs_current_a": max(abs(value) for value in current_history),
        "peak_temp_c": max(temperature_history),
        "resistive_heat_j": resistive_heat_j,
        "peak_current_tracking_error_a": peak_current_tracking_error_a,
    }
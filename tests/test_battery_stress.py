from voltagent.battery_stress import (
    BatteryStressConfig,
    simulate_battery_current_profile,
)


def test_slew_limiting_exposes_thermal_and_power_tradeoff():
    config = BatteryStressConfig(
        dt_s=0.2,
        dc_voltage_v=50.0,
        internal_resistance_ohm=0.02,
        thermal_capacity_j_per_c=2000.0,
        cooling_w_per_c=4.0,
        ambient_temp_c=25.0,
        max_abs_current_a=50.0,
    )

    # Synthetic discharge-support demand lasting four seconds.
    shock_profile = [40.0] * 20

    immediate = simulate_battery_current_profile(
        shock_profile,
        config,
    )

    slew_limited = simulate_battery_current_profile(
        shock_profile,
        config,
        max_slew_a_per_s=5.0,
    )

    # The ramp-limited response must respect its per-step rate limit.
    currents = slew_limited["current_profile_a"]
    max_delta = 5.0 * config.dt_s

    assert currents[0] <= max_delta
    assert all(
        abs(currents[i] - currents[i - 1]) <= max_delta + 1e-9
        for i in range(1, len(currents))
    )

    # Under this short scenario, ramp limiting reduces modeled heating.
    assert (
        slew_limited["resistive_heat_j"]
        < immediate["resistive_heat_j"]
    )
    assert slew_limited["peak_temp_c"] < immediate["peak_temp_c"]

    # But the slower response also leaves some requested power unsupported.
    assert (
        slew_limited["peak_current_tracking_error_a"]
        > immediate["peak_current_tracking_error_a"]
    )

def test_slew_limit_controls_bidirectional_current_shocks():
    config = BatteryStressConfig(
        dt_s=0.2,
        dc_voltage_v=50.0,
        internal_resistance_ohm=0.02,
        thermal_capacity_j_per_c=2000.0,
        cooling_w_per_c=4.0,
        ambient_temp_c=25.0,
        max_abs_current_a=50.0,
    )

    # Synthetic sequence: idle, discharge support, charging, then idle.
    profile = (
        [0.0] * 5
        + [40.0] * 20
        + [-20.0] * 30
        + [0.0] * 30
    )

    immediate = simulate_battery_current_profile(profile, config)

    controlled = simulate_battery_current_profile(
        profile,
        config,
        max_slew_a_per_s=5.0,
    )

    currents = controlled["current_profile_a"]
    max_delta = 5.0 * config.dt_s

    assert all(
        abs(currents[i] - currents[i - 1]) <= max_delta + 1e-9
        for i in range(1, len(currents))
    )

    assert controlled["peak_abs_current_a"] < immediate["peak_abs_current_a"]
    assert controlled["resistive_heat_j"] < immediate["resistive_heat_j"]
    assert (
        controlled["peak_current_tracking_error_a"]
        > immediate["peak_current_tracking_error_a"]
    )
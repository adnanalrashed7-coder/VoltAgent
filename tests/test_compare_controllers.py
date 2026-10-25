import numpy as np

from compare_controllers import controller_action


def test_frequency_aware_controller_discharges_when_frequency_is_low():
    obs = np.array([49.5, -0.6, 0.0, 25.0, 0.5, 0.6, 1.0], dtype=np.float32)

    action = controller_action("frequency_aware", obs, None)

    assert np.array_equal(action, np.array([1.0], dtype=np.float32))


def test_thermal_aware_controller_does_not_dispatch_above_temperature_threshold():
    obs = np.array([49.5, -0.6, 0.0, 41.0, 0.5, 0.6, 1.0], dtype=np.float32)

    action = controller_action("thermal_aware", obs, None)

    assert np.array_equal(action, np.array([0.0], dtype=np.float32))
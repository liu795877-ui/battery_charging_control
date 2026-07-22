from types import SimpleNamespace

import numpy as np

from battery_fast_charge.mpc import ConstrainedMPC, ReducedState


def test_constraint_slacks_are_reported_per_constraint_family() -> None:
    controller = object.__new__(ConstrainedMPC)
    controller.config = SimpleNamespace(
        constraints=SimpleNamespace(
            mpc_maximum_voltage_v=4.18,
            mpc_maximum_temperature_c=34.5,
            maximum_current_change_a_per_step=2.0,
        ),
        battery=SimpleNamespace(target_soc=0.8),
    )
    state = ReducedState(0.2, 0.0, 0.0, 25.0, 25.0, 0.0)
    prediction = {
        "voltage_v": np.asarray([4.17, 4.20]),
        "temperature_c": np.asarray([34.0, 35.0]),
        "soc": np.asarray([0.79, 0.81]),
    }
    slacks = controller._constraint_slacks(state, np.asarray([3.0, 0.0]), prediction)
    assert np.isclose(slacks["slack_voltage_v"], 0.02)
    assert np.isclose(slacks["slack_temperature_c"], 0.5)
    assert np.isclose(slacks["slack_soc"], 0.01)
    assert np.isclose(slacks["slack_current_change_a"], 1.0)

import numpy as np

from battery_fast_charge.reduced_model import (
    simulate_electrical_2rc,
    simulate_two_node_thermal,
)


def test_electrical_model_charging_direction() -> None:
    """正充电电流应使 SOC、极化压升和端电压上升。"""
    time_s = np.array([0.0, 5.0, 10.0])
    current_a = np.array([5.0, 5.0, 5.0])
    parameters = {
        "r0_ohm": 0.01,
        "r1_ohm": 0.01,
        "tau1_s": 10.0,
        "r2_ohm": 0.02,
        "tau2_s": 100.0,
    }
    frame = simulate_electrical_2rc(
        time_s,
        current_a,
        0.20,
        5.0,
        lambda soc: 3.5 + 0.5 * np.asarray(soc),
        parameters,
    )

    assert frame["soc_predicted"].is_monotonic_increasing
    assert frame["terminal_voltage_predicted_v"].iloc[-1] > 3.5
    assert (frame["electrical_loss_predicted_w"] >= 0.0).all()


def test_two_node_model_heats_and_cools() -> None:
    """正热输入应升温，停止加热后温度应向环境温度回落。"""
    time_s = np.arange(0.0, 205.0, 5.0)
    heat_w = np.where(time_s < 100.0, 2.0, 0.0)
    frame = simulate_two_node_thermal(
        time_s,
        heat_w,
        25.0,
        25.0,
        0.8,
        {
            "total_heat_capacity_j_per_k": 100.0,
            "r_core_surface_k_per_w": 1.0,
            "r_surface_ambient_k_per_w": 5.0,
            "heat_gain": 1.0,
        },
    )

    assert frame["average_temperature_predicted_c"].max() > 25.0
    assert (
        frame["average_temperature_predicted_c"].iloc[-1]
        < frame["average_temperature_predicted_c"].max()
    )

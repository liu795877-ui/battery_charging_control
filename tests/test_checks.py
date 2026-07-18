import pandas as pd

from battery_fast_charge.checks import check_trajectory


def test_valid_trajectory_checks_pass() -> None:
    """构造一条简单且物理方向正确的轨迹，确认所有检查均能通过。"""
    # 三个采样点模拟“时间前进、SOC 上升、电流非负”的正常充电过程。
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 5.0, 10.0],
            "charge_current_a": [1.0, 1.0, 0.5],
            "terminal_voltage_v": [3.5, 3.6, 3.7],
            "soc": [0.10, 0.11, 0.12],
            "cell_temperature_c": [25.0, 25.1, 25.2],
            "ambient_temperature_c": [25.0, 25.0, 25.0],
        }
    )

    # 若新增了检查项，这条断言也会自动要求新检查通过。
    assert all(check_trajectory(frame).values())

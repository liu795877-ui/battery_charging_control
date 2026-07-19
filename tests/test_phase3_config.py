from pathlib import Path

import pytest

from battery_fast_charge.phase3_config import load_phase_three_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase_three_configuration_and_safety_margins() -> None:
    """锁定第三阶段第一版的研究范围和保守内部约束。"""
    config = load_phase_three_config(PROJECT_ROOT / "configs" / "phase3.yaml")

    assert config.battery.initial_soc == 0.10
    assert config.battery.target_soc == 0.80
    assert config.constraints.mpc_maximum_voltage_v == pytest.approx(4.14)
    assert config.constraints.mpc_maximum_temperature_c == pytest.approx(33.5)
    assert config.control.number_of_control_blocks == 12

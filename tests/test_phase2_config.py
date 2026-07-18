from pathlib import Path

from battery_fast_charge.phase2_config import load_phase_two_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase_two_configuration() -> None:
    """确认第二阶段关键协议和双节点约束没有被意外改坏。"""
    config = load_phase_two_config(PROJECT_ROOT / "configs" / "phase2.yaml")

    assert config.battery.parameter_set == "Chen2020"
    assert config.experiment.sample_period_s == 5.0
    assert config.experiment.ocv_soc_points[0] == 0.10
    assert config.experiment.ocv_soc_points[-1] == 0.80
    assert config.identification.core_heat_capacity_fraction == 0.80

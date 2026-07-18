from pathlib import Path

from battery_fast_charge.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase_one_configuration() -> None:
    """防止关键实验参数被误改，或 YAML 到数据类的映射失效。"""
    config = load_config(PROJECT_ROOT / "configs" / "phase1.yaml")

    # 这些断言对应第一版已经确认的参数，而不是随意选择的测试数字。
    assert config.battery.parameter_set == "Chen2020"
    assert config.battery.initial_soc == 0.10
    assert config.battery.target_soc == 0.80
    assert config.constraints.maximum_current_a == 10.0
    assert config.control.prediction_horizon_s == 300.0

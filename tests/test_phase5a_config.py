from dataclasses import replace

import pytest

from battery_fast_charge.phase5a_config import load_phase_five_a_config


def test_phase5a_configuration_loads_expected_validation_scope() -> None:
    """第一版鲁棒性验证应固定 64 个 LHS 样本和 3 个 DFN 温度锚点。"""
    config = load_phase_five_a_config("configs/phase5a.yaml")

    assert config.reduced_stress_test.random_scenario_count == 64
    assert config.dfn_temperature_anchors.temperatures_c == (15.0, 25.0, 30.0)
    assert config.success_criteria.minimum_reduced_physical_safety_fraction == 1.0


def test_phase5a_rejects_ambient_temperature_at_control_limit() -> None:
    """初始环境温度不能已经处在 33.5 ℃ 收紧边界上。"""
    config = load_phase_five_a_config("configs/phase5a.yaml")
    invalid_stress = replace(
        config.reduced_stress_test,
        ambient_temperature_c_range=(15.0, 33.5),
    )

    with pytest.raises(ValueError):
        # 通过临时 YAML 之外的内部校验入口验证边界契约。
        from battery_fast_charge.phase5a_config import _validate

        _validate(replace(config, reduced_stress_test=invalid_stress))

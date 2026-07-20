from pathlib import Path

from battery_fast_charge.phase4b_config import load_phase_four_b_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase_four_b_uses_state_triggered_thermal_budget() -> None:
    """锁定8 A到5 A参考、状态触发和1%改进闸门。"""
    config = load_phase_four_b_config(PROJECT_ROOT / "configs" / "phase4b.yaml")

    reference = config.thermal_budget_reference
    assert reference.peak_current_a == 8.0
    assert reference.sustainable_current_a == 5.0
    assert 0.17 < reference.switch_soc < 0.18
    assert reference.reference_release_soc == 0.20
    assert reference.reoptimize_every_control_step is True
    assert config.success_criteria.minimum_improvement_fraction_over_filtered_1c == 0.01

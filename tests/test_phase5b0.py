from battery_fast_charge.phase5b0_config import load_phase_five_b_zero_config
from battery_fast_charge.phase5b0_runner import classify_scenario, infeasibility_reasons


def test_phase5b0_contract_preserves_phase5a_scope_and_oracle_state_boundary() -> None:
    config = load_phase_five_b_zero_config("configs/phase5b0_mpc_feasibility_envelope.yaml")
    assert config.scope.expected_reduced_scenario_count == 69
    assert not config.scope.run_dfn_anchors
    assert config.information_boundary["oracle_has_perfect_state"] is False


def test_phase5b0_four_primary_classes() -> None:
    assert classify_scenario(True, True, False) == "teacher_feasible_ann_failed"
    assert classify_scenario(False, False, False) == "teacher_and_ann_infeasible"
    assert classify_scenario(True, True, True) == "teacher_and_ann_feasible"
    assert classify_scenario(False, True, False) == "nominal_teacher_failed_oracle_teacher_feasible"


def test_infeasibility_reasons_are_explicit() -> None:
    summary = {
        "completion_success": False, "physical_safe": False,
        "voltage_limit_exceeded": True, "temperature_limit_exceeded": False,
        "current_limit_exceeded": False, "current_change_limit_exceeded": False,
        "optimizer_success_fraction": 0.8, "required_optimizer_success_fraction": 0.95,
        "fallback_count": 1, "maximum_allowed_fallback_count": 0,
    }
    assert infeasibility_reasons(summary) == "target_not_reached;voltage_violation;optimizer_success_fraction;mpc_fallback"

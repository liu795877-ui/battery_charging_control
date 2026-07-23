import json
from pathlib import Path

from battery_fast_charge.phase7a_level3_model import Level3State
from battery_fast_charge.phase7b1b_config import load_phase7b1b_config
from battery_fast_charge.phase7b1b_runner import (
    _load_context,
    _maximum_safe_current,
    _predicted_next_voltage,
    verify_frozen_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase7b1b_voltage_safety_layer.yaml"


def test_phase7b1b_contract_uses_frozen_phase7b1a_guard() -> None:
    config = load_phase7b1b_config(CONFIG)
    verification = verify_frozen_artifacts(config, ROOT)
    assert len(verification) == 5
    assert all(item["matched"] for item in verification.values())
    assert config.safety.residual_growth_guard_v > 0.0


def test_voltage_safe_current_respects_corrected_one_step_limit() -> None:
    config = load_phase7b1b_config(CONFIG)
    _, _, model, _, _ = _load_context(config, ROOT)
    state = Level3State(
        soc=0.74,
        polarization_1_v=0.04,
        polarization_2_v=0.06,
        previous_current_a=6.0,
    )
    correction = 0.025 + config.safety.residual_growth_guard_v
    maximum = _maximum_safe_current(state, correction, config, model)
    assert 0.0 <= maximum <= 10.0
    corrected_voltage = (
        _predicted_next_voltage(state, maximum, model) + correction
    )
    assert corrected_voltage <= config.safety.voltage_limit_v + 1.0e-9
    assert (
        _predicted_next_voltage(
            state,
            min(maximum + 1.0e-5, 10.0),
            model,
        )
        + correction
        >= config.safety.voltage_limit_v - 1.0e-6
    )


def test_phase7b1_regression_and_confirmation_pass_strict_gates() -> None:
    result_directory = ROOT / "outputs/phase7b1b_voltage_safety"
    for filename in ("regression_metrics.json", "confirmation_metrics.json"):
        payload = json.loads(
            (result_directory / filename).read_text(encoding="utf-8")
        )
        assert payload["decision"]["success"]
        assert all(payload["decision"]["checks"].values())

        summary = payload["residual_guard"]
        assert summary["maximum_ann_voltage_v"] <= 4.2 + 1.0e-6
        assert summary["maximum_mpc_voltage_v"] <= 4.2 + 1.0e-6
        assert summary["current_nrmse_max"] < 1.0
        assert summary["maximum_charge_time_gap_fraction"] < 0.02
        assert summary["minimum_target_reach_fraction"] == 1.0
        assert summary["minimum_speedup"] > 100.0


def test_phase7b1_final_decision_does_not_require_short_horizon() -> None:
    payload = json.loads(
        (
            ROOT / "outputs/phase7b1b_voltage_safety/metrics.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["regression_success"]
    assert payload["confirmation_success"]
    assert payload["decision"]["phase7b1_complete_success"]
    assert payload["decision"]["proceed_to_multi_temperature"]
    assert not payload["decision"]["short_horizon_required"]

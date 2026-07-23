import json
from pathlib import Path

from battery_fast_charge.phase7cr1_config import load_phase7cr1_config
from battery_fast_charge.phase7cr1_runner import (
    _maximum_predicted_temperature,
    maximum_thermal_safe_current,
    verify_frozen_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase7cr1_thermal_supervisor.yaml"


def test_phase7cr1_preserves_frozen_r0_evidence() -> None:
    config = load_phase7cr1_config(CONFIG)
    verification = verify_frozen_artifacts(config, ROOT)
    assert len(verification) == 5
    assert all(item["matched"] for item in verification.values())


def test_phase7cr1_uses_preregistered_short_horizon_contract() -> None:
    config = load_phase7cr1_config(CONFIG)
    assert config.thermal["prediction_horizon_steps"] == 60
    assert config.thermal["prediction_horizon_s"] == 300.0
    assert config.thermal["future_current_policy"] == "constant_candidate"
    assert config.thermal["braking_floor_current_a"] == 0.0
    assert config.thermal["temperature_guard_c"] == 0.10
    assert config.thermal["sustainable_current_a"] == 3.0
    assert config.development["trajectory_count"] == 8


def test_phase7cr1_thermal_limit_respects_guarded_prediction() -> None:
    config = load_phase7cr1_config(CONFIG)
    safe_current, peak = maximum_thermal_safe_current(
        temperature_c=34.5,
        search_upper_a=10.0,
        config=config,
    )
    guarded_limit = (
        config.thermal["maximum_average_temperature_c"]
        - config.thermal["temperature_guard_c"]
    )
    assert 0.0 <= safe_current <= 10.0
    assert peak <= guarded_limit + 1.0e-9
    assert (
        _maximum_predicted_temperature(
            min(10.0, safe_current + 0.01),
            34.5,
            config,
        )
        > guarded_limit
    )


def test_phase7cr1_never_authorizes_ann_execution() -> None:
    config = load_phase7cr1_config(CONFIG)
    path = ROOT / config.output["result_directory"] / "metrics.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert not payload["decision"]["run_ann"]


def test_phase7cr1_success_requires_all_strict_development_gates() -> None:
    config = load_phase7cr1_config(CONFIG)
    path = ROOT / config.output["result_directory"] / "metrics.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["success"] == all(payload["checks"].values())
    assert payload["development_set"]["not_independent_confirmation"]

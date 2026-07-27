from __future__ import annotations

import hashlib
import json
from pathlib import Path

from battery_fast_charge.phase7d2_config import load_phase7d2_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_phase7d2_config(
    ROOT / "configs/phase7d2_thermal_performance_development.yaml"
)


def test_level4_2_scope_is_preregistered_before_rollouts() -> None:
    thermal = CONFIG.section("thermal_candidates")
    selection = CONFIG.section("selection_contract")
    assert thermal["prediction_horizon_steps"] == 60
    assert thermal["baseline_temperature_guard_c"] == 0.1
    assert thermal["candidate_temperature_guards_c"] == [0.075, 0.05, 0.025]
    assert thermal["only_30c_is_developed"] is True
    assert thermal["temperature_model_change_authorized"] is False
    assert thermal["horizon_change_authorized"] is False
    assert selection["safety_first"] is True
    assert selection["selection_temperature_c"] == 30
    assert selection["primary_metric"] == "mean_charge_time_s"
    assert selection["internal_validation_may_not_modify_selection"] is True
    assert selection["internal_validation_is_one_shot"] is True
    assert selection["r3_confirmation_used_for_tuning"] is False
    assert selection["ann_retraining_authorized"] is False


def test_level4_2_frozen_sources_match() -> None:
    sources = CONFIG.section("sources")
    for path_key, hash_key in (
        ("phase7d1_state_freeze", "phase7d1_state_freeze_sha256"),
        ("frozen_voltage_guards", "frozen_voltage_guards_sha256"),
    ):
        actual = hashlib.sha256((ROOT / sources[path_key]).read_bytes()).hexdigest()
        assert actual == sources[hash_key]


def test_level4_2_selection_freeze_if_present() -> None:
    result_dir = ROOT / CONFIG.section("output")["result_directory"]
    path = result_dir / "selected_parameters.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["voltage_guards_unchanged"] is True
    assert payload["ann_retrained"] is False
    assert payload["internal_validation_started"] is False
    assert payload["internal_validation_used_for_retuning"] is False
    assert payload["independent_confirmation_created"] is False


def test_level4_2_internal_validation_is_strict_if_present() -> None:
    result_dir = ROOT / CONFIG.section("output")["result_directory"]
    path = result_dir / "internal_validation_metrics.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] in {"strict_passed", "strict_stop_failed"}
    assert payload["internal_validation_used_for_retuning"] is False
    if payload["success"]:
        assert payload["failed_checks"] == []
        assert payload["independent_confirmation_authorized"] is True


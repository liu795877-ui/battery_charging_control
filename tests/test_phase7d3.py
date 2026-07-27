from __future__ import annotations

import hashlib
import json
from pathlib import Path

from battery_fast_charge.phase7d3_config import load_phase7d3_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_phase7d3_config(ROOT / "configs/phase7d3_final_confirmation.yaml")


def test_level4_final_confirmation_contract_is_preregistered() -> None:
    contract = CONFIG.section("confirmation_contract")
    thermal = CONFIG.section("thermal_candidates")
    assert contract["expected_trajectory_count_per_variant"] == 240
    assert contract["parameters_frozen_before_state_generation"] is True
    assert contract["confirmation_may_not_modify_parameters"] is True
    assert contract["r3_confirmation_used_for_tuning"] is False
    assert contract["ann_retraining_authorized"] is False
    assert thermal["prediction_horizon_steps"] == 60
    assert thermal["baseline_temperature_guard_c"] == 0.1
    assert thermal["selected_temperature_guard_c"] == 0.025


def test_level4_final_sources_match() -> None:
    sources = CONFIG.section("sources")
    for path_key, hash_key in (
        ("phase7d2_selection", "phase7d2_selection_sha256"),
        ("phase7d2_internal_validation", "phase7d2_internal_validation_sha256"),
        ("frozen_voltage_guards", "frozen_voltage_guards_sha256"),
    ):
        assert hashlib.sha256((ROOT / sources[path_key]).read_bytes()).hexdigest() == sources[hash_key]


def test_level4_final_states_are_frozen_if_present() -> None:
    data_dir = ROOT / CONFIG.section("output")["data_directory"]
    path = data_dir / "state_freeze.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "states_frozen_before_final_confirmation"
    assert payload["prior_states_excluded"] is True
    assert payload["parameters_frozen_before_state_generation"] is True
    assert payload["confirmation_started"] is False
    assert payload["confirmation_used_for_retuning"] is False
    assert payload["level4_completed"] is False
    assert sum(record["trajectory_count"] for record in payload["files"].values()) == 48
    assert sum(record["zero_residual_count"] for record in payload["files"].values()) == 0
    for name, record in payload["files"].items():
        assert hashlib.sha256((data_dir / name).read_bytes()).hexdigest() == record["sha256"]


def test_level4_final_result_is_strict_if_present() -> None:
    result_dir = ROOT / CONFIG.section("output")["result_directory"]
    path = result_dir / "metrics.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] in {"strict_passed", "strict_stop_failed"}
    assert payload["confirmation_used_for_retuning"] is False
    assert payload["ann_retrained"] is False
    if payload["success"]:
        assert payload["failed_checks"] == []
        assert payload["decision"]["level4_completed"] is True


from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from battery_fast_charge.phase7cr2f4_config import load_phase7cr2f4_config
from battery_fast_charge.phase7cr2f4_runner import derive_25c_running_guard


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_phase7cr2f4_config(
    ROOT / "configs/phase7cr2f4_25c_running_guard.yaml"
)


def test_only_25c_running_guard_is_developed() -> None:
    contract = CONFIG.section("control_contract")
    assert contract["only_25c_running_guard_is_developed"] is True
    assert contract["r3_generation_authorized"] is False
    assert contract["ann_execution_authorized"] is False
    assert contract["level4_entry_authorized"] is False


def test_preregistered_running_guard_formula_and_frozen_other_guards() -> None:
    frame = pd.DataFrame(
        {
            "ambient_temperature_c": [25.0, 25.0],
            "step_index": [2, 3],
            "positive_residual_growth_v": [0.013, 0.014],
        }
    )
    result = derive_25c_running_guard(CONFIG.section("voltage_guard"), frame)
    assert result["guards"]["25"]["running_v"] == pytest.approx(0.0145)
    frozen = CONFIG.section("voltage_guard")["frozen_guards"]
    assert result["guards"]["15"] == frozen[15]
    assert result["guards"]["25"]["boot_v"] == frozen[25]["boot_v"]
    assert result["guards"]["30"] == frozen[30]


def test_minimum_new_running_guard_includes_failure_plus_margin() -> None:
    voltage = CONFIG.section("voltage_guard")
    minimum = (
        voltage["historical_25c_running_minimum_v"]
        + voltage["engineering_margin_v"]
    )
    assert minimum == pytest.approx(0.01411071391754)


def test_new_data_and_validation_counts_are_preregistered() -> None:
    datasets = CONFIG.section("datasets")
    validation = CONFIG.section("validation_contract")
    assert datasets["development_count"] == 48
    assert datasets["internal_validation_count"] == 24
    assert validation["expected_historical_regression_trajectory_count"] == 296
    assert validation["expected_new_internal_trajectory_count"] == 24
    assert validation["expected_total_validation_trajectory_count"] == 320


def test_new_state_files_are_frozen_isolated_and_measured() -> None:
    data_dir = ROOT / CONFIG.section("output")["data_directory"]
    freeze = json.loads(
        (data_dir / "initial_state_freeze.json").read_text(encoding="utf-8")
    )
    assert freeze["frozen_before_any_closed_loop_rollout"] is True
    development = pd.read_csv(data_dir / "development_initial_states_25c.csv")
    internal = pd.read_csv(data_dir / "internal_validation_initial_states_25c.csv")
    assert len(development) == 48
    assert len(internal) == 24
    assert set(development.design_seed) == {20260901}
    assert set(internal.design_seed) == {20260902}
    columns = [
        "initial_soc",
        "initial_polarization_1_v",
        "initial_polarization_2_v",
        "initial_previous_current_a",
    ]
    development_states = {tuple(row) for row in development[columns].to_numpy()}
    internal_states = {tuple(row) for row in internal[columns].to_numpy()}
    assert development_states.isdisjoint(internal_states)
    for frame in (development, internal):
        assert frame.initial_measured_residual_v.notna().all()
        assert (frame.initial_measured_residual_v != 0.0).all()
    for name, record in freeze["files"].items():
        assert hashlib.sha256((data_dir / name).read_bytes()).hexdigest() == record[
            "sha256"
        ]


def test_r2f4_strictly_stops_during_development() -> None:
    result_dir = ROOT / CONFIG.section("output")["result_directory"]
    metrics = json.loads(
        (result_dir / "metrics.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (result_dir / "freeze_manifest.json").read_text(encoding="utf-8")
    )
    assert metrics["stage"] == "development"
    assert metrics["status"] == "strict_stop_failed"
    assert metrics["reason"] == "new_25c_boot_exceedance_while_boot_guard_frozen"
    assert metrics["boot_exceedance_count"] == 1
    assert metrics["decision"]["running_guard_derived_or_frozen"] is False
    assert metrics["decision"]["internal_validation_started"] is False
    assert manifest["status"] == "strict_stop_failed_during_development"
    assert manifest["running_guard_frozen"] is False
    assert manifest["internal_validation_started"] is False
    assert manifest["level4_entered"] is False


def test_single_boot_failure_is_frozen_and_validation_never_started() -> None:
    data_dir = ROOT / CONFIG.section("output")["data_directory"]
    result_dir = ROOT / CONFIG.section("output")["result_directory"]
    metrics = json.loads(
        (result_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert len(metrics["events"]) == 1
    event = metrics["events"][0]
    assert event["trajectory_id"] == "phase7cr2f4_development_25c_010"
    assert event["step_index"] == 0
    assert event["positive_residual_growth_v"] == pytest.approx(
        0.0358591351352006
    )
    assert event["guard_v"] == pytest.approx(0.0346117967629136)
    assert not (data_dir / "frozen_25c_running_guard.json").exists()
    assert not (data_dir / "validation_started.json").exists()

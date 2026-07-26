from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from battery_fast_charge.phase7cr2f5_config import load_phase7cr2f5_config
from battery_fast_charge.phase7cr2f5_runner import (
    derive_25c_two_stage_guards,
    verify_frozen_r2f4,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_phase7cr2f5_config(
    ROOT / "configs/phase7cr2f5_25c_two_stage_guards.yaml"
)


def test_r2f5_scope_does_not_authorize_r3_ann_or_level4() -> None:
    contract = CONFIG.section("control_contract")
    assert contract["only_25c_boot_and_running_guards_are_developed"] is True
    assert contract["guards_15c_and_30c_frozen"] is True
    assert contract["r3_generation_authorized"] is False
    assert contract["ann_execution_authorized"] is False
    assert contract["level4_entry_authorized"] is False


def test_r2f4_failure_manifest_is_frozen_and_verified() -> None:
    sources = CONFIG.section("sources")
    path = ROOT / sources["phase7cr2f4_freeze_manifest"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sources[
        "phase7cr2f4_freeze_manifest_sha256"
    ]
    result = verify_frozen_r2f4(CONFIG, ROOT)
    assert result["r2f4_failure_preserved"] is True
    assert all(record["matched"] for record in result["records"].values())


def test_two_stage_formula_uses_independent_observations_and_margin() -> None:
    frame = pd.DataFrame(
        {
            "ambient_temperature_c": [25.0, 25.0, 25.0, 25.0],
            "step_index": [0, 1, 2, 3],
            "positive_residual_growth_v": [0.036, 0.034, 0.014, 0.013],
        }
    )
    result = derive_25c_two_stage_guards(CONFIG.section("voltage_guard"), frame)
    assert result["guards"]["25"]["boot_v"] == pytest.approx(0.0365)
    assert result["guards"]["25"]["running_v"] == pytest.approx(0.0145)


def test_historical_floors_require_minimum_new_guards() -> None:
    voltage = CONFIG.section("voltage_guard")
    margin = voltage["engineering_margin_v"]
    assert voltage["historical_25c_boot_minimum_v"] + margin == pytest.approx(
        0.0363591351352006
    )
    assert voltage["historical_25c_running_minimum_v"] + margin == pytest.approx(
        0.01411071391754
    )


def test_15c_and_30c_guards_are_unchanged_by_derivation() -> None:
    frame = pd.DataFrame(
        {
            "ambient_temperature_c": [25.0, 25.0],
            "step_index": [0, 2],
            "positive_residual_growth_v": [0.04, 0.02],
        }
    )
    result = derive_25c_two_stage_guards(CONFIG.section("voltage_guard"), frame)
    frozen = CONFIG.section("voltage_guard")["frozen_guards"]
    assert result["guards"]["15"] == frozen[15]
    assert result["guards"]["30"] == frozen[30]


def test_new_data_and_one_shot_validation_counts_are_preregistered() -> None:
    datasets = CONFIG.section("datasets")
    validation = CONFIG.section("validation_contract")
    assert datasets["development_count"] == 48
    assert datasets["internal_validation_count"] == 24
    assert datasets["designs"]["development"][25]["seed"] == 20261001
    assert datasets["designs"]["internal_validation"][25]["seed"] == 20261002
    assert validation["expected_historical_regression_trajectory_count"] == 344
    assert validation["expected_new_internal_trajectory_count"] == 24
    assert validation["expected_total_validation_trajectory_count"] == 368


def test_frozen_state_files_are_isolated_and_measurement_initialized() -> None:
    data_dir = ROOT / CONFIG.section("output")["data_directory"]
    freeze_path = data_dir / "initial_state_freeze.json"
    if not freeze_path.exists():
        pytest.skip("R2F5 state generation has not started")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    development = pd.read_csv(data_dir / "development_initial_states_25c.csv")
    internal = pd.read_csv(data_dir / "internal_validation_initial_states_25c.csv")
    assert len(development) == 48
    assert len(internal) == 24
    assert set(development.design_seed) == {20261001}
    assert set(internal.design_seed) == {20261002}
    columns = [
        "initial_soc",
        "initial_polarization_1_v",
        "initial_polarization_2_v",
        "initial_previous_current_a",
    ]
    assert {
        tuple(row) for row in development[columns].to_numpy()
    }.isdisjoint({tuple(row) for row in internal[columns].to_numpy()})
    for frame in (development, internal):
        assert frame.initial_measured_residual_v.notna().all()
        assert (frame.initial_measured_residual_v != 0.0).all()
    for name, record in freeze["files"].items():
        assert hashlib.sha256((data_dir / name).read_bytes()).hexdigest() == record[
            "sha256"
        ]

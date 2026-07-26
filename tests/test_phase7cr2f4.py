from __future__ import annotations

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

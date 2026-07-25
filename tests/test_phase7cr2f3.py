from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from battery_fast_charge.phase7cr2f3_config import load_phase7cr2f3_config
from battery_fast_charge.phase7cr2f3_runner import (
    _temperature_stage_counts,
    derive_temperature_guards,
    guard_for_step,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_phase7cr2f3_config(
    ROOT / "configs/phase7cr2f3_temperature_two_stage_guards.yaml"
)


def test_all_temperatures_share_exact_stage_boundary() -> None:
    guards = {
        "15": {"boot_v": 0.037, "running_v": 0.015},
        "25": {"boot_v": 0.027, "running_v": 0.013},
        "30": {"boot_v": 0.038, "running_v": 0.022},
    }
    for temperature in (15, 25, 30):
        assert guard_for_step(temperature, 0, guards)[1] == "boot"
        assert guard_for_step(temperature, 1, guards)[1] == "boot"
        assert guard_for_step(temperature, 2, guards)[1] == "running"


def test_preregistered_guard_formulas_and_frozen_30c_values() -> None:
    voltage = CONFIG.section("voltage_guard")
    rows = []
    for temperature, values in (
        (15, (0.030, 0.013)),
        (25, (0.020, 0.012)),
    ):
        rows.extend(
            [
                {
                    "ambient_temperature_c": temperature,
                    "step_index": 0,
                    "positive_residual_growth_v": values[0],
                },
                {
                    "ambient_temperature_c": temperature,
                    "step_index": 2,
                    "positive_residual_growth_v": values[1],
                },
            ]
        )
    derived = derive_temperature_guards(voltage, pd.DataFrame(rows))["guards"]
    assert derived["15"]["boot_v"] == pytest.approx(0.0371612309890537)
    assert derived["15"]["running_v"] == pytest.approx(0.014098037646160801)
    assert derived["25"]["boot_v"] == pytest.approx(0.026540820770421)
    assert derived["25"]["running_v"] == pytest.approx(0.0128420547282164)
    assert derived["30"]["boot_v"] == pytest.approx(0.0376479265035423)
    assert derived["30"]["running_v"] == pytest.approx(0.0218918472128131)


def test_metric_contract_cannot_hide_other_temperature_exceedances() -> None:
    rows = pd.DataFrame(
        {
            "ambient_temperature_c": [15, 15, 25, 25, 30, 30],
            "guard_stage": ["boot", "running"] * 3,
            "guard_exceeded": [True, False, False, True, False, False],
        }
    )
    counts = _temperature_stage_counts(rows)
    assert counts["15"] == {
        "boot_exceedance_count": 1,
        "running_exceedance_count": 0,
        "total_exceedance_count": 1,
    }
    assert counts["25"]["running_exceedance_count"] == 1
    assert counts["30"]["total_exceedance_count"] == 0
    assert counts["all_temperatures"] == {
        "boot_exceedance_count": 1,
        "running_exceedance_count": 1,
        "total_exceedance_count": 2,
    }


def test_r2f3_does_not_authorize_r3_or_ann() -> None:
    contract = CONFIG.section("control_contract")
    assert contract["residual_initialization"] == "measured"
    assert contract["allow_zero_residual_fallback"] is False
    assert contract["r3_generation_authorized"] is False
    assert contract["ann_execution_authorized"] is False


def test_new_data_contract_is_96_development_and_48_internal() -> None:
    datasets = CONFIG.section("datasets")
    assert datasets["development_count_per_temperature"] == 48
    assert datasets["internal_validation_count_per_temperature"] == 24
    assert len(datasets["temperatures_c"]) == 2
    assert CONFIG.section("validation_contract")[
        "expected_total_validation_trajectory_count"
    ] == 296

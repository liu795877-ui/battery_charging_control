from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from battery_fast_charge.phase7cr2f2_config import load_phase7cr2f2_config
from battery_fast_charge.phase7cr2f2_runner import (
    derive_two_stage_guards,
    guard_for_step,
)
from battery_fast_charge.phase7cr2f_teacher import (
    StrictTeacherSelectionError,
    select_qualified_teacher_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_phase7cr2f2_config(
    ROOT / "configs/phase7cr2f2_two_stage_voltage_guard.yaml"
)


def test_residual_initialization_is_measured_without_zero_fallback() -> None:
    contract = CONFIG.section("control_contract")
    assert contract["residual_initialization"] == "measured"
    assert contract["allow_zero_residual_fallback"] is False


def test_two_stage_guard_switches_at_exact_boundary() -> None:
    guards = {
        "15": 0.014,
        "25": 0.011,
        "30": {"boot_v": 0.035, "running_v": 0.022},
    }
    assert guard_for_step(30.0, 0, guards) == (0.035, "boot")
    assert guard_for_step(30.0, 1, guards) == (0.035, "boot")
    assert guard_for_step(30.0, 2, guards) == (0.022, "running")


def test_guard_derivation_obeys_preregistered_floors_and_margin() -> None:
    voltage = CONFIG.section("voltage_guard")
    development = pd.DataFrame(
        {
            "step_index": [0, 1, 2, 3],
            "positive_residual_growth_v": [0.030, 0.034, 0.010, 0.021],
        }
    )
    guards = derive_two_stage_guards(voltage, development)
    assert guards["30"]["boot_v"] == pytest.approx(0.03493951909904808)
    assert guards["30"]["running_v"] == pytest.approx(0.0218918472128131)
    assert guards["30"]["boot_v"] >= 0.03443951909904808 + 0.0005
    assert guards["30"]["running_v"] >= 0.0218918472128131


def test_r2f_25_001527_mv_event_is_covered_by_boot_floor() -> None:
    minimum_boot_guard = (
        CONFIG.section("voltage_guard")["boot_historical_minimum_30c_v"]
        + CONFIG.section("voltage_guard")["engineering_margin_v"]
    )
    assert minimum_boot_guard > 0.0250015270785617


def test_no_qualified_teacher_candidate_is_strict_failure() -> None:
    candidates = {
        "default": SimpleNamespace(
            optimizer_success=False,
            prediction_feasible=True,
            objective_value=1.0,
            status="failed",
        ),
        "alternative": SimpleNamespace(
            optimizer_success=True,
            prediction_feasible=False,
            objective_value=0.5,
            status="infeasible",
        ),
    }
    with pytest.raises(StrictTeacherSelectionError):
        select_qualified_teacher_candidate(candidates)


def test_internal_validation_is_one_shot_and_cannot_retune() -> None:
    contract = CONFIG.section("validation_contract")
    assert contract["internal_validation_is_one_shot"] is True
    assert contract["internal_validation_may_not_modify_guards"] is True
    assert contract["failed_internal_validation_becomes_permanent_regression"] is True


def test_f2_does_not_authorize_r3_or_ann() -> None:
    contract = CONFIG.section("control_contract")
    assert contract["r3_generation_authorized"] is False
    assert contract["ann_execution_authorized"] is False


def test_new_states_are_frozen_before_rollout_and_hashes_match() -> None:
    data_dir = ROOT / CONFIG.section("output")["data_directory"]
    freeze = json.loads(
        (data_dir / "initial_state_freeze.json").read_text(encoding="utf-8")
    )
    assert freeze["frozen_before_any_closed_loop_rollout"] is True
    assert freeze["not_r3_confirmation_data"] is True
    assert freeze["not_ann_teacher_data"] is True
    for name, record in freeze["files"].items():
        actual = hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        assert actual == record["sha256"]


def test_new_state_sets_are_isolated_and_have_measured_initial_residuals() -> None:
    data_dir = ROOT / CONFIG.section("output")["data_directory"]
    development = pd.read_csv(data_dir / "development_initial_states.csv")
    internal = pd.read_csv(data_dir / "internal_validation_initial_states.csv")
    columns = [
        "initial_soc",
        "initial_polarization_1_v",
        "initial_polarization_2_v",
        "initial_previous_current_a",
    ]
    assert len(development) == 48
    assert len(internal) == 24
    assert set(development.design_seed) == {20260811}
    assert set(internal.design_seed) == {20260812}
    assert set(development.design_candidate_index).isdisjoint(
        set(internal.design_candidate_index)
    )
    development_states = {tuple(row) for row in development[columns].to_numpy()}
    internal_states = {tuple(row) for row in internal[columns].to_numpy()}
    assert development_states.isdisjoint(internal_states)
    for frame in (development, internal):
        assert frame.initial_measured_residual_v.notna().all()
        assert (frame.initial_measured_residual_v != 0.0).all()
        assert (
            frame.initial_state_history_contract
            == "dfn_and_2rc_do_not_share_current_history"
        ).all()

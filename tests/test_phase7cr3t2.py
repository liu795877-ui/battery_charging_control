from __future__ import annotations

from pathlib import Path

import hashlib
import json

import pandas as pd

from battery_fast_charge.phase7cr3t2_config import load_phase7cr3t2_config
from battery_fast_charge.phase7cr3t2_runner import verify_frozen_r3t


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_phase7cr3t2_config(
    ROOT / "configs/phase7cr3t2_dfn_repeatability.yaml"
)


def test_r3t2_scope_and_sequencing_are_preregistered() -> None:
    control = CONFIG.section("control_contract")
    repeatability = CONFIG.section("repeatability_contract")
    assert control["optimized_supervisor_frozen"] is True
    assert control["ann_retraining_authorized"] is False
    assert control["safety_contract_change_authorized"] is False
    assert control["same_state_current_limit_difference_a"] == 0.0
    assert control["level4_entry_authorized"] is False
    assert repeatability["confirmation_may_not_modify_tolerances"] is True
    assert repeatability["expected_development_pair_count"] == 16
    assert repeatability["expected_confirmation_pair_count"] == 120


def test_r3t_speed_pass_and_repeatability_failure_are_frozen() -> None:
    result = verify_frozen_r3t(CONFIG, ROOT)
    assert result["r3t_failed_checks"] == ["closed_loop_current_exact"]
    assert result["same_state_current_limit_difference_a"] == 0.0
    assert all(record["matched"] for record in result["records"].values())


def test_development_and_confirmation_designs_are_isolated() -> None:
    designs = CONFIG.section("datasets")["designs"]
    assert designs["development"][15]["seed"] == 20261201
    assert designs["development"][30]["seed"] == 20261202
    assert designs["confirmation"][15]["seed"] == 20261203
    assert designs["confirmation"][30]["seed"] == 20261204
    assert set(CONFIG.section("repeatability_contract")["confirmation_seeds"]) == {
        22,
        42,
        73,
        101,
        137,
    }


def test_new_state_files_are_frozen_before_any_rollout() -> None:
    data_dir = ROOT / CONFIG.section("output")["data_directory"]
    freeze_path = data_dir / "initial_state_freeze.json"
    if not freeze_path.exists():
        return
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert freeze["status"] == "states_frozen_before_any_rollout"
    assert freeze["development_confirmation_isolated"] is True
    assert freeze["development_started"] is False
    assert freeze["confirmation_started"] is False
    development = pd.concat(
        [
            pd.read_csv(data_dir / f"development_initial_states_{temperature}c.csv")
            for temperature in (15, 30)
        ],
        ignore_index=True,
    )
    confirmation = pd.concat(
        [
            pd.read_csv(data_dir / f"confirmation_initial_states_{temperature}c.csv")
            for temperature in (15, 30)
        ],
        ignore_index=True,
    )
    assert len(development) == 16
    assert len(confirmation) == 24
    columns = [
        "initial_soc",
        "initial_polarization_1_v",
        "initial_polarization_2_v",
        "initial_previous_current_a",
    ]
    assert {tuple(row) for row in development[columns].to_numpy()}.isdisjoint(
        {tuple(row) for row in confirmation[columns].to_numpy()}
    )
    for name, record in freeze["files"].items():
        assert hashlib.sha256((data_dir / name).read_bytes()).hexdigest() == record[
            "sha256"
        ]


def test_repeatability_tolerances_are_frozen_before_confirmation() -> None:
    data_dir = ROOT / CONFIG.section("output")["data_directory"]
    path = data_dir / "frozen_repeatability_tolerances.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "tolerances_frozen_before_confirmation"
    assert payload["development_summary"]["pair_count"] == 16
    assert payload["development_summary"]["all_steps_paired"] is True
    assert payload["development_summary"]["decision_mismatch_count"] == 0
    assert payload["frozen_tolerances"]["current_difference_a"] == (
        2.975499059073627e-06
    )
    assert payload["frozen_tolerances"]["soc_difference"] == (
        9.249783967073655e-09
    )
    assert payload["frozen_tolerances"]["temperature_difference_c"] == (
        7.572148439687453e-06
    )
    assert payload["confirmation_started"] is False
    assert payload["confirmation_used_for_retuning"] is False
    assert payload["level4_entered"] is False


def test_confirmation_strictly_passes_and_authorizes_level4() -> None:
    result_dir = ROOT / CONFIG.section("output")["result_directory"]
    metrics_path = result_dir / "metrics.json"
    manifest_path = result_dir / "freeze_manifest.json"
    if not metrics_path.exists() or not manifest_path.exists():
        return

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = metrics["confirmation_summary"]
    physical = metrics["physical_summary"]
    timing = metrics["timing_summary"]
    decision = metrics["decision"]

    assert metrics["status"] == "strict_passed"
    assert metrics["success"] is True
    assert metrics["failed_checks"] == []
    assert summary["pair_count"] == 120
    assert summary["all_steps_paired"] is True
    assert summary["decision_mismatch_count"] == 0
    assert summary["maximum_current_difference_a"] <= (
        metrics["tolerance_freeze"]["frozen_tolerances"]["current_difference_a"]
    )
    assert summary["maximum_soc_difference"] <= (
        metrics["tolerance_freeze"]["frozen_tolerances"]["soc_difference"]
    )
    assert summary["maximum_temperature_difference_c"] <= (
        metrics["tolerance_freeze"]["frozen_tolerances"][
            "temperature_difference_c"
        ]
    )
    assert physical["trajectory_count"] == 240
    assert physical["target_reach_fraction"] == 1.0
    assert physical["maximum_voltage_v"] <= 4.200001
    assert physical["maximum_temperature_c"] <= 35.0
    assert physical["guard_exceedance_count"] == 0
    assert physical["empty_voltage_slew_count"] == 0
    assert physical["empty_thermal_slew_count"] == 0
    assert physical["prediction_infeasible_count"] == 0
    assert physical["sustained_oscillation_count"] == 0
    assert timing["minimum_end_to_end_speedup"] > 100.0
    assert decision["level4_authorized"] is True
    assert decision["level4_entered"] is False
    assert decision["ann_retrained"] is False
    assert decision["safety_contract_changed"] is False
    assert decision["confirmation_used_for_retuning"] is False

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "strict_passed"
    assert manifest["level4_authorized"] is True
    assert manifest["level4_entered"] is False
    for relative, expected in manifest["artifacts"].items():
        path = ROOT / relative
        assert path.exists(), relative
        payload = path.read_bytes()
        if path.suffix in {".py", ".yaml", ".md"}:
            payload = payload.replace(b"\r\n", b"\n")
        actual = hashlib.sha256(payload).hexdigest()
        assert actual == expected, relative

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from battery_fast_charge.phase7d1_config import load_phase7d1_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_phase7d1_config(
    ROOT / "configs/phase7d1_performance_optimization.yaml"
)


def test_level4_1_contract_uses_new_data_and_safety_first_selection() -> None:
    contract = CONFIG.section("optimization_contract")
    assert contract["ann_retraining_authorized"] is False
    assert contract["r3_confirmation_tuning_authorized"] is False
    assert contract["physical_limits_changed"] is False
    assert contract["development_trajectory_count"] == 96
    assert contract["internal_validation_trajectory_count"] == 48
    assert contract["internal_validation_one_shot"] is True
    assert contract["internal_validation_may_not_change_parameters"] is True
    assert contract["primary_selection_rule"] == (
        "safety_first_then_minimum_mean_charge_time"
    )
    assert contract["independent_confirmation_created"] is False


def test_level4_1_designs_are_independently_seeded() -> None:
    designs = CONFIG.section("datasets")["designs"]
    seeds = {
        designs[role][temperature]["seed"]
        for role in ("development", "internal_validation")
        for temperature in (15, 30)
    }
    assert len(seeds) == 4


def test_level4_1_states_are_frozen_before_rollouts() -> None:
    data_dir = ROOT / CONFIG.section("output")["data_directory"]
    freeze_path = data_dir / "state_freeze.json"
    if not freeze_path.exists():
        return
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert freeze["status"] == "states_frozen_before_any_level4_1_rollout"
    assert freeze["development_internal_validation_isolated"] is True
    assert freeze["r3t2_states_excluded"] is True
    assert freeze["r3_confirmation_used_for_tuning"] is False
    assert freeze["rollouts_started"] is False
    assert freeze["optimization_parameters_selected"] is False
    assert freeze["internal_validation_consumed"] is False
    assert freeze["independent_confirmation_created"] is False
    total = 0
    for name, record in freeze["files"].items():
        path = data_dir / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        frame = pd.read_csv(path)
        assert len(frame) == record["trajectory_count"]
        assert record["zero_residual_count"] == 0
        total += len(frame)
    assert total == 144


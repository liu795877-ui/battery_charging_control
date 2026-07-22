from pathlib import Path

import pandas as pd

from battery_fast_charge.phase2rc_runner import _audit, load_phase_two_rc_config


ROOT = Path(__file__).resolve().parents[1]


def test_phase2rc_preserves_frozen_teacher_contract() -> None:
    config = load_phase_two_rc_config(ROOT / "configs" / "phase2rc_prospective_control_memory.yaml")
    assert config.trajectory_count == 240
    assert config.trajectory_steps == 8
    assert config.minimum_acceptance_fraction == 0.70


def test_phase2rc_audit_contains_native_memory_comparison() -> None:
    source = pd.read_csv(ROOT / "data" / "phase2r_sufficiency_audit" / "rolling_teacher_control_memory.csv").head(80).copy()
    source["previous_plan_range_a"] = source[["previous_plan_first_a", "previous_plan_last_a"]].max(axis=1) - source[["previous_plan_first_a", "previous_plan_last_a"]].min(axis=1)
    source["previous_plan_available"] = (source.step_index > 0).astype(float)
    table, summary = _audit(source, load_phase_two_rc_config(ROOT / "configs" / "phase2rc_prospective_control_memory.yaml"))
    assert "plus_previous_plan_full_summary" in set(table.feature_set)
    assert "native_memory_locally_sufficient" in summary

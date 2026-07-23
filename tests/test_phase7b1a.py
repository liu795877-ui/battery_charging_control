import hashlib
import json
from pathlib import Path

import pandas as pd

from battery_fast_charge.phase7b1a_config import load_phase7b1a_config
from battery_fast_charge.phase7b1a_runner import _verify_frozen


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase7b1a_voltage_mismatch_audit.yaml"


def test_phase7b1a_contract_and_confirmation_set_are_frozen() -> None:
    config = load_phase7b1a_config(CONFIG)
    verification = _verify_frozen(config, ROOT)
    assert all(item["matched"] for item in verification.values())
    freeze = json.loads(
        (
            ROOT
            / "data/phase7b1a_voltage_mismatch/confirmation_freeze.json"
        ).read_text(encoding="utf-8")
    )
    path = ROOT / freeze["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == freeze["sha256"]
    assert freeze["trajectory_count"] == 24
    assert freeze["created_before_phase7b1b"] is True
    assert freeze["not_teacher_data"] is True


def test_phase7b1a_one_step_voltage_and_slew_intervals_are_nonempty() -> None:
    audit = pd.read_csv(
        ROOT
        / "data/phase7b1a_voltage_mismatch/voltage_residual_step_audit.csv"
    )
    assert len(audit) == 16024
    assert not audit.voltage_slew_conflict.astype(bool).any()
    assert audit.voltage_slew_feasibility_margin_a.min() > 0.0
    assert (audit.one_step_safe_current_a >= audit.slew_lower_a - 1.0e-12).all()
    assert (audit.one_step_safe_current_a <= audit.slew_upper_a + 1.0e-12).all()


def test_phase7b1a_proactive_thresholds_allow_maximum_slew_braking() -> None:
    timing = pd.read_csv(
        ROOT / "data/phase7b1a_voltage_mismatch/threshold_timing.csv"
    )
    proactive = timing[
        timing.entered.astype(bool) & (timing.threshold_v < 4.20 - 1.0e-12)
    ]
    assert len(proactive) == 216
    assert proactive.maximum_slew_braking_in_time.astype(bool).all()
    metrics = json.loads(
        (
            ROOT / "outputs/phase7b1a_voltage_mismatch/metrics.json"
        ).read_text(encoding="utf-8")
    )
    assert metrics["decision"]["phase7b1a_success"] is True
    assert metrics["decision"]["proceed_to_one_step_phase7b1b"] is True

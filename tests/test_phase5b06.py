from types import SimpleNamespace
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from battery_fast_charge.mpc import ConstrainedMPC, ReducedState
from battery_fast_charge.phase5b06_runner import feasibility_count_table


def test_constraint_slacks_are_reported_per_constraint_family() -> None:
    controller = object.__new__(ConstrainedMPC)
    controller.config = SimpleNamespace(
        constraints=SimpleNamespace(
            mpc_maximum_voltage_v=4.18,
            mpc_maximum_temperature_c=34.5,
            maximum_current_change_a_per_step=2.0,
        ),
        battery=SimpleNamespace(target_soc=0.8),
    )
    state = ReducedState(0.2, 0.0, 0.0, 25.0, 25.0, 0.0)
    prediction = {
        "voltage_v": np.asarray([4.17, 4.20]),
        "temperature_c": np.asarray([34.0, 35.0]),
        "soc": np.asarray([0.79, 0.81]),
    }
    slacks = controller._constraint_slacks(state, np.asarray([3.0, 0.0]), prediction)
    assert np.isclose(slacks["slack_voltage_v"], 0.02)
    assert np.isclose(slacks["slack_temperature_c"], 0.5)
    assert np.isclose(slacks["slack_soc"], 0.01)
    assert np.isclose(slacks["slack_current_change_a"], 1.0)


def test_phase5b06_feasible_count_is_consistent_across_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = pd.read_csv(root / "data/phase5b06_contract_audit/paired_summary.csv")
    counts = pd.read_csv(root / "data/phase5b06_contract_audit/feasibility_counts.csv")
    metrics = json.loads((root / "outputs/metrics/phase5b06_metrics.json").read_text(encoding="utf-8"))
    report = (root / "outputs/phase5b06_report.md").read_text(encoding="utf-8")
    csv_count = int(summary.loc[summary.controller == "recovery_mpc", "operational_feasible"].sum())
    count_table = feasibility_count_table(summary)
    pd.testing.assert_frame_equal(counts, count_table, check_dtype=False)
    report_count = int(re.search(r"recovery_operational_feasible_count: (\d+)", report).group(1))
    assert metrics["canonical_feasibility_field"] == "operational_feasible"
    assert csv_count == metrics["recovery_feasible_count"] == report_count == 5

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from battery_fast_charge.phase2r_config import load_phase_two_r_config
from battery_fast_charge.phase2r_model_audit import simulate_single_node_thermal
from battery_fast_charge.phase2r_state_audit import run_state_sufficiency_audit


CONFIG = Path("configs/phase2r_model_and_state_sufficiency.yaml")


def test_phase2r_contract_covers_requested_grid() -> None:
    config = load_phase_two_r_config(CONFIG)
    assert config.model_audit.initial_soc_points == (0.60, 0.65, 0.70, 0.75, 0.80)
    assert config.model_audit.temperatures_c == (15.0, 25.0, 30.0)
    assert config.model_audit.prediction_horizons_s == (5.0, 25.0, 300.0)
    assert config.model_audit.parameter_anchor_soc_points == (0.60, 0.70, 0.80)


def test_single_node_thermal_relaxes_to_ambient_without_heat() -> None:
    time = np.arange(0.0, 101.0, 5.0)
    temperature = simulate_single_node_thermal(time, np.zeros_like(time), 30.0, 25.0, 50.0, 2.0)
    assert temperature[0] == 30.0
    assert np.all(np.diff(temperature) < 0.0)
    assert temperature[-1] > 25.0


def test_previous_current_reduces_synthetic_conditional_variance() -> None:
    random = np.random.default_rng(17)
    count = 400
    previous = random.uniform(0.0, 10.0, count)
    frame = pd.DataFrame(
        {
            "state_soc": random.uniform(0.2, 0.7, count),
            "state_polarization_fast_v": random.uniform(0.0, 0.08, count),
            "state_polarization_slow_v": random.uniform(0.0, 0.08, count),
            "state_average_temperature_c": random.uniform(25.0, 32.0, count),
            "state_core_temperature_c": random.uniform(25.0, 32.0, count),
            "state_surface_temperature_c": random.uniform(25.0, 32.0, count),
            "state_previous_current_a": previous,
            "control_block_phase": random.integers(0, 5, count),
            "mode_voltage_active": 0.0,
            "mode_temperature_active": 0.0,
            "mode_slew_active": 0.0,
            "previous_plan_first_a": previous,
            "previous_plan_mean_a": previous,
            "previous_plan_last_a": previous,
            "target_cap_active": 0.0,
            "teacher_current_a": np.clip(previous + 1.0, 0.0, 10.0),
        }
    )
    metrics, _, _ = run_state_sufficiency_audit(frame, load_phase_two_r_config(CONFIG))
    variance = metrics.set_index("feature_set")["mean_conditional_variance_a2"]
    assert variance["current_dnn_5_plus_previous_current"] < variance["electrothermal_4"]

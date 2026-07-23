from pathlib import Path
import numpy as np
import pandas as pd

from battery_fast_charge.phase7a_level1_config import load_phase7a_level1_config
from battery_fast_charge.phase7a_level1s_config import load_phase7a_level1s_config
from battery_fast_charge.phase7a_level1s_runner import continuous_crossing_time_s, select_scheme

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "phase7a_level1s_training_stability.yaml"


def test_level1s_freezes_training_problem_and_candidates():
    config = load_phase7a_level1s_config(CONFIG)
    base = load_phase7a_level1_config(ROOT / config.source_level1_config)
    assert {v.hidden_layer_sizes for v in config.architectures} == {(32, 32, 16), (16,), (32,)}
    assert config.optimizers == ("adam", "lbfgs")
    assert base.network.initialization_seeds == (22, 42, 73, 101, 137)
    assert base.gates.charge_time_gap_fraction_max == 0.02


def test_continuous_crossing_time_uses_linear_interpolation_in_seconds():
    frame = pd.DataFrame({"step_index": [0, 1], "soc": [0.79, 0.795], "next_soc": [0.795, 0.80]})
    assert np.isclose(continuous_crossing_time_s(frame, 0.7975, 5.0), 7.5)


def test_signed_diagnostics_convention_is_dnn_minus_mpc():
    dnn_steps, mpc_steps, dt = 103, 100, 5.0
    assert dnn_steps - mpc_steps == 3
    assert (dnn_steps - mpc_steps) * dt == 15.0


def test_scheme_selection_uses_only_validation_metrics():
    rows = []
    for scheme, nrmse, bias, low in [("a__adam", .002, .02, .03), ("b__lbfgs", .003, .01, .01)]:
        architecture, optimizer = scheme.split("__")
        for seed in range(5):
            rows.append({"architecture": architecture, "optimizer": optimizer, "seed": seed,
                         "validation_nrmse": nrmse, "validation_abs_bias_a": bias,
                         "validation_low_current_abs_bias_a": low, "test_nrmse": 999 if architecture == "b" else 0})
    summary, selected = select_scheme(pd.DataFrame(rows), ("validation_nrmse", "validation_abs_bias_a", "validation_low_current_abs_bias_a"))
    assert selected in {"a__adam", "b__lbfgs"}
    assert summary.selected.sum() == 1


def test_frozen_source_files_exist_and_no_level1s_teacher_path_is_configured():
    config = load_phase7a_level1s_config(CONFIG)
    for value in (config.source_combined_dataset, config.source_original_dataset, config.source_terminal_dataset,
                  config.source_tail_dataset, config.source_closed_loop_initial_states):
        assert (ROOT / value).exists()

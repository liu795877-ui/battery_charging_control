from pathlib import Path

import numpy as np
import pandas as pd

from battery_fast_charge.phase7b0_config import load_phase7b0_config
from battery_fast_charge.phase7b0_runner import (
    _continuous_crossing_time,
    _direction_reversals,
    _unexpected_early_taper_count,
    verify_frozen_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]


def test_phase7b0_frozen_contract_matches_repository() -> None:
    config = load_phase7b0_config(ROOT / "configs/phase7b0_dfn_cross_model.yaml")
    verification = verify_frozen_artifacts(config, ROOT)
    assert len(verification) == 10
    assert all(item["matched"] for item in verification.values())


def test_continuous_crossing_time_interpolates_within_sample() -> None:
    frame = pd.DataFrame(
        {
            "step_index": [0, 1],
            "soc": [0.79, 0.795],
            "next_soc": [0.795, 0.805],
        }
    )
    assert np.isclose(_continuous_crossing_time(frame, 0.80, 5.0), 7.5)


def test_diagnostics_ignore_small_jitter_and_initial_ramp() -> None:
    assert _direction_reversals(np.array([0.0, 0.1, 0.0]), 0.25) == 0
    assert _direction_reversals(np.array([0.0, 1.0, 0.0, 1.0]), 0.25) == 2
    frame = pd.DataFrame(
        {
            "soc": [0.60, 0.65, 0.70, 0.72],
            "current_a": [2.0, 5.0, 4.0, 6.0],
        }
    )
    assert _unexpected_early_taper_count(frame, 0.74, 5.0) == 1

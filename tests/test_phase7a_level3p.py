from pathlib import Path

import numpy as np

from battery_fast_charge.phase7a_level3p_config import load_phase7a_level3p_config
from battery_fast_charge.phase7a_level3p_runner import (
    project_current,
    verify_frozen_artifacts,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "phase7a_level3p_projection.yaml"


def test_level3p_projection_matches_registered_formula():
    safe, lower, upper = project_current(9.5, 7.0)
    assert (lower, upper, safe) == (5.0, 9.0, 9.0)
    safe, lower, upper = project_current(0.2, 3.0)
    assert (lower, upper, safe) == (1.0, 5.0, 1.0)
    safe, lower, upper = project_current(5.2, 4.0)
    assert (lower, upper, safe) == (2.0, 6.0, 5.2)


def test_level3p_projection_always_enforces_current_and_slew_bounds():
    for previous in np.linspace(0.0, 10.0, 41):
        for raw in np.linspace(-2.0, 12.0, 57):
            safe, lower, upper = project_current(raw, previous)
            assert 0.0 <= lower <= safe <= upper <= 10.0
            assert abs(safe - previous) <= 2.0 + 1e-12


def test_level3p_frozen_artifact_contract_matches_level3_outputs():
    config = load_phase7a_level3p_config(CONFIG)
    verification = verify_frozen_artifacts(config, ROOT)
    assert len(verification) == 13
    assert all(item["matched"] for item in verification.values())


def test_level3p_explicitly_does_not_add_level4_factors():
    raw = CONFIG.read_text(encoding="utf-8").lower()
    assert "temperature" not in raw
    assert "dfn" not in raw
    assert "disturbance" not in raw
    assert "phase5a" not in raw

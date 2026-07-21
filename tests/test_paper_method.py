import numpy as np

from battery_fast_charge.paper_method import (
    generate_initial_state_design,
    hammersley_points,
)
from battery_fast_charge.phase6_config import FEATURE_NAMES, load_phase_six_config


def test_hammersley_points_are_deterministic_and_inside_unit_cube() -> None:
    first = hammersley_points(20, 5)
    second = hammersley_points(20, 5)

    assert np.array_equal(first, second)
    assert first.shape == (20, 5)
    assert np.all((first >= 0.0) & (first <= 1.0))


def test_initial_state_design_has_paper_scale_and_declared_bounds() -> None:
    config = load_phase_six_config("configs/phase6_paper_method_validation.yaml")
    design = generate_initial_state_design(config)

    assert len(design) == 500
    assert design["initial_state_id"].is_unique
    assert set(design["sampling_method"]) == {"hammersley", "boundary_factorial"}
    for feature in FEATURE_NAMES:
        low, high = config.paper_method.state_ranges[feature]
        assert design[feature].between(low, high, inclusive="both").all()

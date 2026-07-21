import pandas as pd

from battery_fast_charge.phase6c2_config import load_phase_six_c2_config
from battery_fast_charge.phase6c2_runner import (
    DAGGER_SOURCE,
    TARGETED_SOURCE,
    assign_new_splits,
    generate_targeted_boundary_design,
)


def test_phase6c2_targeted_design_has_declared_boundary_ranges() -> None:
    config = load_phase_six_c2_config("configs/phase6c2_targeted_teacher_data.yaml")
    design = generate_targeted_boundary_design(config)

    assert len(design) == 500
    assert set(design["sampling_method"]) == {TARGETED_SOURCE}
    for feature, (low, high) in config.sampling.targeted_ranges.items():
        assert design[feature].between(low, high, inclusive="both").all()


def test_phase6c2_new_split_assignment_keeps_sources_in_both_sets() -> None:
    config = load_phase_six_c2_config("configs/phase6c2_targeted_teacher_data.yaml")
    design = pd.DataFrame(
        {
            "initial_state_id": [f"target-{i}" for i in range(10)] + [f"dagger-{i}" for i in range(10)],
            "sampling_method": [TARGETED_SOURCE] * 10 + [DAGGER_SOURCE] * 10,
        }
    )
    mapping = assign_new_splits(design, config)
    design["split"] = design["initial_state_id"].map(mapping)

    assert set(design["split"]) == {"phase6c_train", "phase6c_validation"}
    assert design.groupby("sampling_method")["split"].nunique().eq(2).all()

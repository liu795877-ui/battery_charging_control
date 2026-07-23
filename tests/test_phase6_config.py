from dataclasses import replace

import pytest

from battery_fast_charge.phase6_config import load_phase_six_config


def test_phase6_configuration_matches_paper_method_scope() -> None:
    config = load_phase_six_config("configs/phase6_paper_method_validation.yaml")

    assert config.paper_method.initial_state_count == 500
    assert config.paper_method.trajectory_steps == 8
    assert config.success_criteria.maximum_nominal_current_nrmse == 0.01
    assert config.network.candidate_hidden_layer_sizes[0] == (7, 5, 3)


def test_phase6_rejects_missing_test_split() -> None:
    config = load_phase_six_config("configs/phase6_paper_method_validation.yaml")
    invalid = replace(
        config,
        paper_method=replace(
            config.paper_method,
            train_fraction=0.9,
            validation_fraction=0.1,
        ),
    )

    from battery_fast_charge.phase6_config import _validate

    with pytest.raises(ValueError):
        _validate(invalid)

import pytest

from battery_fast_charge.phase6b_config import load_phase_six_b_config


def test_phase6b_configuration_uses_larger_paper_style_scope() -> None:
    config = load_phase_six_b_config("configs/phase6b_dnn_failure_diagnosis.yaml")

    assert config.paper_method.initial_state_count == 1000
    assert (32, 32, 16) in config.network.candidate_hidden_layer_sizes
    assert (64, 64, 32) in config.network.candidate_hidden_layer_sizes
    assert config.diagnostics.slew_margin_close_a == 0.10


def test_phase6b_rejects_small_datasets() -> None:
    with pytest.raises(ValueError):
        load_phase_six_b_config("configs/phase6_paper_method_validation.yaml")

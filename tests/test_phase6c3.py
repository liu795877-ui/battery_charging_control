import numpy as np

from battery_fast_charge.ann_model import TinyANN
from battery_fast_charge.phase6_config import FEATURE_NAMES
from battery_fast_charge.phase6c3_config import load_phase_six_c3_config
from battery_fast_charge.phase6c3_runner import structured_current_prediction


def test_phase6c3_configuration_keeps_five_seeds_and_strict_gate() -> None:
    config = load_phase_six_c3_config("configs/phase6c3_structured_dnn_comparison.yaml")

    assert len(config.network.initialization_seeds) >= 5
    assert config.network.hidden_layer_sizes == (16, 16)
    assert config.success_criteria.maximum_nominal_current_nrmse == 0.01


def test_structured_prediction_satisfies_slew_by_construction() -> None:
    model = TinyANN(
        feature_names=FEATURE_NAMES,
        feature_mean=np.zeros(5),
        feature_scale=np.ones(5),
        target_mean=0.0,
        target_scale=1.0,
        weights=(np.zeros((5, 1)), np.zeros((1, 1))),
        biases=(np.array([20.0]), np.array([20.0])),
    )
    features = np.array([[0.2, 0.01, 0.01, 25.0, 3.0]])
    current = structured_current_prediction(model, features, 2.0)

    assert np.abs(current[0] - features[0, -1]) <= 2.0

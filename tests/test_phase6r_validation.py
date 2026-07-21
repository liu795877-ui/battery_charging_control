from types import SimpleNamespace

import numpy as np

from battery_fast_charge.ann_model import TinyANN
from battery_fast_charge.mpc import ReducedState
from battery_fast_charge.phase6r_config import ROLLING_STATE_FEATURES
from battery_fast_charge.phase6r_validation import ann_current, controller_features


def test_full_state_features_keep_core_surface_and_ambient_separate() -> None:
    state = ReducedState(0.2, 0.01, 0.02, 31.0, 27.0, 4.0)
    model = SimpleNamespace(
        config=SimpleNamespace(battery=SimpleNamespace(ambient_temperature_c=25.0)),
        average_temperature=lambda value: 29.0,
    )
    features = controller_features(model, state, ROLLING_STATE_FEATURES)
    np.testing.assert_allclose(features, [0.2, 0.01, 0.02, 31.0, 27.0, 25.0, 4.0])


def test_feasible_closed_loop_output_respects_current_and_slew() -> None:
    state = ReducedState(0.2, 0.0, 0.0, 25.0, 25.0, 9.5)
    model = SimpleNamespace(
        config=SimpleNamespace(battery=SimpleNamespace(ambient_temperature_c=25.0)),
        average_temperature=lambda value: 25.0,
    )
    phase3 = SimpleNamespace(
        constraints=SimpleNamespace(maximum_current_a=10.0, maximum_current_change_a_per_step=2.0)
    )
    ann = TinyANN(
        feature_names=ROLLING_STATE_FEATURES,
        feature_mean=np.zeros(7),
        feature_scale=np.ones(7),
        target_mean=0.0,
        target_scale=1.0,
        weights=(np.zeros((7, 1)), np.ones((1, 1))),
        biases=(np.zeros(1), np.zeros(1)),
    )
    current, _ = ann_current("full_state_feasible_interval", ann, model, state, phase3)
    assert 7.5 <= current <= 10.0

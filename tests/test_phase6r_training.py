import numpy as np

from battery_fast_charge.ann_model import TinyANN
from battery_fast_charge.phase6r_config import ROLLING_STATE_FEATURES
from battery_fast_charge.phase6r_training import (
    feasible_current_from_latent,
    feasible_latent_target,
)


def test_phase6r_feasible_interval_respects_constraints() -> None:
    previous = np.array([0.0, 4.0, 9.5])
    model = TinyANN(
        feature_names=ROLLING_STATE_FEATURES,
        feature_mean=np.zeros(7),
        feature_scale=np.ones(7),
        target_mean=0.0,
        target_scale=1.0,
        weights=(np.zeros((7, 1)), np.ones((1, 1))),
        biases=(np.zeros(1), np.zeros(1)),
    )
    features = np.zeros((3, 7))
    features[:, -1] = previous
    prediction = feasible_current_from_latent(model, features, 6, 10.0, 2.0)

    assert np.all((prediction >= 0.0) & (prediction <= 10.0))
    assert np.all(np.abs(prediction - previous) <= 2.0)


def test_phase6r_latent_target_is_finite_at_active_boundaries() -> None:
    latent = feasible_latent_target(
        np.array([2.0, 10.0]),
        np.array([0.0, 9.0]),
        maximum_current_a=10.0,
        maximum_change_a=2.0,
    )

    assert np.isfinite(latent).all()

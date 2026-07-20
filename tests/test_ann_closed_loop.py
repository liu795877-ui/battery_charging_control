import numpy as np

from battery_fast_charge.ann_closed_loop import ann_features
from battery_fast_charge.ann_model import TinyANN
from battery_fast_charge.mpc import ReducedState


class _AverageTemperatureModel:
    @staticmethod
    def average_temperature(state: ReducedState) -> float:
        return 0.8 * state.core_temperature_c + 0.2 * state.surface_temperature_c


def test_ann_feature_order_matches_training_contract() -> None:
    """闭环输入顺序必须与教师数据训练列完全一致。"""
    state = ReducedState(0.4, 0.01, 0.02, 30.0, 28.0, 4.0)
    values = ann_features(_AverageTemperatureModel(), state)

    assert np.allclose(values, [0.4, 0.01, 0.02, 29.6, 4.0])


def test_ann_output_is_bounded_but_unclipped_value_is_auditable() -> None:
    """部署电流受0到10 A限制，同时保留原始网络输出用于诊断。"""
    model = TinyANN(
        feature_names=("x",),
        feature_mean=np.array([0.0]),
        feature_scale=np.array([1.0]),
        target_mean=12.0,
        target_scale=1.0,
        weights=(np.array([[0.0]]),),
        biases=(np.array([0.0]),),
    )

    assert model.predict_unclipped(np.array([0.0])) == 12.0
    assert model.predict(np.array([0.0])) == 10.0

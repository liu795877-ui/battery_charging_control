from pathlib import Path

import numpy as np

from battery_fast_charge.ann_model import TinyANN


def _example_model() -> TinyANN:
    return TinyANN(
        feature_names=("a", "b"),
        feature_mean=np.array([1.0, 2.0]),
        feature_scale=np.array([2.0, 4.0]),
        target_mean=5.0,
        target_scale=2.0,
        weights=(np.array([[1.0], [-1.0]]), np.array([[0.5]])),
        biases=(np.array([0.0]), np.array([0.0])),
    )


def test_tiny_ann_matches_manual_forward_pass() -> None:
    """标准化、tanh隐藏层和反标准化必须按约定顺序执行。"""
    model = _example_model()
    features = np.array([3.0, 2.0])
    expected = 5.0 + 2.0 * 0.5 * np.tanh(1.0)

    assert np.isclose(model.predict_unclipped(features), expected)
    assert model.parameter_count == 5


def test_tiny_ann_npz_round_trip(tmp_path: Path) -> None:
    """保存后的NumPy部署模型必须与内存模型逐位一致。"""
    model = _example_model()
    path = model.save(tmp_path / "tiny_ann.npz")
    loaded = TinyANN.load(path)
    features = np.array([[1.0, 2.0], [3.0, 6.0]])

    assert loaded.feature_names == model.feature_names
    assert np.allclose(loaded.predict(features), model.predict(features))

"""训练小型ANN并导出为只依赖NumPy的可审计推理模型。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .phase4_config import PhaseFourAConfig


@dataclass(frozen=True)
class TinyANN:
    """包含标准化参数和全连接层权重的小型ANN。"""

    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: float
    target_scale: float
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    minimum_current_a: float = 0.0
    maximum_current_a: float = 10.0

    def predict_unclipped(self, features: np.ndarray) -> np.ndarray:
        """计算未经电流边界裁剪的网络输出，输入形状为(N,5)。"""
        values = np.asarray(features, dtype=float)
        one_dimensional = values.ndim == 1
        if one_dimensional:
            values = values.reshape(1, -1)
        if values.shape[1] != len(self.feature_names):
            raise ValueError("ANN输入列数与训练特征数量不一致。")
        hidden = (values - self.feature_mean) / self.feature_scale
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            hidden = np.tanh(hidden @ weight + bias)
        normalized_output = hidden @ self.weights[-1] + self.biases[-1]
        output = normalized_output.reshape(-1) * self.target_scale + self.target_mean
        return output[0] if one_dimensional else output

    def predict(self, features: np.ndarray) -> np.ndarray:
        """输出0到10 A内的期望充电电流；物理安全仍由外部过滤器负责。"""
        return np.clip(
            self.predict_unclipped(features),
            self.minimum_current_a,
            self.maximum_current_a,
        )

    @property
    def parameter_count(self) -> int:
        """返回权重和偏置的可训练参数总数。"""
        return int(
            sum(weight.size + bias.size for weight, bias in zip(self.weights, self.biases))
        )

    def save(self, path: str | Path) -> Path:
        """保存非可执行NPZ权重，避免pickle模型带来的任意代码加载风险。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "feature_names": np.asarray(self.feature_names),
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "target_mean": np.asarray([self.target_mean]),
            "target_scale": np.asarray([self.target_scale]),
            "minimum_current_a": np.asarray([self.minimum_current_a]),
            "maximum_current_a": np.asarray([self.maximum_current_a]),
            "layer_count": np.asarray([len(self.weights)], dtype=int),
        }
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            payload[f"weight_{index}"] = weight
            payload[f"bias_{index}"] = bias
        np.savez(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TinyANN":
        """从NPZ恢复模型并拒绝对象反序列化。"""
        with np.load(Path(path), allow_pickle=False) as payload:
            count = int(payload["layer_count"][0])
            return cls(
                feature_names=tuple(str(value) for value in payload["feature_names"]),
                feature_mean=payload["feature_mean"].astype(float),
                feature_scale=payload["feature_scale"].astype(float),
                target_mean=float(payload["target_mean"][0]),
                target_scale=float(payload["target_scale"][0]),
                weights=tuple(payload[f"weight_{index}"].astype(float) for index in range(count)),
                biases=tuple(payload[f"bias_{index}"].astype(float) for index in range(count)),
                minimum_current_a=float(payload["minimum_current_a"][0]),
                maximum_current_a=float(payload["maximum_current_a"][0]),
            )


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """计算单位明确的电流回归误差。"""
    if len(target) == 0:
        return {
            "sample_count": 0,
            "mae_a": float("nan"),
            "rmse_a": float("nan"),
            "maximum_absolute_error_a": float("nan"),
            "bias_a": float("nan"),
            "r2": float("nan"),
        }
    error = np.asarray(prediction) - np.asarray(target)
    denominator = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "sample_count": int(len(target)),
        "mae_a": float(np.mean(np.abs(error))),
        "rmse_a": float(np.sqrt(np.mean(error**2))),
        "maximum_absolute_error_a": float(np.max(np.abs(error))),
        "bias_a": float(np.mean(error)),
        "r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0.0 else 0.0,
    }


def train_tiny_ann(
    dataset: pd.DataFrame,
    config: PhaseFourAConfig,
    sample_weight_column: str | None = None,
) -> tuple[TinyANN, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """只用训练轨迹拟合标准化和权重，用验证轨迹选择超参数。

    可选样本权重只作用于训练集；验证和测试指标始终逐唯一状态等权计算。
    """
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    required = set(config.features) | {
        config.target,
        "split",
        "trajectory_id",
        "active_temperature_constraint",
        "active_voltage_constraint",
    }
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"教师数据缺少列：{missing}")
    if dataset.groupby("trajectory_id")["split"].nunique().max() != 1:
        raise ValueError("检测到同一轨迹跨越多个数据集合，禁止训练以避免数据泄漏。")
    if "teacher_accepted" in dataset and not dataset["teacher_accepted"].astype(bool).all():
        raise ValueError("教师数据包含未通过阶段3B接受规则的标签。")
    if sample_weight_column is not None:
        if sample_weight_column not in dataset:
            raise ValueError(f"教师数据缺少样本权重列：{sample_weight_column}")
        if not np.isfinite(dataset[sample_weight_column]).all() or (
            dataset[sample_weight_column] <= 0.0
        ).any():
            raise ValueError("训练样本权重必须是有限正数。")
    train = dataset[dataset["split"] == "train"]
    validation = dataset[dataset["split"] == "validation"]
    test = dataset[dataset["split"] == "test"]
    if any(frame.empty for frame in (train, validation, test)):
        raise ValueError("训练、验证和测试数据都必须非空。")

    train_weight = (
        train[sample_weight_column].to_numpy(dtype=float)
        if sample_weight_column is not None
        else None
    )
    feature_scaler = StandardScaler().fit(
        train[list(config.features)], sample_weight=train_weight
    )
    target_scaler = StandardScaler().fit(
        train[[config.target]], sample_weight=train_weight
    )
    x_train = feature_scaler.transform(train[list(config.features)])
    y_train = target_scaler.transform(train[[config.target]]).reshape(-1)
    x_validation = feature_scaler.transform(validation[list(config.features)])
    y_validation = validation[config.target].to_numpy(dtype=float)

    selection_records: list[dict[str, Any]] = []
    candidates: list[tuple[float, int, Any]] = []
    for alpha in config.network.regularization_candidates:
        for seed in config.network.initialization_seeds:
            estimator = MLPRegressor(
                hidden_layer_sizes=config.network.hidden_layer_sizes,
                activation=config.network.activation,
                solver=config.network.solver,
                alpha=alpha,
                max_iter=config.network.maximum_iterations,
                tol=config.network.convergence_tolerance,
                random_state=seed,
            )
            estimator.fit(x_train, y_train, sample_weight=train_weight)
            validation_prediction = (
                estimator.predict(x_validation) * float(target_scaler.scale_[0])
                + float(target_scaler.mean_[0])
            )
            validation_metrics = _regression_metrics(
                y_validation, validation_prediction
            )
            selection_records.append(
                {
                    "regularization_alpha": alpha,
                    "initialization_seed": seed,
                    "validation_mae_a": validation_metrics["mae_a"],
                    "validation_rmse_a": validation_metrics["rmse_a"],
                    "optimization_iterations": int(estimator.n_iter_),
                    "optimization_loss": float(estimator.loss_),
                }
            )
            candidates.append((validation_metrics["rmse_a"], seed, estimator))

    # 验证RMSE相同时按种子稳定排序，测试集从未参与选择。
    _, selected_seed, selected = min(candidates, key=lambda item: (item[0], item[1]))
    selected_row = min(
        selection_records,
        key=lambda row: (row["validation_rmse_a"], row["initialization_seed"]),
    )
    model = TinyANN(
        feature_names=config.features,
        feature_mean=feature_scaler.mean_.astype(float),
        feature_scale=feature_scaler.scale_.astype(float),
        target_mean=float(target_scaler.mean_[0]),
        target_scale=float(target_scaler.scale_[0]),
        weights=tuple(np.asarray(value, dtype=float) for value in selected.coefs_),
        biases=tuple(np.asarray(value, dtype=float) for value in selected.intercepts_),
    )

    predictions = dataset.copy()
    predictions["ann_unclipped_current_a"] = model.predict_unclipped(
        predictions[list(config.features)].to_numpy(dtype=float)
    )
    predictions["ann_current_a"] = model.predict(
        predictions[list(config.features)].to_numpy(dtype=float)
    )
    predictions["ann_error_a"] = (
        predictions["ann_current_a"] - predictions[config.target]
    )

    split_metrics = {
        split: _regression_metrics(
            frame[config.target].to_numpy(dtype=float),
            frame["ann_current_a"].to_numpy(dtype=float),
        )
        for split, frame in predictions.groupby("split")
    }
    test_predictions = predictions[predictions["split"] == "test"]
    temperature_active = test_predictions[
        test_predictions["active_temperature_constraint"].astype(bool)
    ]
    voltage_active = test_predictions[
        test_predictions["active_voltage_constraint"].astype(bool)
    ]

    # 线性岭回归是最小基线：若它已经更好，就没有使用ANN的证据。
    linear_candidates = []
    for alpha in config.network.regularization_candidates:
        ridge = Ridge(alpha=alpha).fit(
            x_train, y_train, sample_weight=train_weight
        )
        prediction = (
            ridge.predict(x_validation) * float(target_scaler.scale_[0])
            + float(target_scaler.mean_[0])
        )
        metrics = _regression_metrics(y_validation, prediction)
        linear_candidates.append((metrics["rmse_a"], alpha, ridge))
    _, linear_alpha, linear_model = min(linear_candidates, key=lambda item: item[0])
    x_test = feature_scaler.transform(test[list(config.features)])
    linear_test_prediction = (
        linear_model.predict(x_test) * float(target_scaler.scale_[0])
        + float(target_scaler.mean_[0])
    )

    metrics: dict[str, Any] = {
        "architecture": [len(config.features), *config.network.hidden_layer_sizes, 1],
        "parameter_count": model.parameter_count,
        "selected_regularization_alpha": float(
            selected_row["regularization_alpha"]
        ),
        "selected_initialization_seed": int(selected_seed),
        "selected_optimization_iterations": int(
            selected_row["optimization_iterations"]
        ),
        "selected_optimizer_converged": bool(
            selected_row["optimization_iterations"]
            < config.network.maximum_iterations
        ),
        "split_metrics": split_metrics,
        "test_temperature_active_metrics": _regression_metrics(
            temperature_active[config.target].to_numpy(dtype=float),
            temperature_active["ann_current_a"].to_numpy(dtype=float),
        ),
        "test_voltage_active_metrics": _regression_metrics(
            voltage_active[config.target].to_numpy(dtype=float),
            voltage_active["ann_current_a"].to_numpy(dtype=float),
        ),
        "linear_baseline": {
            "selected_regularization_alpha": float(linear_alpha),
            "test_metrics": _regression_metrics(
                test[config.target].to_numpy(dtype=float), linear_test_prediction
            ),
        },
        "training_weight": {
            "column": sample_weight_column,
            "minimum": float(np.min(train_weight)) if train_weight is not None else 1.0,
            "maximum": float(np.max(train_weight)) if train_weight is not None else 1.0,
            "sum": float(np.sum(train_weight)) if train_weight is not None else float(len(train)),
        },
    }
    return model, pd.DataFrame(selection_records), predictions, metrics

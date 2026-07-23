"""2-7-5-3-1 sigmoid DNN 与 Bayesian-regularized LM 近似训练器。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit

from .phase6p0_config import PhaseSixPZeroConfig


FEATURES = ("surface_voltage_v", "bulk_voltage_v")


@dataclass(frozen=True)
class PaperNDCNetwork:
    feature_minimum: np.ndarray
    feature_maximum: np.ndarray
    target_minimum: float
    target_maximum: float
    layer_sizes: tuple[int, ...]
    parameters: np.ndarray

    def _normalize_features(self, values: np.ndarray) -> np.ndarray:
        span = np.maximum(self.feature_maximum - self.feature_minimum, 1.0e-12)
        return 2.0 * (np.asarray(values, dtype=float) - self.feature_minimum) / span - 1.0

    def _denormalize_target(self, values: np.ndarray) -> np.ndarray:
        return self.target_minimum + 0.5 * (np.asarray(values, dtype=float) + 1.0) * (self.target_maximum - self.target_minimum)

    def predict(self, values: np.ndarray) -> np.ndarray:
        one_dimensional = np.asarray(values).ndim == 1
        x = self._normalize_features(np.atleast_2d(values))
        prediction, _ = _forward_and_jacobian(x, self.parameters, self.layer_sizes, need_jacobian=False)
        physical = self._denormalize_target(prediction)
        return physical[0] if one_dimensional else physical

    @property
    def parameter_count(self) -> int:
        return int(len(self.parameters))

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            feature_minimum=self.feature_minimum,
            feature_maximum=self.feature_maximum,
            target_minimum=np.asarray([self.target_minimum]),
            target_maximum=np.asarray([self.target_maximum]),
            layer_sizes=np.asarray(self.layer_sizes, dtype=int),
            parameters=self.parameters,
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "PaperNDCNetwork":
        with np.load(Path(path), allow_pickle=False) as payload:
            return cls(
                feature_minimum=payload["feature_minimum"].astype(float),
                feature_maximum=payload["feature_maximum"].astype(float),
                target_minimum=float(payload["target_minimum"][0]),
                target_maximum=float(payload["target_maximum"][0]),
                layer_sizes=tuple(int(value) for value in payload["layer_sizes"]),
                parameters=payload["parameters"].astype(float),
            )


def _parameter_count(layer_sizes: tuple[int, ...]) -> int:
    return int(sum((left + 1) * right for left, right in zip(layer_sizes[:-1], layer_sizes[1:])))


def _unpack(parameters: np.ndarray, layer_sizes: tuple[int, ...]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    offset = 0
    for left, right in zip(layer_sizes[:-1], layer_sizes[1:]):
        count = left * right
        weights.append(np.asarray(parameters[offset : offset + count]).reshape(left, right))
        offset += count
        biases.append(np.asarray(parameters[offset : offset + right]))
        offset += right
    if offset != len(parameters):
        raise ValueError("网络参数向量长度与结构不一致。")
    return weights, biases


def _forward_and_jacobian(
    x: np.ndarray,
    parameters: np.ndarray,
    layer_sizes: tuple[int, ...],
    need_jacobian: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    weights, biases = _unpack(parameters, layer_sizes)
    activations = [np.asarray(x, dtype=float)]
    preactivations: list[np.ndarray] = []
    hidden = activations[0]
    for weight, bias in zip(weights[:-1], biases[:-1]):
        z = hidden @ weight + bias
        preactivations.append(z)
        hidden = expit(z)
        activations.append(hidden)
    output = (hidden @ weights[-1] + biases[-1]).reshape(-1)
    if not need_jacobian:
        return output, None

    sample_count = len(x)
    blocks_by_layer: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(weights)
    delta = np.ones((sample_count, 1), dtype=float)
    blocks_by_layer[-1] = (
        np.einsum("ni,nj->nij", activations[-1], delta).reshape(sample_count, -1),
        delta.copy(),
    )
    for layer in range(len(weights) - 2, -1, -1):
        sigmoid = activations[layer + 1]
        delta = (delta @ weights[layer + 1].T) * sigmoid * (1.0 - sigmoid)
        blocks_by_layer[layer] = (
            np.einsum("ni,nj->nij", activations[layer], delta).reshape(sample_count, -1),
            delta.copy(),
        )
    ordered: list[np.ndarray] = []
    for blocks in blocks_by_layer:
        assert blocks is not None
        ordered.extend(blocks)
    return output, np.concatenate(ordered, axis=1)


def _initialize(layer_sizes: tuple[int, ...], seed: int) -> np.ndarray:
    random = np.random.default_rng(seed)
    values: list[np.ndarray] = []
    for left, right in zip(layer_sizes[:-1], layer_sizes[1:]):
        limit = np.sqrt(6.0 / (left + right))
        values.append(random.uniform(-limit, limit, size=(left, right)).reshape(-1))
        values.append(np.zeros(right, dtype=float))
    return np.concatenate(values)


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - target
    scale = float(np.max(target) - np.min(target))
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "sample_count": int(len(target)),
        "rmse_a": rmse,
        "mae_a": float(np.mean(np.abs(error))),
        "maximum_absolute_error_a": float(np.max(np.abs(error))),
        "nrmse_percent": float(100.0 * rmse / scale) if scale > 0.0 else float("nan"),
    }


def train_paper_ndc_network(
    dataset: pd.DataFrame,
    config: PhaseSixPZeroConfig,
) -> tuple[PaperNDCNetwork, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """按论文 90/10 随机样本划分训练 BR-LM 网络，并保留所有种子结果。"""
    required = {*FEATURES, "teacher_current_a"}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"NDC 教师数据缺少列：{missing}")
    random = np.random.default_rng(config.data.random_seed)
    order = random.permutation(len(dataset))
    train_count = int(round(config.data.internal_training_fraction * len(dataset)))
    train_indices = order[:train_count]
    test_indices = order[train_count:]
    split = np.full(len(dataset), "internal_test", dtype=object)
    split[train_indices] = "train"

    x_all = dataset[list(FEATURES)].to_numpy(dtype=float)
    y_all = dataset["teacher_current_a"].to_numpy(dtype=float)
    feature_minimum = x_all[train_indices].min(axis=0)
    feature_maximum = x_all[train_indices].max(axis=0)
    target_minimum = float(y_all[train_indices].min())
    target_maximum = float(y_all[train_indices].max())
    x_normalized = 2.0 * (x_all - feature_minimum) / np.maximum(feature_maximum - feature_minimum, 1.0e-12) - 1.0
    y_normalized = 2.0 * (y_all - target_minimum) / max(target_maximum - target_minimum, 1.0e-12) - 1.0
    layer_sizes = (2, *config.dnn.hidden_layer_sizes, 1)
    records: list[dict[str, Any]] = []
    candidates: list[tuple[float, int, np.ndarray, list[dict[str, float]]]] = []

    x_train = x_normalized[train_indices]
    y_train = y_normalized[train_indices]
    identity = np.eye(_parameter_count(layer_sizes))
    for seed in config.dnn.initialization_seeds:
        parameters = _initialize(layer_sizes, seed)
        alpha = config.dnn.regularization_initial
        beta = config.dnn.noise_precision_initial
        history: list[dict[str, float]] = []
        total_evaluations = 0
        for update in range(config.dnn.bayesian_updates):
            sqrt_alpha = np.sqrt(alpha)
            sqrt_beta = np.sqrt(beta)

            def residual(values: np.ndarray) -> np.ndarray:
                prediction, _ = _forward_and_jacobian(x_train, values, layer_sizes, need_jacobian=False)
                return np.concatenate([sqrt_beta * (prediction - y_train), sqrt_alpha * values])

            def jacobian(values: np.ndarray) -> np.ndarray:
                _, data_jacobian = _forward_and_jacobian(x_train, values, layer_sizes, need_jacobian=True)
                assert data_jacobian is not None
                return np.vstack([sqrt_beta * data_jacobian, sqrt_alpha * identity])

            result = least_squares(
                residual,
                parameters,
                jac=jacobian,
                method="trf",
                max_nfev=config.dnn.maximum_function_evaluations,
                ftol=config.dnn.optimization_tolerance,
                xtol=config.dnn.optimization_tolerance,
                gtol=config.dnn.optimization_tolerance,
            )
            total_evaluations += int(result.nfev)
            parameters = result.x
            prediction, data_jacobian = _forward_and_jacobian(x_train, parameters, layer_sizes, need_jacobian=True)
            assert data_jacobian is not None
            data_error = prediction - y_train
            error_energy = 0.5 * float(data_error @ data_error)
            weight_energy = 0.5 * float(parameters @ parameters)
            hessian = beta * (data_jacobian.T @ data_jacobian) + alpha * identity
            gamma = float(len(parameters) - alpha * np.trace(np.linalg.pinv(hessian)))
            new_alpha = float(np.clip(gamma / max(2.0 * weight_energy, 1.0e-12), 1.0e-12, 1.0e12))
            new_beta = float(np.clip((len(train_indices) - gamma) / max(2.0 * error_energy, 1.0e-12), 1.0e-12, 1.0e12))
            history.append({"update": float(update), "alpha": alpha, "beta": beta, "gamma": gamma, "error_energy": error_energy, "weight_energy": weight_energy})
            if abs(np.log(new_alpha / alpha)) < 1.0e-3 and abs(np.log(new_beta / beta)) < 1.0e-3:
                alpha, beta = new_alpha, new_beta
                break
            alpha, beta = new_alpha, new_beta

        model = PaperNDCNetwork(feature_minimum, feature_maximum, target_minimum, target_maximum, layer_sizes, parameters)
        internal_prediction = model.predict(x_all[test_indices])
        internal_metrics = _metrics(y_all[test_indices], internal_prediction)
        records.append(
            {
                "seed": seed,
                "internal_test_nrmse_percent": internal_metrics["nrmse_percent"],
                "internal_test_rmse_a": internal_metrics["rmse_a"],
                "final_alpha": alpha,
                "final_beta": beta,
                "bayesian_updates_completed": len(history),
                "function_evaluations": total_evaluations,
            }
        )
        candidates.append((internal_metrics["rmse_a"], seed, parameters.copy(), history))

    _, selected_seed, selected_parameters, selected_history = min(candidates, key=lambda item: (item[0], item[1]))
    selected = PaperNDCNetwork(feature_minimum, feature_maximum, target_minimum, target_maximum, layer_sizes, selected_parameters)
    predictions = dataset.copy()
    predictions["split"] = split
    predictions["dnn_current_a"] = selected.predict(x_all)
    predictions["dnn_error_a"] = predictions["dnn_current_a"] - y_all
    metrics = {
        "architecture": list(layer_sizes),
        "parameter_count": selected.parameter_count,
        "selected_seed": selected_seed,
        "normalization": "MATLAB-style mapminmax to [-1,1] inferred because the paper does not report preprocessing",
        "training_algorithm": "Bayesian evidence updates around regularized nonlinear least squares (BR-LM approximation)",
        "split_metrics": {
            name: _metrics(group["teacher_current_a"].to_numpy(), group["dnn_current_a"].to_numpy())
            for name, group in predictions.groupby("split")
        },
        "selected_bayesian_history": selected_history,
    }
    return selected, pd.DataFrame.from_records(records), predictions, metrics

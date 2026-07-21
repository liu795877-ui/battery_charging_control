"""Phase 6 的论文式初态采样、短轨迹 MPC 标签和纯 DNN 训练。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .ann_model import TinyANN
from .mpc import ReducedBatteryModel, ReducedState
from .phase3_config import PhaseThreeConfig
from .phase6_config import FEATURE_NAMES, PhaseSixConfig


@dataclass(frozen=True)
class PaperTrajectoryResult:
    """一个初始状态对应的一次有限时域 MPC 优化结果。"""

    currents_a: np.ndarray
    optimizer_success: bool
    prediction_feasible: bool
    status: str
    objective_value: float
    solve_time_s: float
    minimum_constraint_margin: float


def _radical_inverse(index: int, base: int) -> float:
    """Van der Corput 根逆，用于构造确定性的 Hammersley 点。"""
    value = 0.0
    factor = 1.0 / base
    while index:
        index, digit = divmod(index, base)
        value += digit * factor
        factor /= base
    return value


def hammersley_points(count: int, dimensions: int) -> np.ndarray:
    """返回 [0,1]^d 内的 Hammersley 低差异序列。"""
    if dimensions < 1 or count < 1:
        raise ValueError("Hammersley 序列需要正的样本数和维数。")
    primes = (2, 3, 5, 7, 11, 13, 17)
    if dimensions - 1 > len(primes):
        raise ValueError("当前实现最多支持 8 维 Hammersley 序列。")
    points = np.empty((count, dimensions), dtype=float)
    for row in range(count):
        points[row, 0] = (row + 0.5) / count
        for column in range(1, dimensions):
            points[row, column] = _radical_inverse(row + 1, primes[column - 1])
    return points


def generate_initial_state_design(config: PhaseSixConfig) -> pd.DataFrame:
    """生成 Hammersley 主体与边界全因子点组成的混合 DOCE 设计。"""
    paper = config.paper_method
    ranges = list(paper.state_ranges.values())
    factorial_axes = [
        np.linspace(0.0, 1.0, level) if level > 1 else np.array([0.5])
        for level in paper.factorial_levels
    ]
    factorial = np.asarray(list(product(*factorial_axes)), dtype=float)
    hammersley_count = paper.initial_state_count - len(factorial)
    hammersley = hammersley_points(hammersley_count, len(FEATURE_NAMES))
    unit = np.vstack([hammersley, factorial])
    method = np.array(
        ["hammersley"] * len(hammersley) + ["boundary_factorial"] * len(factorial),
        dtype=object,
    )
    scaled = np.column_stack(
        [low + unit[:, i] * (high - low) for i, (low, high) in enumerate(ranges)]
    )
    frame = pd.DataFrame(scaled, columns=FEATURE_NAMES)
    frame.insert(0, "sampling_method", method)
    frame.insert(0, "initial_state_id", [f"paper_ic_{i:03d}" for i in range(len(frame))])
    return frame


def row_to_state(row: pd.Series) -> ReducedState:
    """把五维论文迁移输入嵌入当前双节点热状态。"""
    temperature = float(row["state_average_temperature_c"])
    return ReducedState(
        soc=float(row["state_soc"]),
        polarization_fast_v=float(row["state_polarization_fast_v"]),
        polarization_slow_v=float(row["state_polarization_slow_v"]),
        core_temperature_c=temperature,
        surface_temperature_c=temperature,
        previous_current_a=float(row["state_previous_current_a"]),
    )


class PaperTrajectoryMPC:
    """一次求解整条短轨迹，对应论文的训练场景生成方式。

    与在线 ``ConstrainedMPC`` 的区别是：这里一次优化完整 5 min 预测窗口，再把
    最前面的八个 5 s 状态—动作对展开为监督样本；不对展开状态再次求解。
    这样保留论文“一次初态优化产生一条短训练轨迹”的结构，同时避免把论文
    1 min 采样下的热预测范围错误缩短成只有 40 s。
    """

    def __init__(
        self,
        model: ReducedBatteryModel,
        phase3: PhaseThreeConfig,
        trajectory_steps: int,
    ) -> None:
        self.model = model
        self.phase3 = phase3
        self.trajectory_steps = trajectory_steps
        self.number_of_blocks = phase3.control.number_of_control_blocks

    def _expand_blocks(self, block_currents: np.ndarray) -> np.ndarray:
        return np.repeat(
            np.asarray(block_currents, dtype=float),
            self.phase3.control.control_block_steps,
        )[: self.phase3.control.prediction_horizon_steps]

    def _prediction(self, state: ReducedState, block_currents: np.ndarray) -> dict[str, np.ndarray]:
        return self.model.predict(state, self._expand_blocks(block_currents))

    def _margins(
        self,
        state: ReducedState,
        block_currents: np.ndarray,
        prediction: dict[str, np.ndarray],
    ) -> np.ndarray:
        constraints = self.phase3.constraints
        changes = np.diff(np.concatenate([[state.previous_current_a], block_currents]))
        return np.concatenate(
            [
                constraints.mpc_maximum_voltage_v - prediction["voltage_v"],
                constraints.mpc_maximum_temperature_c - prediction["temperature_c"],
                self.phase3.battery.target_soc - prediction["soc"],
                constraints.maximum_current_change_a_per_step - changes,
                constraints.maximum_current_change_a_per_step + changes,
            ]
        )

    def _objective(
        self,
        state: ReducedState,
        block_currents: np.ndarray,
        prediction: dict[str, np.ndarray],
    ) -> float:
        objective = self.phase3.objective
        gap = np.maximum(self.phase3.battery.target_soc - prediction["soc"], 0.0)
        changes = np.diff(np.concatenate([[state.previous_current_a], block_currents]))
        maximum = self.phase3.constraints.maximum_current_a
        return float(
            objective.soc_tracking_weight * np.mean(gap)
            + objective.terminal_soc_weight * gap[-1] ** 2
            + objective.current_change_weight * np.mean((changes / maximum) ** 2)
        )

    def solve(self, state: ReducedState) -> PaperTrajectoryResult:
        """求解一个初态对应的最优短轨迹并保留完整动作序列。"""
        constraints = self.phase3.constraints
        delta = constraints.maximum_current_change_a_per_step
        guess = np.empty(self.number_of_blocks, dtype=float)
        previous = state.previous_current_a
        for index in range(self.number_of_blocks):
            previous = min(previous + delta, constraints.maximum_current_a)
            guess[index] = previous

        cache_x: np.ndarray | None = None
        cache_prediction: dict[str, np.ndarray] | None = None

        def evaluate(currents: np.ndarray) -> dict[str, np.ndarray]:
            nonlocal cache_x, cache_prediction
            values = np.asarray(currents, dtype=float)
            if cache_x is None or not np.array_equal(values, cache_x):
                cache_x = values.copy()
                cache_prediction = self._prediction(state, values)
            assert cache_prediction is not None
            return cache_prediction

        start = perf_counter()
        result = minimize(
            lambda x: self._objective(state, x, evaluate(x)),
            guess,
            method="SLSQP",
            bounds=[(0.0, constraints.maximum_current_a)] * self.number_of_blocks,
            constraints=[
                {"type": "ineq", "fun": lambda x: self._margins(state, x, evaluate(x))}
            ],
            options={
                "maxiter": self.phase3.optimizer.maximum_iterations,
                "ftol": self.phase3.optimizer.function_tolerance,
                "disp": False,
            },
        )
        elapsed = perf_counter() - start
        block_currents = np.asarray(result.x, dtype=float)
        prediction = self._prediction(state, block_currents)
        minimum_margin = float(np.min(self._margins(state, block_currents, prediction)))
        feasible = minimum_margin >= -self.phase3.optimizer.constraint_tolerance
        return PaperTrajectoryResult(
            currents_a=self._expand_blocks(block_currents)[: self.trajectory_steps],
            optimizer_success=bool(result.success),
            prediction_feasible=bool(feasible),
            status=str(result.message),
            objective_value=float(result.fun),
            solve_time_s=float(elapsed),
            minimum_constraint_margin=minimum_margin,
        )


def _assign_splits(ids: list[str], config: PhaseSixConfig) -> dict[str, str]:
    """按整条初态轨迹划分，避免相邻展开状态跨集合泄漏。"""
    values = np.asarray(sorted(ids), dtype=object)
    random = np.random.default_rng(config.random_seed)
    random.shuffle(values)
    train_end = int(round(len(values) * config.paper_method.train_fraction))
    validation_end = train_end + int(
        round(len(values) * config.paper_method.validation_fraction)
    )
    mapping: dict[str, str] = {}
    for value in values[:train_end]:
        mapping[str(value)] = "train"
    for value in values[train_end:validation_end]:
        mapping[str(value)] = "validation"
    for value in values[validation_end:]:
        mapping[str(value)] = "test"
    return mapping


def generate_paper_teacher_dataset(
    design: pd.DataFrame,
    model: ReducedBatteryModel,
    phase3: PhaseThreeConfig,
    config: PhaseSixConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """对每个初态只求解一次 MPC，并展开通过可行性审计的短轨迹。"""
    optimizer = PaperTrajectoryMPC(
        model, phase3, config.paper_method.trajectory_steps
    )
    attempts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for _, initial_row in design.iterrows():
        state = row_to_state(initial_row)
        result = optimizer.solve(state)
        accepted = result.optimizer_success and result.prediction_feasible
        attempts.append(
            {
                **initial_row.to_dict(),
                "teacher_optimizer_success": result.optimizer_success,
                "teacher_prediction_feasible": result.prediction_feasible,
                "teacher_accepted": accepted,
                "teacher_status": result.status,
                "teacher_objective": result.objective_value,
                "teacher_solve_time_s": result.solve_time_s,
                "teacher_minimum_constraint_margin": result.minimum_constraint_margin,
            }
        )
        if not accepted:
            continue
        running_state = state
        for step_index, current in enumerate(result.currents_a):
            next_state, output = model.step(running_state, float(current))
            change = float(current - running_state.previous_current_a)
            rows.append(
                {
                    "trajectory_id": str(initial_row["initial_state_id"]),
                    "sampling_method": str(initial_row["sampling_method"]),
                    "step_index": step_index,
                    "state_soc": running_state.soc,
                    "state_polarization_fast_v": running_state.polarization_fast_v,
                    "state_polarization_slow_v": running_state.polarization_slow_v,
                    "state_average_temperature_c": model.average_temperature(running_state),
                    "state_previous_current_a": running_state.previous_current_a,
                    "teacher_current_a": float(current),
                    "teacher_accepted": True,
                    "active_voltage_constraint": bool(
                        output.constraint_voltage_v >= phase3.constraints.mpc_maximum_voltage_v - 0.01
                    ),
                    "active_temperature_constraint": bool(
                        output.constraint_temperature_c >= phase3.constraints.mpc_maximum_temperature_c - 0.10
                    ),
                    "active_current_upper_constraint": bool(
                        current >= phase3.constraints.maximum_current_a - 0.05
                    ),
                    "active_current_change_constraint": bool(
                        abs(change) >= phase3.constraints.maximum_current_change_a_per_step - 0.05
                    ),
                    "next_soc": next_state.soc,
                    "next_voltage_v": output.terminal_voltage_v,
                    "next_average_temperature_c": output.average_temperature_c,
                }
            )
            running_state = next_state
    attempt_frame = pd.DataFrame.from_records(attempts)
    dataset = pd.DataFrame.from_records(rows)
    accepted_ids = dataset["trajectory_id"].drop_duplicates().tolist() if len(dataset) else []
    split_map = _assign_splits(accepted_ids, config) if accepted_ids else {}
    if len(dataset):
        dataset["split"] = dataset["trajectory_id"].map(split_map)
    acceptance = float(attempt_frame["teacher_accepted"].mean())
    split_counts = dataset["split"].value_counts().to_dict() if len(dataset) else {}
    trajectory_counts = (
        dataset[["trajectory_id", "split"]].drop_duplicates()["split"].value_counts().to_dict()
        if len(dataset)
        else {}
    )
    metrics = {
        "attempted_initial_state_count": int(len(attempt_frame)),
        "accepted_initial_state_count": int(len(accepted_ids)),
        "teacher_acceptance_fraction": acceptance,
        "unfolded_sample_count": int(len(dataset)),
        "split_sample_counts": {str(k): int(v) for k, v in split_counts.items()},
        "split_trajectory_counts": {str(k): int(v) for k, v in trajectory_counts.items()},
        "duplicate_feature_row_count": int(dataset.duplicated(list(FEATURE_NAMES)).sum()) if len(dataset) else 0,
        "active_constraint_counts": {
            "voltage": int(dataset["active_voltage_constraint"].sum()) if len(dataset) else 0,
            "temperature": int(dataset["active_temperature_constraint"].sum()) if len(dataset) else 0,
            "current_upper": int(dataset["active_current_upper_constraint"].sum()) if len(dataset) else 0,
            "current_change": int(dataset["active_current_change_constraint"].sum()) if len(dataset) else 0,
        },
        "mean_teacher_solve_time_ms": float(attempt_frame["teacher_solve_time_s"].mean() * 1000.0),
        "maximum_teacher_solve_time_ms": float(attempt_frame["teacher_solve_time_s"].max() * 1000.0),
    }
    criteria = config.success_criteria
    metrics["checks"] = {
        "accepted_initial_states": len(accepted_ids) >= criteria.minimum_accepted_initial_states,
        "teacher_acceptance_fraction": acceptance >= criteria.minimum_teacher_acceptance_fraction,
        "trajectory_split_isolation": bool(
            len(dataset) and dataset.groupby("trajectory_id")["split"].nunique().max() == 1
        ),
        "all_splits_nonempty": all(split_counts.get(name, 0) > 0 for name in ("train", "validation", "test")),
        "current_labels_bounded": bool(
            len(dataset) and dataset["teacher_current_a"].between(0.0, phase3.constraints.maximum_current_a).all()
        ),
    }
    metrics["success"] = bool(all(metrics["checks"].values()))
    return attempt_frame, dataset, metrics


def _regression_metrics(target: np.ndarray, prediction: np.ndarray, scale: float) -> dict[str, float]:
    error = np.asarray(prediction) - np.asarray(target)
    rmse = float(np.sqrt(np.mean(error**2)))
    denominator = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "sample_count": int(len(target)),
        "mae_a": float(np.mean(np.abs(error))),
        "rmse_a": rmse,
        "nrmse": rmse / scale,
        "maximum_absolute_error_a": float(np.max(np.abs(error))),
        "bias_a": float(np.mean(error)),
        "r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0.0 else 0.0,
    }


def train_paper_dnn(
    dataset: pd.DataFrame, config: PhaseSixConfig
) -> tuple[TinyANN, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """按整轨迹划分训练纯 DNN，并用验证集执行论文式 cut-and-try。"""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    if dataset.groupby("trajectory_id")["split"].nunique().max() != 1:
        raise ValueError("同一论文式短轨迹不能跨越训练、验证和测试集合。")
    train = dataset[dataset["split"] == "train"]
    validation = dataset[dataset["split"] == "validation"]
    test = dataset[dataset["split"] == "test"]
    if any(frame.empty for frame in (train, validation, test)):
        raise ValueError("训练、验证和测试集合都必须非空。")
    feature_scaler = StandardScaler().fit(train[list(FEATURE_NAMES)])
    target_scaler = StandardScaler().fit(train[["teacher_current_a"]])
    x_train = feature_scaler.transform(train[list(FEATURE_NAMES)])
    y_train = target_scaler.transform(train[["teacher_current_a"]]).reshape(-1)
    x_validation = feature_scaler.transform(validation[list(FEATURE_NAMES)])
    y_validation = validation["teacher_current_a"].to_numpy(dtype=float)
    scale = config.nominal_validation.current_nrmse_normalization_a
    selections: list[dict[str, Any]] = []
    candidates: list[tuple[float, int, tuple[int, ...], Any]] = []
    for architecture in config.network.candidate_hidden_layer_sizes:
        for alpha in config.network.regularization_candidates:
            for seed in config.network.initialization_seeds:
                estimator = MLPRegressor(
                    hidden_layer_sizes=architecture,
                    activation=config.network.activation,
                    solver=config.network.solver,
                    alpha=alpha,
                    max_iter=config.network.maximum_iterations,
                    tol=config.network.convergence_tolerance,
                    random_state=seed,
                )
                estimator.fit(x_train, y_train)
                prediction = estimator.predict(x_validation) * target_scaler.scale_[0] + target_scaler.mean_[0]
                metrics = _regression_metrics(y_validation, prediction, scale)
                selections.append(
                    {
                        "architecture": "-".join(str(v) for v in architecture),
                        "regularization_alpha": alpha,
                        "initialization_seed": seed,
                        "validation_mae_a": metrics["mae_a"],
                        "validation_rmse_a": metrics["rmse_a"],
                        "validation_nrmse": metrics["nrmse"],
                        "optimization_iterations": int(estimator.n_iter_),
                    }
                )
                candidates.append((metrics["rmse_a"], seed, architecture, estimator))
    _, selected_seed, architecture, selected = min(candidates, key=lambda item: (item[0], item[1]))
    selected_row = min(selections, key=lambda row: (row["validation_rmse_a"], row["initialization_seed"]))
    model = TinyANN(
        feature_names=FEATURE_NAMES,
        feature_mean=feature_scaler.mean_.astype(float),
        feature_scale=feature_scaler.scale_.astype(float),
        target_mean=float(target_scaler.mean_[0]),
        target_scale=float(target_scaler.scale_[0]),
        weights=tuple(np.asarray(value, dtype=float) for value in selected.coefs_),
        biases=tuple(np.asarray(value, dtype=float) for value in selected.intercepts_),
    )
    predictions = dataset.copy()
    feature_values = predictions[list(FEATURE_NAMES)].to_numpy(dtype=float)
    predictions["dnn_current_a"] = model.predict_unclipped(feature_values)
    predictions["dnn_error_a"] = predictions["dnn_current_a"] - predictions["teacher_current_a"]
    split_metrics = {
        name: _regression_metrics(
            frame["teacher_current_a"].to_numpy(dtype=float),
            frame["dnn_current_a"].to_numpy(dtype=float),
            scale,
        )
        for name, frame in predictions.groupby("split")
    }
    metrics = {
        "architecture": [len(FEATURE_NAMES), *architecture, 1],
        "parameter_count": model.parameter_count,
        "selected_regularization_alpha": float(selected_row["regularization_alpha"]),
        "selected_initialization_seed": int(selected_seed),
        "split_metrics": split_metrics,
        "raw_output_below_zero_count": int((predictions["dnn_current_a"] < 0.0).sum()),
        "raw_output_above_maximum_count": int((predictions["dnn_current_a"] > 10.0).sum()),
    }
    return model, pd.DataFrame.from_records(selections), predictions, metrics

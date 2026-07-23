"""Multi-seed offline controller comparison on the corrected Phase 6R dataset."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .ann_model import TinyANN
from .phase6_config import FEATURE_NAMES as FIVE_STATE_FEATURES
from .phase6c1_runner import file_sha256
from .phase6r_config import PhaseSixRConfig, ROLLING_STATE_FEATURES


CONTROLLERS = ("five_state_pure", "full_state_pure", "full_state_feasible_interval")


def feasible_current_from_latent(
    latent_model: TinyANN,
    features: np.ndarray,
    previous_current_column: int,
    maximum_current_a: float,
    maximum_change_a: float,
) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values.reshape(1, -1)
    previous = values[:, previous_current_column]
    lower = np.maximum(0.0, previous - maximum_change_a)
    upper = np.minimum(maximum_current_a, previous + maximum_change_a)
    latent = np.asarray(latent_model.predict_unclipped(values), dtype=float).reshape(-1)
    current = 0.5 * (lower + upper) + 0.5 * (upper - lower) * np.tanh(latent)
    return current[0] if one_dimensional else current


def feasible_latent_target(
    teacher_current: np.ndarray,
    previous_current: np.ndarray,
    maximum_current_a: float,
    maximum_change_a: float,
    clip: float = 0.999,
) -> np.ndarray:
    previous = np.asarray(previous_current, dtype=float)
    lower = np.maximum(0.0, previous - maximum_change_a)
    upper = np.minimum(maximum_current_a, previous + maximum_change_a)
    normalized = (2.0 * np.asarray(teacher_current, dtype=float) - lower - upper) / (upper - lower)
    return np.arctanh(np.clip(normalized, -clip, clip))


def _metrics(target: np.ndarray, prediction: np.ndarray, scale: float) -> dict[str, float]:
    error = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    rmse = float(np.sqrt(np.mean(error**2)))
    denominator = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "mae_a": float(np.mean(np.abs(error))),
        "rmse_a": rmse,
        "nrmse": rmse / scale,
        "maximum_absolute_error_a": float(np.max(np.abs(error))),
        "bias_a": float(np.mean(error)),
        "r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0.0 else 0.0,
    }


def _fit_model(
    train: pd.DataFrame,
    features: tuple[str, ...],
    target: np.ndarray,
    config: PhaseSixRConfig,
    seed: int,
) -> tuple[TinyANN, dict[str, Any]]:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    feature_scaler = StandardScaler().fit(train[list(features)])
    target_scaler = StandardScaler().fit(np.asarray(target, dtype=float).reshape(-1, 1))
    estimator = MLPRegressor(
        hidden_layer_sizes=config.network.hidden_layer_sizes,
        activation=config.network.activation,
        solver="adam",
        alpha=config.network.regularization_alpha,
        max_iter=config.network.maximum_iterations,
        tol=config.network.convergence_tolerance,
        learning_rate_init=config.network.learning_rate_init,
        n_iter_no_change=config.network.no_improvement_iterations,
        random_state=seed,
        shuffle=True,
    )
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(
            feature_scaler.transform(train[list(features)]),
            target_scaler.transform(np.asarray(target).reshape(-1, 1)).reshape(-1),
        )
    model = TinyANN(
        feature_names=features,
        feature_mean=feature_scaler.mean_.astype(float),
        feature_scale=feature_scaler.scale_.astype(float),
        target_mean=float(target_scaler.mean_[0]),
        target_scale=float(target_scaler.scale_[0]),
        weights=tuple(np.asarray(value, dtype=float) for value in estimator.coefs_),
        biases=tuple(np.asarray(value, dtype=float) for value in estimator.intercepts_),
    )
    return model, {
        "fit_time_s": perf_counter() - started,
        "optimization_iterations": int(estimator.n_iter_),
        "reached_iteration_limit": bool(estimator.n_iter_ >= config.network.maximum_iterations),
        "warning_count": len(caught),
        "warnings": sorted({str(item.message) for item in caught}),
    }


def _predict(
    controller: str,
    model: TinyANN,
    frame: pd.DataFrame,
    maximum_current_a: float,
    maximum_change_a: float,
) -> np.ndarray:
    features = frame[list(model.feature_names)].to_numpy(dtype=float)
    if controller in {"five_state_pure", "full_state_pure"}:
        return np.asarray(model.predict_unclipped(features), dtype=float)
    if controller == "full_state_feasible_interval":
        return np.asarray(
            feasible_current_from_latent(
                model,
                features,
                model.feature_names.index("state_previous_current_a"),
                maximum_current_a,
                maximum_change_a,
            ),
            dtype=float,
        )
    raise ValueError(f"Unknown Phase 6R controller: {controller}")


def _summarize(runs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for controller, group in runs.groupby("controller", sort=False):
        record: dict[str, Any] = {"controller": controller, "seed_count": int(len(group))}
        for split in ("train", "validation", "test"):
            for metric in ("nrmse", "maximum_absolute_error_a"):
                values = group[f"{split}_{metric}"].to_numpy(dtype=float)
                record[f"{split}_{metric}_mean"] = float(np.mean(values))
                record[f"{split}_{metric}_std"] = float(np.std(values, ddof=1))
                record[f"{split}_{metric}_best"] = float(np.min(values))
        record["passing_test_seed_count"] = int(
            (group["test_nrmse"] < 0.01).sum()
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def run_phase_six_r_training(
    config: PhaseSixRConfig,
    project_root: str | Path,
) -> dict[str, Any]:
    from .phase3_config import load_phase_three_config

    root = Path(project_root)
    data_dir = root / "data" / "phase6r_corrected_policy_distillation"
    model_dir = root / "outputs" / "models" / "phase6r"
    metrics_dir = root / "outputs" / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = data_dir / "rolling_first_action_dataset.csv"
    dataset = pd.read_csv(dataset_path)
    teacher_metrics = json.loads(
        (metrics_dir / "phase6r_teacher_metrics.json").read_text(encoding="utf-8")
    )
    if not teacher_metrics["teacher_dataset"]["success"]:
        raise RuntimeError("Phase 6R corrected teacher dataset did not pass its gate.")
    if dataset.groupby("trajectory_id")["split"].nunique().max() != 1:
        raise RuntimeError("Phase 6R trajectory split leakage detected.")
    phase3 = load_phase_three_config(root / config.source_phase3_config)
    maximum_current = phase3.constraints.maximum_current_a
    maximum_change = phase3.constraints.maximum_current_change_a_per_step
    train = dataset[dataset["split"] == "train"]
    frames = {name: dataset[dataset["split"] == name] for name in ("train", "validation", "test")}
    runs_path = data_dir / "offline_controller_runs.csv"
    records = pd.read_csv(runs_path).to_dict("records") if runs_path.exists() else []
    completed = {str(record["run_key"]) for record in records}
    for seed in config.network.initialization_seeds:
        for controller in CONTROLLERS:
            run_key = f"{controller}__seed-{seed}"
            if run_key in completed:
                continue
            if controller == "five_state_pure":
                features = tuple(FIVE_STATE_FEATURES)
                target = train["teacher_current_a"].to_numpy(dtype=float)
            elif controller == "full_state_pure":
                features = tuple(ROLLING_STATE_FEATURES)
                target = train["teacher_current_a"].to_numpy(dtype=float)
            else:
                features = tuple(ROLLING_STATE_FEATURES)
                target = feasible_latent_target(
                    train["teacher_current_a"].to_numpy(dtype=float),
                    train["state_previous_current_a"].to_numpy(dtype=float),
                    maximum_current,
                    maximum_change,
                )
            model, optimization = _fit_model(train, features, target, config, seed)
            model.save(model_dir / f"{run_key}.npz")
            record: dict[str, Any] = {
                "run_key": run_key,
                "controller": controller,
                "initialization_seed": seed,
                "parameter_count": model.parameter_count,
                "fit_time_s": optimization["fit_time_s"],
                "optimization_iterations": optimization["optimization_iterations"],
                "optimizer_reached_iteration_limit": optimization["reached_iteration_limit"],
                "warning_count": optimization["warning_count"],
            }
            for split, frame in frames.items():
                split_metrics = _metrics(
                    frame["teacher_current_a"].to_numpy(dtype=float),
                    _predict(controller, model, frame, maximum_current, maximum_change),
                    config.validation.current_nrmse_normalization_a,
                )
                record.update({f"{split}_{name}": value for name, value in split_metrics.items()})
            records.append(record)
            completed.add(run_key)
            pd.DataFrame.from_records(records).sort_values("run_key").to_csv(runs_path, index=False)
            print(
                f"completed {run_key}: train={100 * record['train_nrmse']:.3f}% "
                f"validation={100 * record['validation_nrmse']:.3f}% "
                f"test={100 * record['test_nrmse']:.3f}%",
                flush=True,
            )
    runs = pd.DataFrame.from_records(records)
    summary = _summarize(runs)
    summary.to_csv(data_dir / "offline_controller_summary.csv", index=False)
    payload = {
        "status": "completed",
        "configuration": asdict(config),
        "dataset": {
            "path": str(dataset_path),
            "sha256": file_sha256(dataset_path),
            "sample_count": int(len(dataset)),
            "split_sample_counts": {
                str(key): int(value) for key, value in dataset["split"].value_counts().items()
            },
        },
        "summary": summary.to_dict("records"),
        "offline_gate": {
            "threshold_nrmse": config.validation.maximum_offline_nrmse,
            "controllers_with_passing_majority": summary.loc[
                summary["passing_test_seed_count"] > summary["seed_count"] / 2,
                "controller",
            ].tolist(),
        },
    }
    (metrics_dir / "phase6r_offline_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload

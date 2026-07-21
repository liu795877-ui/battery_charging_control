"""Run Phase 6C-1 without changing the frozen Phase 6B teacher dataset."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .ann_model import TinyANN
from .phase6_config import FEATURE_NAMES
from .phase6c1_config import PhaseSixC1Config


SPLITS = ("train", "validation", "test")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_frozen_dataset(dataset: pd.DataFrame, path: Path, config: PhaseSixC1Config) -> dict[str, Any]:
    baseline = config.baseline
    required_columns = {*FEATURE_NAMES, "trajectory_id", "split", "teacher_current_a"}
    missing = sorted(required_columns.difference(dataset.columns))
    sample_counts = {name: int((dataset["split"] == name).sum()) for name in SPLITS}
    trajectory_counts = {
        name: int(dataset.loc[dataset["split"] == name, "trajectory_id"].nunique())
        for name in SPLITS
    }
    trajectory_split_leak_count = int(
        (dataset.groupby("trajectory_id")["split"].nunique() != 1).sum()
    )
    digest = file_sha256(path)
    checks = {
        "sha256": digest.lower() == baseline.dataset_sha256.lower(),
        "sample_count": len(dataset) == baseline.expected_sample_count,
        "split_sample_counts": sample_counts == baseline.expected_split_sample_counts,
        "split_trajectory_counts": trajectory_counts == baseline.expected_split_trajectory_counts,
        "trajectory_split_isolation": trajectory_split_leak_count == 0,
        "required_columns": not missing,
        "finite_features_and_target": bool(
            np.isfinite(dataset[[*FEATURE_NAMES, "teacher_current_a"]].to_numpy(dtype=float)).all()
        ),
    }
    result = {
        "path": str(path),
        "sha256": digest,
        "sample_count": int(len(dataset)),
        "split_sample_counts": sample_counts,
        "split_trajectory_counts": trajectory_counts,
        "trajectory_split_leak_count": trajectory_split_leak_count,
        "missing_columns": missing,
        "checks": checks,
        "success": bool(all(checks.values())),
    }
    if not result["success"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Frozen Phase 6B dataset audit failed: {failed}")
    return result


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


def _new_estimator(config: PhaseSixC1Config, architecture: tuple[int, ...], seed: int, solver: str, max_iter: int):
    from sklearn.neural_network import MLPRegressor

    common: dict[str, Any] = {
        "hidden_layer_sizes": architecture,
        "activation": config.network.activation,
        "solver": solver,
        "alpha": config.network.regularization_alpha,
        "max_iter": max_iter,
        "tol": config.optimizers.convergence_tolerance,
        "random_state": seed,
    }
    if solver == "adam":
        common.update(
            learning_rate_init=config.optimizers.adam_learning_rate_init,
            n_iter_no_change=config.optimizers.adam_no_improvement_iterations,
            shuffle=True,
        )
    return MLPRegressor(**common)


def _fit_estimator(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: PhaseSixC1Config,
    architecture: tuple[int, ...],
    seed: int,
    optimizer: str,
) -> tuple[Any, dict[str, Any]]:
    started = perf_counter()
    caught: list[warnings.WarningMessage] = []
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        if optimizer == "lbfgs":
            estimator = _new_estimator(
                config, architecture, seed, "lbfgs", config.optimizers.lbfgs_maximum_iterations
            )
            estimator.fit(x_train, y_train)
            stage_iterations = {"lbfgs": int(estimator.n_iter_)}
            stage_limits = {"lbfgs": config.optimizers.lbfgs_maximum_iterations}
        elif optimizer == "adam":
            estimator = _new_estimator(
                config, architecture, seed, "adam", config.optimizers.adam_maximum_iterations
            )
            estimator.fit(x_train, y_train)
            stage_iterations = {"adam": int(estimator.n_iter_)}
            stage_limits = {"adam": config.optimizers.adam_maximum_iterations}
        elif optimizer == "adam_lbfgs":
            estimator = _new_estimator(
                config, architecture, seed, "adam", config.optimizers.adam_pretrain_iterations
            )
            estimator.set_params(warm_start=True)
            estimator.fit(x_train, y_train)
            adam_iterations = int(estimator.n_iter_)
            estimator.set_params(
                solver="lbfgs",
                max_iter=config.optimizers.finetune_lbfgs_maximum_iterations,
                tol=config.optimizers.convergence_tolerance,
                warm_start=True,
            )
            estimator.fit(x_train, y_train)
            stage_iterations = {"adam": adam_iterations, "lbfgs": int(estimator.n_iter_)}
            stage_limits = {
                "adam": config.optimizers.adam_pretrain_iterations,
                "lbfgs": config.optimizers.finetune_lbfgs_maximum_iterations,
            }
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        caught.extend(recorded)
    return estimator, {
        "fit_time_s": perf_counter() - started,
        "stage_iterations": stage_iterations,
        "stage_iteration_limits": stage_limits,
        "stage_reached_iteration_limit": {
            name: stage_iterations[name] >= limit for name, limit in stage_limits.items()
        },
        "warning_count": len(caught),
        "warnings": sorted({str(item.message) for item in caught}),
    }


def _to_tiny_ann(estimator: Any, feature_scaler: Any, target_scaler: Any) -> TinyANN:
    return TinyANN(
        feature_names=FEATURE_NAMES,
        feature_mean=feature_scaler.mean_.astype(float),
        feature_scale=feature_scaler.scale_.astype(float),
        target_mean=float(target_scaler.mean_[0]),
        target_scale=float(target_scaler.scale_[0]),
        weights=tuple(np.asarray(value, dtype=float) for value in estimator.coefs_),
        biases=tuple(np.asarray(value, dtype=float) for value in estimator.intercepts_),
    )


def _run_key(architecture: tuple[int, ...], optimizer: str, seed: int) -> str:
    return f"{'-'.join(map(str, architecture))}__{optimizer}__seed-{seed}"


def _summarize(runs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (architecture, optimizer), group in runs.groupby(["architecture", "optimizer"], sort=False):
        best = group.sort_values(["validation_nrmse", "initialization_seed"]).iloc[0]
        record: dict[str, Any] = {
            "architecture": architecture,
            "optimizer": optimizer,
            "seed_count": int(len(group)),
            "best_validation_seed": int(best["initialization_seed"]),
            "best_validation_nrmse": float(best["validation_nrmse"]),
            "best_seed_test_nrmse": float(best["test_nrmse"]),
        }
        for split in SPLITS:
            for metric in ("nrmse", "maximum_absolute_error_a"):
                values = group[f"{split}_{metric}"].to_numpy(dtype=float)
                record[f"{split}_{metric}_mean"] = float(np.mean(values))
                record[f"{split}_{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                record[f"{split}_{metric}_best"] = float(np.min(values))
        record["test_nrmse_range"] = float(group["test_nrmse"].max() - group["test_nrmse"].min())
        records.append(record)
    return pd.DataFrame.from_records(records)


def _interpret(summary: pd.DataFrame, config: PhaseSixC1Config) -> dict[str, Any]:
    analysis = config.analysis
    best_index = summary["best_validation_nrmse"].idxmin()
    selected = summary.loc[best_index]
    generalization_limited = summary[
        (summary["train_nrmse_mean"] < analysis.low_training_nrmse_threshold)
        & (summary["test_nrmse_mean"] >= analysis.high_test_nrmse_threshold)
    ]
    all_test_above_five_percent = bool(
        (summary["test_nrmse_mean"] >= analysis.high_test_nrmse_threshold).all()
    )
    unstable = summary[
        (summary["test_nrmse_std"] >= analysis.seed_instability_test_nrmse_std_threshold)
        | (summary["test_nrmse_range"] >= analysis.seed_instability_test_nrmse_range_threshold)
    ]
    if len(generalization_limited):
        primary_diagnosis = "generalization_or_coverage_limited"
    elif all_test_above_five_percent:
        primary_diagnosis = "optimization_or_representation_limited"
    else:
        primary_diagnosis = "offline_threshold_reached"
    return {
        "selection_rule": "minimum mean validation NRMSE across five-seed groups",
        "selected_architecture": str(selected["architecture"]),
        "selected_optimizer": str(selected["optimizer"]),
        "selected_group_train_nrmse_mean": float(selected["train_nrmse_mean"]),
        "selected_group_validation_nrmse_mean": float(selected["validation_nrmse_mean"]),
        "selected_group_test_nrmse_mean": float(selected["test_nrmse_mean"]),
        "selected_group_generalization_gap": float(
            selected["test_nrmse_mean"] - selected["train_nrmse_mean"]
        ),
        "primary_diagnosis": primary_diagnosis,
        "all_group_mean_test_nrmse_at_or_above_threshold": all_test_above_five_percent,
        "stop_pure_network_scaling": all_test_above_five_percent,
        "generalization_limited_group_count": int(len(generalization_limited)),
        "generalization_limited_groups": generalization_limited[
            ["architecture", "optimizer", "train_nrmse_mean", "test_nrmse_mean"]
        ].to_dict("records"),
        "unstable_group_count": int(len(unstable)),
        "unstable_groups": unstable[["architecture", "optimizer"]].to_dict("records"),
        "thresholds": asdict(analysis),
    }


def _write_report(path: Path, payload: dict[str, Any], summary: pd.DataFrame) -> None:
    interpretation = payload["interpretation"]
    lines = [
        "# Phase 6C-1: optimization versus generalization ablation",
        "",
        "## Frozen baseline",
        "",
        f"- Phase 6B commit: `{payload['configuration']['baseline']['phase6b_commit']}`",
        f"- Dataset SHA-256: `{payload['dataset_audit']['sha256']}`",
        f"- Samples: {payload['dataset_audit']['sample_count']}",
        f"- Split samples: {payload['dataset_audit']['split_sample_counts']}",
        f"- Split trajectories: {payload['dataset_audit']['split_trajectory_counts']}",
        "- No teacher data were generated and no split was changed.",
        "",
        "## Five-seed results",
        "",
        "| Architecture | Optimizer | Train NRMSE mean ± SD | Validation NRMSE mean ± SD | Test NRMSE mean ± SD | Best test NRMSE | Best validation seed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['architecture']} | {row['optimizer']} | "
            f"{100 * row['train_nrmse_mean']:.3f}% ± {100 * row['train_nrmse_std']:.3f}% | "
            f"{100 * row['validation_nrmse_mean']:.3f}% ± {100 * row['validation_nrmse_std']:.3f}% | "
            f"{100 * row['test_nrmse_mean']:.3f}% ± {100 * row['test_nrmse_std']:.3f}% | "
            f"{100 * row['test_nrmse_best']:.3f}% | {int(row['best_validation_seed'])} |"
        )
    lines.extend(
        [
            "",
            "## Maximum absolute error across seeds",
            "",
            "| Architecture | Optimizer | Train mean ± SD / best | Validation mean ± SD / best | Test mean ± SD / best |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['architecture']} | {row['optimizer']} | "
            f"{row['train_maximum_absolute_error_a_mean']:.3f} ± {row['train_maximum_absolute_error_a_std']:.3f} / {row['train_maximum_absolute_error_a_best']:.3f} A | "
            f"{row['validation_maximum_absolute_error_a_mean']:.3f} ± {row['validation_maximum_absolute_error_a_std']:.3f} / {row['validation_maximum_absolute_error_a_best']:.3f} A | "
            f"{row['test_maximum_absolute_error_a_mean']:.3f} ± {row['test_maximum_absolute_error_a_std']:.3f} / {row['test_maximum_absolute_error_a_best']:.3f} A |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Selected group: {interpretation['selected_architecture']} / {interpretation['selected_optimizer']}",
            f"- Primary diagnosis: `{interpretation['primary_diagnosis']}`",
            f"- Selected-group generalization gap: {100 * interpretation['selected_group_generalization_gap']:.3f} percentage points",
            f"- Stop pure network scaling: {interpretation['stop_pure_network_scaling']}",
            f"- Generalization-limited groups: {interpretation['generalization_limited_group_count']}",
            f"- Seed-unstable groups: {interpretation['unstable_group_count']}",
            "",
            "The diagnosis applies only to the frozen Phase 6B distribution. It does not yet establish 25 °C nominal closed-loop acceptance.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase_six_c1(config: PhaseSixC1Config, project_root: str | Path) -> dict[str, Any]:
    from sklearn.preprocessing import StandardScaler

    root = Path(project_root)
    data_dir = root / "data" / "phase6c_constraint_regime_learning" / "c1_ablation"
    model_dir = root / "outputs" / "models" / "phase6c1_ablation"
    metrics_dir = root / "outputs" / "metrics"
    for directory in (data_dir, model_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dataset_path = root / config.baseline.dataset
    dataset = pd.read_csv(dataset_path)
    audit = audit_frozen_dataset(dataset, dataset_path, config)
    (data_dir / "frozen_dataset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    train = dataset[dataset["split"] == "train"]
    feature_scaler = StandardScaler().fit(train[list(FEATURE_NAMES)])
    target_scaler = StandardScaler().fit(train[["teacher_current_a"]])
    x_train = feature_scaler.transform(train[list(FEATURE_NAMES)])
    y_train = target_scaler.transform(train[["teacher_current_a"]]).reshape(-1)

    checkpoint_path = data_dir / "ablation_runs.csv"
    records = pd.read_csv(checkpoint_path).to_dict("records") if checkpoint_path.exists() else []
    completed = {str(record["run_key"]) for record in records}
    expected_keys = {
        _run_key(architecture, optimizer, seed)
        for architecture in config.network.hidden_layer_sizes
        for optimizer in config.optimizers.methods
        for seed in config.network.initialization_seeds
    }
    for architecture in config.network.hidden_layer_sizes:
        for optimizer in config.optimizers.methods:
            for seed in config.network.initialization_seeds:
                run_key = _run_key(architecture, optimizer, seed)
                if run_key in completed:
                    continue
                estimator, optimization = _fit_estimator(
                    x_train, y_train, config, architecture, seed, optimizer
                )
                model = _to_tiny_ann(estimator, feature_scaler, target_scaler)
                model.save(model_dir / f"{run_key}.npz")
                record: dict[str, Any] = {
                    "run_key": run_key,
                    "dataset_sha256": audit["sha256"],
                    "architecture": "-".join(map(str, architecture)),
                    "optimizer": optimizer,
                    "initialization_seed": seed,
                    "parameter_count": model.parameter_count,
                    "fit_time_s": optimization["fit_time_s"],
                    "stage_iterations_json": json.dumps(optimization["stage_iterations"], sort_keys=True),
                    "stage_reached_iteration_limit_json": json.dumps(
                        optimization["stage_reached_iteration_limit"], sort_keys=True
                    ),
                    "warning_count": optimization["warning_count"],
                    "warnings_json": json.dumps(optimization["warnings"], ensure_ascii=False),
                }
                for split in SPLITS:
                    frame = dataset[dataset["split"] == split]
                    prediction = model.predict_unclipped(frame[list(FEATURE_NAMES)].to_numpy(dtype=float))
                    split_metrics = _metrics(
                        frame["teacher_current_a"].to_numpy(dtype=float),
                        prediction,
                        config.baseline.current_nrmse_normalization_a,
                    )
                    record.update({f"{split}_{name}": value for name, value in split_metrics.items()})
                records.append(record)
                completed.add(run_key)
                pd.DataFrame.from_records(records).sort_values("run_key").to_csv(checkpoint_path, index=False)
                print(
                    f"completed {len(completed)}/{len(expected_keys)} {run_key}: "
                    f"train={100 * record['train_nrmse']:.3f}% "
                    f"validation={100 * record['validation_nrmse']:.3f}% "
                    f"test={100 * record['test_nrmse']:.3f}%",
                    flush=True,
                )

    runs = pd.DataFrame.from_records(records)
    current = runs[runs["run_key"].isin(expected_keys)].copy()
    if len(current) != len(expected_keys):
        raise RuntimeError("Phase 6C-1 checkpoint does not contain the full declared matrix.")
    summary = _summarize(current)
    summary.to_csv(data_dir / "ablation_summary.csv", index=False)
    interpretation = _interpret(summary, config)
    payload = {
        "status": "completed",
        "configuration": asdict(config),
        "dataset_audit": audit,
        "run_count": int(len(current)),
        "group_count": int(len(summary)),
        "summary": summary.to_dict("records"),
        "interpretation": interpretation,
    }
    metrics_path = metrics_dir / "phase6c1_metrics.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(root / "outputs" / "phase6c1_report.md", payload, summary)
    return payload

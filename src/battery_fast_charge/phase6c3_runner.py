"""Train and validate pure, structured-delta, and projected Phase 6C controllers."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter, perf_counter_ns
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .ann_model import TinyANN
from .closed_loop import Chen2020DFNPlant, _correct_reduced_state_from_dfn, initial_reduced_state
from .mpc import ReducedBatteryModel
from .phase6_closed_loop import pure_dnn_features
from .phase6_config import FEATURE_NAMES
from .phase6b_runner import _load_context, _nominal_result, _project_current, simulate_dnn_dfn_closed_loop
from .phase6c1_runner import file_sha256
from .phase6c3_config import PhaseSixC3Config


CONTROLLERS = ("pure_dnn", "structured_delta_dnn", "projected_dnn")


def _regression_metrics(target: np.ndarray, prediction: np.ndarray, scale: float) -> dict[str, float]:
    error = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "mae_a": float(np.mean(np.abs(error))),
        "rmse_a": rmse,
        "nrmse": rmse / scale,
        "maximum_absolute_error_a": float(np.max(np.abs(error))),
        "bias_a": float(np.mean(error)),
    }


def structured_current_prediction(
    model: TinyANN,
    features: np.ndarray,
    maximum_change_a: float,
) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values.reshape(1, -1)
    latent = np.asarray(model.predict_unclipped(values), dtype=float).reshape(-1)
    current = values[:, -1] + maximum_change_a * np.tanh(latent)
    return current[0] if one_dimensional else current


def _fit_model(
    train: pd.DataFrame,
    config: PhaseSixC3Config,
    seed: int,
    target_kind: str,
) -> tuple[TinyANN, dict[str, Any]]:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    feature_scaler = StandardScaler().fit(train[list(FEATURE_NAMES)])
    x_train = feature_scaler.transform(train[list(FEATURE_NAMES)])
    if target_kind == "current":
        target = train["teacher_current_a"].to_numpy(dtype=float)
    elif target_kind == "delta_latent":
        maximum_change = config.structured_output.maximum_current_change_a_per_step
        ratio = (
            train["teacher_current_a"].to_numpy(dtype=float)
            - train["state_previous_current_a"].to_numpy(dtype=float)
        ) / maximum_change
        ratio = np.clip(
            ratio,
            -config.structured_output.inverse_tanh_clip,
            config.structured_output.inverse_tanh_clip,
        )
        target = np.arctanh(ratio)
    else:
        raise ValueError(f"Unknown target kind: {target_kind}")
    target_scaler = StandardScaler().fit(target.reshape(-1, 1))
    y_train = target_scaler.transform(target.reshape(-1, 1)).reshape(-1)
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
        estimator.fit(x_train, y_train)
    model = TinyANN(
        feature_names=FEATURE_NAMES,
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


def _offline_predictions(
    controller: str,
    model: TinyANN,
    frame: pd.DataFrame,
    config: PhaseSixC3Config,
    phase3: Any,
) -> np.ndarray:
    features = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
    if controller == "pure_dnn":
        return np.asarray(model.predict_unclipped(features), dtype=float)
    if controller == "structured_delta_dnn":
        return np.asarray(
            structured_current_prediction(
                model, features, config.structured_output.maximum_current_change_a_per_step
            ),
            dtype=float,
        )
    if controller == "projected_dnn":
        raw = np.asarray(model.predict_unclipped(features), dtype=float)
        previous = frame["state_previous_current_a"].to_numpy(dtype=float)
        lower = np.maximum(0.0, previous - phase3.constraints.maximum_current_change_a_per_step)
        upper = np.minimum(
            phase3.constraints.maximum_current_a,
            previous + phase3.constraints.maximum_current_change_a_per_step,
        )
        return np.minimum(np.maximum(raw, lower), upper)
    raise ValueError(f"Unknown controller: {controller}")


def simulate_structured_dfn_closed_loop(
    model: TinyANN,
    reduced_model: ReducedBatteryModel,
    phase3: Any,
    maximum_simulation_time_s: float,
    maximum_change_a: float,
) -> pd.DataFrame:
    plant = Chen2020DFNPlant(phase3)
    state = initial_reduced_state(phase3)
    records: list[dict[str, Any]] = [
        {
            "time_s": 0.0,
            "charge_current_a": 0.0,
            "latent_delta_output": 0.0,
            "soc": state.soc,
            "terminal_voltage_v": reduced_model.ocv(state.soc),
            "average_temperature_c": reduced_model.average_temperature(state),
            "dnn_inference_time_s": 0.0,
            "source": "chen2020_dfn_structured_delta_dnn",
        }
    ]
    steps = int(np.ceil(maximum_simulation_time_s / phase3.control.control_interval_s))
    for _ in range(steps):
        features = pure_dnn_features(reduced_model, state)
        started = perf_counter_ns()
        latent = float(model.predict_unclipped(features))
        current = float(state.previous_current_a + maximum_change_a * np.tanh(latent))
        inference_time = (perf_counter_ns() - started) * 1.0e-9
        if not np.isfinite(current) or abs(current) > 50.0:
            break
        predicted_state, _ = reduced_model.step(state, current)
        measurement = plant.step(current)
        state = _correct_reduced_state_from_dfn(
            predicted_state, measurement, reduced_model, current
        )
        records.append(
            {
                **measurement,
                "charge_current_a": current,
                "latent_delta_output": latent,
                "dnn_inference_time_s": inference_time,
                "source": "chen2020_dfn_structured_delta_dnn",
            }
        )
        if state.soc >= phase3.battery.target_soc - phase3.validation.target_soc_tolerance:
            break
    return pd.DataFrame.from_records(records)


def _flatten_result(
    seed: int,
    controller: str,
    offline: dict[str, dict[str, float]],
    closed_loop: dict[str, Any],
    optimization: dict[str, Any],
    intervention_fraction: float,
) -> dict[str, Any]:
    result = {
        "run_key": f"{controller}__seed-{seed}",
        "controller": controller,
        "initialization_seed": seed,
        "fit_time_s": optimization["fit_time_s"],
        "optimization_iterations": optimization["optimization_iterations"],
        "optimizer_reached_iteration_limit": optimization["reached_iteration_limit"],
        "warning_count": optimization["warning_count"],
        "projection_intervention_fraction": intervention_fraction,
    }
    for split, metrics in offline.items():
        result.update({f"offline_{split}_{name}": value for name, value in metrics.items()})
    result.update(
        {
            "closed_loop_current_nrmse": closed_loop["comparison"]["current_nrmse"],
            "closed_loop_charge_time_gap_fraction": closed_loop["comparison"]["charge_time_gap_fraction"],
            "closed_loop_inference_speedup": closed_loop["comparison"]["inference_speedup_over_mpc"],
            "closed_loop_reached_target_soc": closed_loop["closed_loop"]["reached_target_soc"],
            "closed_loop_charge_time_min": closed_loop["closed_loop"]["charge_time_min"],
            "closed_loop_serious_physical_violation": closed_loop["closed_loop"]["serious_physical_violation"],
            "closed_loop_maximum_current_change_a": closed_loop["closed_loop"]["maximum_current_change_a"],
            "closed_loop_voltage_violation_v": closed_loop["closed_loop"]["voltage_violation_v"],
            "closed_loop_temperature_violation_c": closed_loop["closed_loop"]["temperature_violation_c"],
            "closed_loop_current_violation_a": closed_loop["closed_loop"]["current_violation_a"],
            "closed_loop_current_change_violation_a": closed_loop["closed_loop"]["current_change_violation_a"],
        }
    )
    return result


def _acceptance(record: pd.Series, config: PhaseSixC3Config) -> bool:
    criteria = config.success_criteria
    return bool(
        record["offline_frozen_test_nrmse"] < criteria.maximum_nominal_current_nrmse
        and record["closed_loop_current_nrmse"] < criteria.maximum_nominal_current_nrmse
        and record["closed_loop_charge_time_gap_fraction"] < criteria.maximum_nominal_charge_time_gap_fraction
        and record["closed_loop_inference_speedup"] > criteria.minimum_inference_speedup_over_mpc
        and bool(record["closed_loop_reached_target_soc"])
        and not bool(record["closed_loop_serious_physical_violation"])
    )


def _summary(runs: pd.DataFrame, config: PhaseSixC3Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    acceptance: dict[str, Any] = {}
    for controller, group in runs.groupby("controller", sort=False):
        passes = group.apply(lambda row: _acceptance(row, config), axis=1)
        record: dict[str, Any] = {
            "controller": controller,
            "seed_count": int(len(group)),
            "passing_seed_count": int(passes.sum()),
            "majority_passed": bool(passes.sum() > len(group) / 2),
        }
        for metric in (
            "offline_combined_train_nrmse",
            "offline_original_validation_nrmse",
            "offline_new_validation_nrmse",
            "offline_frozen_test_nrmse",
            "offline_frozen_test_maximum_absolute_error_a",
            "closed_loop_current_nrmse",
            "closed_loop_charge_time_gap_fraction",
            "closed_loop_inference_speedup",
            "projection_intervention_fraction",
        ):
            values = group[metric].to_numpy(dtype=float)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_std"] = float(np.std(values, ddof=1))
            record[f"{metric}_best"] = float(
                np.max(values) if metric == "closed_loop_inference_speedup" else np.min(values)
            )
        records.append(record)
        acceptance[str(controller)] = {
            "passing_seed_count": int(passes.sum()),
            "seed_count": int(len(group)),
            "majority_passed": bool(passes.sum() > len(group) / 2),
            "passing_seeds": group.loc[passes, "initialization_seed"].astype(int).tolist(),
        }
    summary = pd.DataFrame.from_records(records)
    return summary, acceptance


def _write_report(path: Path, payload: dict[str, Any], summary: pd.DataFrame) -> None:
    lines = [
        "# Phase 6C-3: controller-output comparison and 25 °C nominal validation",
        "",
        "## Data isolation",
        "",
        f"- Frozen Phase 6B samples: {payload['data_audit']['frozen_sample_count']}",
        f"- Added Phase 6C training samples: {payload['data_audit']['new_train_sample_count']}",
        f"- New validation samples: {payload['data_audit']['new_validation_sample_count']}",
        "- The original 704-sample test set remains unchanged and is used for every seed.",
        "",
        "## Five-seed comparison",
        "",
        "| Controller | Train NRMSE | Original validation NRMSE | New validation NRMSE | Frozen test NRMSE | 25 °C closed-loop NRMSE | Time gap | Passing seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['controller']} | "
            f"{100 * row['offline_combined_train_nrmse_mean']:.3f}% ± {100 * row['offline_combined_train_nrmse_std']:.3f}% | "
            f"{100 * row['offline_original_validation_nrmse_mean']:.3f}% ± {100 * row['offline_original_validation_nrmse_std']:.3f}% | "
            f"{100 * row['offline_new_validation_nrmse_mean']:.3f}% ± {100 * row['offline_new_validation_nrmse_std']:.3f}% | "
            f"{100 * row['offline_frozen_test_nrmse_mean']:.3f}% ± {100 * row['offline_frozen_test_nrmse_std']:.3f}% | "
            f"{100 * row['closed_loop_current_nrmse_mean']:.3f}% ± {100 * row['closed_loop_current_nrmse_std']:.3f}% | "
            f"{100 * row['closed_loop_charge_time_gap_fraction_mean']:.3f}% | {int(row['passing_seed_count'])}/{int(row['seed_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Frozen-test maximum absolute error",
            "",
            "| Controller | Mean ± SD | Best seed |",
            "|---|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['controller']} | "
            f"{row['offline_frozen_test_maximum_absolute_error_a_mean']:.3f} ± "
            f"{row['offline_frozen_test_maximum_absolute_error_a_std']:.3f} A | "
            f"{row['offline_frozen_test_maximum_absolute_error_a_best']:.3f} A |"
        )
    lines.extend(
        [
            "",
            "## Gate decision",
            "",
            f"- 25 °C nominal majority gate passed: {payload['phase6c_acceptance']['majority_controller_count'] > 0}",
            f"- Controllers with a passing majority: {payload['phase6c_acceptance']['controllers_with_passing_majority']}",
            f"- Proceed to Phase 6D: {payload['phase6c_acceptance']['proceed_to_phase6d']}",
            "",
            "The structured-delta route is an explicit improvement method and is not counted as a pure paper-style DNN result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase_six_c3(config: PhaseSixC3Config, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    data_dir = root / "data" / "phase6c_constraint_regime_learning" / "c3_controller_comparison"
    model_dir = root / "outputs" / "models" / "phase6c3"
    metrics_dir = root / "outputs" / "metrics"
    for directory in (data_dir, model_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frozen_path = root / config.data.frozen_phase6b_dataset
    new_path = root / config.data.phase6c2_dataset
    if file_sha256(frozen_path).lower() != config.data.frozen_phase6b_dataset_sha256.lower():
        raise RuntimeError("Frozen Phase 6B dataset hash changed.")
    if file_sha256(new_path).lower() != config.data.phase6c2_dataset_sha256.lower():
        raise RuntimeError("Phase 6C-2 dataset hash changed.")
    frozen = pd.read_csv(frozen_path)
    new = pd.read_csv(new_path)
    original_train = frozen[frozen["split"] == "train"].copy()
    original_validation = frozen[frozen["split"] == "validation"].copy()
    frozen_test = frozen[frozen["split"] == "test"].copy()
    new_train = new[new["split"] == "phase6c_train"].copy()
    new_validation = new[new["split"] == "phase6c_validation"].copy()
    combined_train = pd.concat([original_train, new_train], ignore_index=True)
    evaluation_frames = {
        "combined_train": combined_train,
        "original_validation": original_validation,
        "new_validation": new_validation,
        "frozen_test": frozen_test,
    }
    data_audit = {
        "frozen_sample_count": int(len(frozen)),
        "original_train_sample_count": int(len(original_train)),
        "original_validation_sample_count": int(len(original_validation)),
        "frozen_test_sample_count": int(len(frozen_test)),
        "frozen_test_trajectory_count": int(frozen_test["trajectory_id"].nunique()),
        "new_train_sample_count": int(len(new_train)),
        "new_validation_sample_count": int(len(new_validation)),
        "combined_train_sample_count": int(len(combined_train)),
    }

    phase3, parameters, ocv_function = _load_context(config, root)
    reduced_model = ReducedBatteryModel(phase3, ocv_function, parameters)
    teacher_frame = pd.read_csv(root / config.nominal_validation.teacher_trajectory)
    teacher_metrics = json.loads(
        (root / config.nominal_validation.teacher_metrics).read_text(encoding="utf-8")
    )["dfn_closed_loop"]
    runs_path = data_dir / "controller_seed_runs.csv"
    records = pd.read_csv(runs_path).to_dict("records") if runs_path.exists() else []
    completed_seeds = {
        int(seed)
        for seed, group in pd.DataFrame.from_records(records).groupby("initialization_seed")
        if set(group["controller"]) == set(CONTROLLERS)
    } if records else set()

    for seed in config.network.initialization_seeds:
        if seed in completed_seeds:
            continue
        pure_model, pure_optimization = _fit_model(combined_train, config, seed, "current")
        delta_model, delta_optimization = _fit_model(combined_train, config, seed, "delta_latent")
        pure_model.save(model_dir / f"pure_dnn_seed-{seed}.npz")
        delta_model.save(model_dir / f"structured_delta_dnn_seed-{seed}.npz")

        offline_by_controller: dict[str, dict[str, dict[str, float]]] = {}
        for controller, model in (
            ("pure_dnn", pure_model),
            ("projected_dnn", pure_model),
            ("structured_delta_dnn", delta_model),
        ):
            offline_by_controller[controller] = {
                split: _regression_metrics(
                    frame["teacher_current_a"].to_numpy(dtype=float),
                    _offline_predictions(controller, model, frame, config, phase3),
                    config.nominal_validation.current_nrmse_normalization_a,
                )
                for split, frame in evaluation_frames.items()
            }

        pure_frame = simulate_dnn_dfn_closed_loop(
            pure_model, reduced_model, phase3, config.nominal_validation.maximum_simulation_time_s, projected=False
        )
        projected_frame = simulate_dnn_dfn_closed_loop(
            pure_model, reduced_model, phase3, config.nominal_validation.maximum_simulation_time_s, projected=True
        )
        structured_frame = simulate_structured_dfn_closed_loop(
            delta_model,
            reduced_model,
            phase3,
            config.nominal_validation.maximum_simulation_time_s,
            config.structured_output.maximum_current_change_a_per_step,
        )
        frames = {
            "pure_dnn": pure_frame,
            "projected_dnn": projected_frame,
            "structured_delta_dnn": structured_frame,
        }
        for controller, frame in frames.items():
            frame.to_csv(data_dir / f"{controller}_dfn_25c_seed-{seed}.csv", index=False)
        closed = {
            controller: _nominal_result(frame, phase3, teacher_frame, teacher_metrics, config)
            for controller, frame in frames.items()
        }
        projection_fraction = float(
            (projected_frame["projection_adjustment_a"].abs() > 1.0e-12).mean()
        )
        seed_records = [
            _flatten_result(
                seed,
                "pure_dnn",
                offline_by_controller["pure_dnn"],
                closed["pure_dnn"],
                pure_optimization,
                0.0,
            ),
            _flatten_result(
                seed,
                "projected_dnn",
                offline_by_controller["projected_dnn"],
                closed["projected_dnn"],
                pure_optimization,
                projection_fraction,
            ),
            _flatten_result(
                seed,
                "structured_delta_dnn",
                offline_by_controller["structured_delta_dnn"],
                closed["structured_delta_dnn"],
                delta_optimization,
                0.0,
            ),
        ]
        records = [record for record in records if int(record["initialization_seed"]) != seed]
        records.extend(seed_records)
        pd.DataFrame.from_records(records).sort_values(["controller", "initialization_seed"]).to_csv(
            runs_path, index=False
        )
        print(
            f"completed seed {seed}: "
            + ", ".join(
                f"{record['controller']} offline={100 * record['offline_frozen_test_nrmse']:.3f}% "
                f"closed={100 * record['closed_loop_current_nrmse']:.3f}%"
                for record in seed_records
            ),
            flush=True,
        )

    runs = pd.DataFrame.from_records(records)
    summary, controller_acceptance = _summary(runs, config)
    summary.to_csv(data_dir / "controller_summary.csv", index=False)
    controllers_with_majority = [
        controller for controller, values in controller_acceptance.items() if values["majority_passed"]
    ]
    phase6c_acceptance = {
        "controllers": controller_acceptance,
        "controllers_with_passing_majority": controllers_with_majority,
        "majority_controller_count": len(controllers_with_majority),
        "pure_paper_method_passed": controller_acceptance["pure_dnn"]["majority_passed"],
        "proceed_to_phase6d": bool(controllers_with_majority),
    }
    payload = {
        "status": "completed",
        "configuration": asdict(config),
        "data_audit": data_audit,
        "summary": summary.to_dict("records"),
        "phase6c_acceptance": phase6c_acceptance,
    }
    (metrics_dir / "phase6c3_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(root / "outputs" / "phase6c3_report.md", payload, summary)
    return payload

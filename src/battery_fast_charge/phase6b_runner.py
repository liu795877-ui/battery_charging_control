"""Phase 6B: diagnose why the pure DNN does not imitate the MPC teacher."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np
import pandas as pd

from .ann_model import TinyANN
from .closed_loop import (
    Chen2020DFNPlant,
    _correct_reduced_state_from_dfn,
    initial_reduced_state,
)
from .identification import build_ocv_function
from .mpc import ReducedBatteryModel, ReducedState
from .paper_method import (
    generate_initial_state_design,
    generate_paper_teacher_dataset,
    train_paper_dnn,
)
from .phase3_config import PhaseThreeConfig, load_phase_three_config
from .phase6_closed_loop import (
    compare_with_teacher,
    pure_dnn_closed_loop_metrics,
    pure_dnn_features,
    temperature_anchor_config,
)
from .phase6_plotting import plot_paper_dataset_audit, plot_paper_dnn_offline
from .phase6b_config import PhaseSixBConfig
from .phase6b_plotting import (
    plot_phase6b_closed_loop_comparison,
    plot_phase6b_error_partitions,
)


def _load_context(config: PhaseSixBConfig, root: Path):
    phase3 = load_phase_three_config(root / config.source_phase3_config)
    validation = json.loads(
        (root / phase3.artifacts.validation_metrics).read_text(encoding="utf-8")
    )
    if not validation.get("success", False):
        raise RuntimeError("Phase 2 reduced model validation is not successful.")
    parameters = json.loads(
        (root / phase3.artifacts.identified_parameters).read_text(encoding="utf-8")
    )
    ocv_function = build_ocv_function(pd.read_csv(root / phase3.artifacts.ocv_curve))
    return phase3, parameters, ocv_function


def _regression_metrics(
    target: np.ndarray, prediction: np.ndarray, scale: float
) -> dict[str, float]:
    error = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    if len(error) == 0:
        return {
            "sample_count": 0,
            "mae_a": float("nan"),
            "rmse_a": float("nan"),
            "nrmse": float("nan"),
            "bias_a": float("nan"),
            "maximum_absolute_error_a": float("nan"),
        }
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "sample_count": int(len(error)),
        "mae_a": float(np.mean(np.abs(error))),
        "rmse_a": rmse,
        "nrmse": rmse / scale,
        "bias_a": float(np.mean(error)),
        "maximum_absolute_error_a": float(np.max(np.abs(error))),
    }


def _append_partition(
    records: list[dict[str, Any]],
    frame: pd.DataFrame,
    family: str,
    label_column: str,
    scale: float,
) -> None:
    for split, split_frame in frame.groupby("split", observed=False):
        for label, group in split_frame.groupby(label_column, observed=False):
            if group.empty:
                continue
            records.append(
                {
                    "split": str(split),
                    "partition_family": family,
                    "partition_label": str(label),
                    **_regression_metrics(
                        group["teacher_current_a"].to_numpy(dtype=float),
                        group["dnn_current_a"].to_numpy(dtype=float),
                        scale,
                    ),
                }
            )


def build_error_partition_table(
    predictions: pd.DataFrame,
    phase3: PhaseThreeConfig,
    config: PhaseSixBConfig,
) -> pd.DataFrame:
    """Group offline DNN errors by state region and active constraint status."""
    frame = predictions.copy()
    scale = config.nominal_validation.current_nrmse_normalization_a
    delta = phase3.constraints.maximum_current_change_a_per_step
    frame["teacher_current_change_a"] = (
        frame["teacher_current_a"] - frame["state_previous_current_a"]
    )
    frame["teacher_slew_margin_a"] = delta - frame["teacher_current_change_a"].abs()
    frame["near_slew_boundary"] = (
        frame["teacher_slew_margin_a"] <= config.diagnostics.slew_margin_close_a
    )
    frame["near_any_constraint"] = frame[
        [
            "active_voltage_constraint",
            "active_temperature_constraint",
            "active_current_upper_constraint",
            "active_current_change_constraint",
        ]
    ].any(axis=1)
    frame["soc_bin"] = pd.cut(
        frame["state_soc"], bins=config.diagnostics.soc_bins, include_lowest=True
    )
    frame["temperature_bin_c"] = pd.cut(
        frame["state_average_temperature_c"],
        bins=config.diagnostics.temperature_bins_c,
        include_lowest=True,
    )
    frame["previous_current_bin_a"] = pd.cut(
        frame["state_previous_current_a"],
        bins=config.diagnostics.previous_current_bins_a,
        include_lowest=True,
    )
    frame["slew_active"] = frame["active_current_change_constraint"].map(
        {True: "slew_active", False: "slew_inactive"}
    )
    frame["slew_near"] = frame["near_slew_boundary"].map(
        {True: "near_slew_boundary", False: "away_from_slew_boundary"}
    )
    frame["any_constraint"] = frame["near_any_constraint"].map(
        {True: "near_any_constraint", False: "not_near_constraint"}
    )
    frame["voltage_active"] = frame["active_voltage_constraint"].map(
        {True: "voltage_active", False: "voltage_inactive"}
    )
    frame["temperature_active"] = frame["active_temperature_constraint"].map(
        {True: "temperature_active", False: "temperature_inactive"}
    )
    frame["current_upper_active"] = frame["active_current_upper_constraint"].map(
        {True: "current_upper_active", False: "current_upper_inactive"}
    )

    records: list[dict[str, Any]] = []
    for family, column in (
        ("soc", "soc_bin"),
        ("temperature", "temperature_bin_c"),
        ("previous_current", "previous_current_bin_a"),
        ("slew_active", "slew_active"),
        ("slew_near_boundary", "slew_near"),
        ("any_constraint", "any_constraint"),
        ("voltage_active", "voltage_active"),
        ("temperature_active", "temperature_active"),
        ("current_upper_active", "current_upper_active"),
    ):
        _append_partition(records, frame, family, column, scale)
    return pd.DataFrame.from_records(records).sort_values(
        ["split", "partition_family", "rmse_a"], ascending=[True, True, False]
    )


def _paper_dataset_metrics_from_frames(
    attempts: pd.DataFrame,
    dataset: pd.DataFrame,
    phase3: PhaseThreeConfig,
    config: PhaseSixBConfig,
) -> dict[str, Any]:
    """Rebuild dataset-gate metrics when reusing cached MPC teacher CSV files."""
    accepted_ids = dataset["trajectory_id"].drop_duplicates().tolist() if len(dataset) else []
    acceptance = float(attempts["teacher_accepted"].mean()) if len(attempts) else 0.0
    split_counts = dataset["split"].value_counts().to_dict() if len(dataset) else {}
    trajectory_counts = (
        dataset[["trajectory_id", "split"]].drop_duplicates()["split"].value_counts().to_dict()
        if len(dataset)
        else {}
    )
    metrics = {
        "attempted_initial_state_count": int(len(attempts)),
        "accepted_initial_state_count": int(len(accepted_ids)),
        "teacher_acceptance_fraction": acceptance,
        "unfolded_sample_count": int(len(dataset)),
        "split_sample_counts": {str(k): int(v) for k, v in split_counts.items()},
        "split_trajectory_counts": {str(k): int(v) for k, v in trajectory_counts.items()},
        "duplicate_feature_row_count": int(dataset.duplicated(list(config.paper_method.state_ranges)).sum()) if len(dataset) else 0,
        "active_constraint_counts": {
            "voltage": int(dataset["active_voltage_constraint"].sum()) if len(dataset) else 0,
            "temperature": int(dataset["active_temperature_constraint"].sum()) if len(dataset) else 0,
            "current_upper": int(dataset["active_current_upper_constraint"].sum()) if len(dataset) else 0,
            "current_change": int(dataset["active_current_change_constraint"].sum()) if len(dataset) else 0,
        },
        "mean_teacher_solve_time_ms": float(attempts["teacher_solve_time_s"].mean() * 1000.0) if len(attempts) else 0.0,
        "maximum_teacher_solve_time_ms": float(attempts["teacher_solve_time_s"].max() * 1000.0) if len(attempts) else 0.0,
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
    metrics["source"] = "cached_csv"
    return metrics


def _project_current(raw_current: float, previous_current: float, phase3: PhaseThreeConfig) -> float:
    constraints = phase3.constraints
    current = min(max(raw_current, 0.0), constraints.maximum_current_a)
    lower = max(0.0, previous_current - constraints.maximum_current_change_a_per_step)
    upper = min(
        constraints.maximum_current_a,
        previous_current + constraints.maximum_current_change_a_per_step,
    )
    return float(min(max(current, lower), upper))


def _dnn_current(
    ann: TinyANN,
    model: ReducedBatteryModel,
    state: ReducedState,
    phase3: PhaseThreeConfig,
    projected: bool,
) -> tuple[float, float, float]:
    start = perf_counter_ns()
    raw_current = float(ann.predict_unclipped(pure_dnn_features(model, state)))
    elapsed = (perf_counter_ns() - start) * 1.0e-9
    applied = _project_current(raw_current, state.previous_current_a, phase3) if projected else raw_current
    return raw_current, applied, elapsed


def simulate_dnn_dfn_closed_loop(
    ann: TinyANN,
    model: ReducedBatteryModel,
    phase3: PhaseThreeConfig,
    maximum_simulation_time_s: float,
    projected: bool,
) -> pd.DataFrame:
    """Run either pure DNN or output-projected DNN on the Chen2020 DFN plant."""
    plant = Chen2020DFNPlant(phase3)
    state = initial_reduced_state(phase3)
    source = "chen2020_dfn_projected_dnn" if projected else "chen2020_dfn_pure_dnn"
    records: list[dict[str, Any]] = [
        {
            "time_s": 0.0,
            "charge_current_a": 0.0,
            "raw_dnn_current_a": 0.0,
            "projection_adjustment_a": 0.0,
            "soc": state.soc,
            "terminal_voltage_v": model.ocv(state.soc),
            "average_temperature_c": model.average_temperature(state),
            "dnn_inference_time_s": 0.0,
            "source": source,
        }
    ]
    steps = int(np.ceil(maximum_simulation_time_s / phase3.control.control_interval_s))
    for _ in range(steps):
        raw_current, current, inference_time = _dnn_current(
            ann, model, state, phase3, projected
        )
        if not np.isfinite(raw_current) or abs(raw_current) > 50.0:
            break
        predicted_state, _ = model.step(state, current)
        measurement = plant.step(current)
        state = _correct_reduced_state_from_dfn(
            predicted_state, measurement, model, current
        )
        records.append(
            {
                **measurement,
                "charge_current_a": current,
                "raw_dnn_current_a": raw_current,
                "projection_adjustment_a": current - raw_current,
                "dnn_inference_time_s": inference_time,
                "source": source,
            }
        )
        if state.soc >= phase3.battery.target_soc - phase3.validation.target_soc_tolerance:
            break
    return pd.DataFrame.from_records(records)


def _nominal_result(
    frame: pd.DataFrame,
    phase3: PhaseThreeConfig,
    teacher_frame: pd.DataFrame,
    teacher_metrics: dict[str, Any],
    config: PhaseSixBConfig,
) -> dict[str, Any]:
    metrics = pure_dnn_closed_loop_metrics(frame, phase3, config)
    comparison = compare_with_teacher(frame, metrics, teacher_frame, teacher_metrics, config)
    return {"closed_loop": metrics, "comparison": comparison}


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    offline = payload["offline_dnn"]["split_metrics"]["test"]
    pure = payload["nominal_25c"]["pure_dnn"]
    projected = payload["nominal_25c"]["projected_dnn"]
    worst = payload["diagnostics"]["worst_test_partitions"][:8]
    lines = [
        "# Phase 6B: why the pure DNN did not learn MPC well",
        "",
        "## Main result",
        "",
        "Phase 6B is diagnostic. It does not replace the Phase 6A pure-DNN conclusion.",
        f"- Accepted teacher trajectories: {payload['paper_dataset']['accepted_initial_state_count']}",
        f"- Unfolded samples: {payload['paper_dataset']['unfolded_sample_count']}",
        f"- Selected DNN architecture: {payload['offline_dnn']['architecture']}",
        f"- Selected optimizer iterations: {payload['offline_dnn']['selected_optimization_iterations']}",
        f"- Selected optimizer reached limit: {payload['offline_dnn']['selected_optimizer_reached_iteration_limit']}",
        f"- Test current NRMSE: {100 * offline['nrmse']:.3f}%",
        f"- Pure DNN closed-loop NRMSE: {100 * pure['comparison']['current_nrmse']:.3f}%",
        f"- Projected DNN closed-loop NRMSE: {100 * projected['comparison']['current_nrmse']:.3f}%",
        f"- Pure DNN slew violation: {pure['closed_loop']['current_change_violation_a']:.4f} A",
        f"- Projected DNN slew violation: {projected['closed_loop']['current_change_violation_a']:.4f} A",
        "",
        "## Worst held-out error partitions",
        "",
    ]
    for row in worst:
        lines.append(
            f"- {row['partition_family']} / {row['partition_label']}: "
            f"RMSE {row['rmse_a']:.4f} A, NRMSE {100 * row['nrmse']:.3f}%, "
            f"n={row['sample_count']}"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "If projection fixes only constraint violations but not current NRMSE, the network did not learn the teacher map accurately. "
            "If projection also reduces NRMSE materially, part of the Phase 6A error came from raw outputs that ignored basic controller bounds.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase_six_b(config: PhaseSixBConfig, project_root: str | Path) -> dict[str, Any]:
    """Run the independent Phase 6B diagnosis experiment."""
    root = Path(project_root)
    data_dir = root / "data" / "phase6b_dnn_failure_diagnosis"
    model_dir = root / "outputs" / "models"
    metrics_dir = root / "outputs" / "metrics"
    figures_dir = root / "outputs" / "figures"
    for directory in (data_dir, model_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    phase3, parameters, ocv_function = _load_context(config, root)
    reduced_model = ReducedBatteryModel(phase3, ocv_function, parameters)
    attempts_path = data_dir / "initial_state_audit.csv"
    dataset_path = data_dir / "paper_teacher_dataset.csv"
    if attempts_path.exists() and dataset_path.exists():
        attempts = pd.read_csv(attempts_path)
        dataset = pd.read_csv(dataset_path)
        if len(attempts) != config.paper_method.initial_state_count:
            design = generate_initial_state_design(config)
            attempts, dataset, dataset_metrics = generate_paper_teacher_dataset(
                design, reduced_model, phase3, config
            )
            attempts.to_csv(attempts_path, index=False)
            dataset.to_csv(dataset_path, index=False)
        else:
            dataset_metrics = _paper_dataset_metrics_from_frames(
                attempts, dataset, phase3, config
            )
    else:
        design = generate_initial_state_design(config)
        attempts, dataset, dataset_metrics = generate_paper_teacher_dataset(
            design, reduced_model, phase3, config
        )
        attempts.to_csv(attempts_path, index=False)
        dataset.to_csv(dataset_path, index=False)
    plot_paper_dataset_audit(
        attempts,
        dataset,
        figures_dir / "phase6b_dataset_audit.png",
        title_prefix="Phase 6B",
    )

    payload: dict[str, Any] = {
        "configuration": asdict(config),
        "paper_dataset": dataset_metrics,
    }
    if not dataset_metrics["success"]:
        payload["status"] = "dataset_gate_failed"
        (metrics_dir / "phase6b_metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return payload

    ann, selection, predictions, offline_metrics = train_paper_dnn(dataset, config)
    selected_architecture = "-".join(str(v) for v in offline_metrics["architecture"][1:-1])
    selected_row = selection[
        (selection["architecture"] == selected_architecture)
        & (
            selection["regularization_alpha"]
            == offline_metrics["selected_regularization_alpha"]
        )
        & (
            selection["initialization_seed"]
            == offline_metrics["selected_initialization_seed"]
        )
    ].iloc[0]
    offline_metrics["selected_optimization_iterations"] = int(
        selected_row["optimization_iterations"]
    )
    offline_metrics["selected_optimizer_reached_iteration_limit"] = bool(
        selected_row["optimization_iterations"] >= config.network.maximum_iterations
    )
    model_path = ann.save(model_dir / "phase6b_paper_dnn.npz")
    selection.to_csv(data_dir / "network_selection.csv", index=False)
    predictions.to_csv(data_dir / "offline_predictions.csv", index=False)
    plot_paper_dnn_offline(
        predictions, figures_dir / "phase6b_offline_fit.png", title_prefix="Phase 6B"
    )

    partitions = build_error_partition_table(predictions, phase3, config)
    partitions.to_csv(data_dir / "error_partition_diagnostics.csv", index=False)
    plot_phase6b_error_partitions(
        partitions, figures_dir / "phase6b_error_partitions.png"
    )

    nominal = temperature_anchor_config(
        phase3,
        config.nominal_validation.temperature_c,
        config.nominal_validation.maximum_simulation_time_s,
    )
    nominal_model = ReducedBatteryModel(nominal, ocv_function, parameters)
    reloaded = TinyANN.load(model_path)
    pure_frame = simulate_dnn_dfn_closed_loop(
        reloaded,
        nominal_model,
        nominal,
        config.nominal_validation.maximum_simulation_time_s,
        projected=False,
    )
    projected_frame = simulate_dnn_dfn_closed_loop(
        reloaded,
        nominal_model,
        nominal,
        config.nominal_validation.maximum_simulation_time_s,
        projected=True,
    )
    pure_frame.to_csv(data_dir / "pure_dnn_dfn_25c.csv", index=False)
    projected_frame.to_csv(data_dir / "projected_dnn_dfn_25c.csv", index=False)
    teacher_frame = pd.read_csv(root / config.nominal_validation.teacher_trajectory)
    teacher_metrics = json.loads(
        (root / config.nominal_validation.teacher_metrics).read_text(encoding="utf-8")
    )["dfn_closed_loop"]
    nominal_payload = {
        "pure_dnn": _nominal_result(
            pure_frame, nominal, teacher_frame, teacher_metrics, config
        ),
        "projected_dnn": _nominal_result(
            projected_frame, nominal, teacher_frame, teacher_metrics, config
        ),
        "teacher": teacher_metrics,
    }
    plot_phase6b_closed_loop_comparison(
        teacher_frame,
        pure_frame,
        projected_frame,
        figures_dir / "phase6b_pure_vs_projected_25c.png",
    )

    test_partitions = partitions[partitions["split"] == "test"].sort_values(
        "rmse_a", ascending=False
    )
    payload.update(
        {
            "status": "completed",
            "offline_dnn": offline_metrics,
            "diagnostics": {
                "worst_test_partitions": test_partitions.head(12).to_dict("records"),
            },
            "nominal_25c": nominal_payload,
        }
    )
    (metrics_dir / "phase6b_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(root / "outputs" / "phase6b_report.md", payload)
    return payload

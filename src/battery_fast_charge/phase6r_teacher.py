"""Rolling-horizon first-action teacher and data audit for Phase 6R."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .mpc import ConstrainedMPC, ReducedBatteryModel, ReducedState
from .paper_method import hammersley_points
from .phase6b_runner import _load_context
from .phase6r_config import DESIGN_FEATURES, PhaseSixRConfig, ROLLING_STATE_FEATURES


def design_initial_states(config: PhaseSixRConfig, ambient_temperature_c: float) -> pd.DataFrame:
    count = config.teacher_data.initial_trajectory_count
    unit = hammersley_points(count, len(DESIGN_FEATURES))
    values = np.column_stack(
        [
            low + unit[:, index] * (high - low)
            for index, (low, high) in enumerate(config.teacher_data.initial_state_ranges.values())
        ]
    )
    frame = pd.DataFrame(values, columns=DESIGN_FEATURES)
    frame.insert(0, "trajectory_id", [f"phase6r_ic_{index:04d}" for index in range(count)])
    frame["state_ambient_temperature_c"] = float(ambient_temperature_c)
    return frame


def row_to_rolling_state(row: pd.Series) -> ReducedState:
    return ReducedState(
        soc=float(row["state_soc"]),
        polarization_fast_v=float(row["state_polarization_fast_v"]),
        polarization_slow_v=float(row["state_polarization_slow_v"]),
        core_temperature_c=float(row["state_core_temperature_c"]),
        surface_temperature_c=float(row["state_surface_temperature_c"]),
        previous_current_a=float(row["state_previous_current_a"]),
    )


def state_features(state: ReducedState, ambient_temperature_c: float) -> dict[str, float]:
    return {
        "state_soc": state.soc,
        "state_polarization_fast_v": state.polarization_fast_v,
        "state_polarization_slow_v": state.polarization_slow_v,
        "state_core_temperature_c": state.core_temperature_c,
        "state_surface_temperature_c": state.surface_temperature_c,
        "state_ambient_temperature_c": float(ambient_temperature_c),
        "state_previous_current_a": state.previous_current_a,
    }


def assign_trajectory_splits(ids: list[str], config: PhaseSixRConfig) -> dict[str, str]:
    values = np.asarray(sorted(ids), dtype=object)
    random = np.random.default_rng(config.random_seed)
    random.shuffle(values)
    train_end = int(round(len(values) * config.teacher_data.train_fraction))
    validation_end = train_end + int(round(len(values) * config.teacher_data.validation_fraction))
    mapping = {str(value): "train" for value in values[:train_end]}
    mapping.update({str(value): "validation" for value in values[train_end:validation_end]})
    mapping.update({str(value): "test" for value in values[validation_end:]})
    return mapping


def compare_independent_rolling_teachers(
    initial_state: ReducedState,
    model: ReducedBatteryModel,
    phase3: Any,
    steps: int,
) -> pd.DataFrame:
    """Compare two independent MPC instances under identical rolling state history."""
    generator_teacher = ConstrainedMPC(model, phase3)
    closed_loop_teacher = ConstrainedMPC(model, phase3)
    state = initial_state
    records: list[dict[str, Any]] = []
    for step_index in range(steps):
        generated = generator_teacher.solve(state)
        executed = closed_loop_teacher.solve(state)
        difference = float(generated.current_a - executed.current_a)
        records.append(
            {
                "step_index": step_index,
                "generated_current_a": generated.current_a,
                "closed_loop_current_a": executed.current_a,
                "absolute_difference_a": abs(difference),
                "generated_used_fallback": generated.used_fallback,
                "closed_loop_used_fallback": executed.used_fallback,
            }
        )
        state, _ = model.step(state, generated.current_a)
    return pd.DataFrame.from_records(records)


def run_consistency_audit(
    design: pd.DataFrame,
    model: ReducedBatteryModel,
    phase3: Any,
    config: PhaseSixRConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[pd.DataFrame] = []
    for _, row in design.head(config.teacher_data.consistency_state_count).iterrows():
        comparison = compare_independent_rolling_teachers(
            row_to_rolling_state(row),
            model,
            phase3,
            config.teacher_data.consistency_steps_per_state,
        )
        comparison.insert(0, "trajectory_id", str(row["trajectory_id"]))
        records.append(comparison)
    audit = pd.concat(records, ignore_index=True)
    maximum = float(audit["absolute_difference_a"].max())
    metrics = {
        "state_count": config.teacher_data.consistency_state_count,
        "steps_per_state": config.teacher_data.consistency_steps_per_state,
        "comparison_count": int(len(audit)),
        "maximum_absolute_difference_a": maximum,
        "tolerance_a": config.teacher_data.consistency_tolerance_a,
        "success": bool(maximum <= config.teacher_data.consistency_tolerance_a),
    }
    return audit, metrics


def _trajectory_rows(
    initial_row: pd.Series,
    split: str,
    model: ReducedBatteryModel,
    phase3: Any,
    config: PhaseSixRConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    controller = ConstrainedMPC(model, phase3)
    state = row_to_rolling_state(initial_row)
    rows: list[dict[str, Any]] = []
    accepted = True
    rejection_reason = ""
    solve_times: list[float] = []
    for step_index in range(config.teacher_data.trajectory_steps):
        result = controller.solve(state)
        solve_times.append(result.solve_time_s)
        if not result.optimizer_success or not result.prediction_feasible or result.used_fallback:
            accepted = False
            rejection_reason = result.status
            break
        current = float(result.current_a)
        next_state, output = model.step(state, current)
        features = state_features(state, phase3.battery.ambient_temperature_c)
        rows.append(
            {
                "trajectory_id": str(initial_row["trajectory_id"]),
                "step_index": step_index,
                **features,
                "state_average_temperature_c": model.average_temperature(state),
                "teacher_current_a": current,
                "teacher_solve_time_s": result.solve_time_s,
                "teacher_optimizer_success": result.optimizer_success,
                "teacher_prediction_feasible": result.prediction_feasible,
                "teacher_used_fallback": result.used_fallback,
                "teacher_minimum_constraint_margin": result.minimum_constraint_margin,
                "active_voltage_constraint": bool(
                    result.predicted_maximum_voltage_v >= phase3.constraints.mpc_maximum_voltage_v - 0.01
                ),
                "active_temperature_constraint": bool(
                    result.predicted_maximum_temperature_c >= phase3.constraints.mpc_maximum_temperature_c - 0.10
                ),
                "active_current_upper_constraint": bool(
                    current >= phase3.constraints.maximum_current_a - 0.05
                ),
                "active_current_change_constraint": bool(
                    abs(current - state.previous_current_a)
                    >= phase3.constraints.maximum_current_change_a_per_step - 0.05
                ),
                "next_soc": next_state.soc,
                "next_core_temperature_c": next_state.core_temperature_c,
                "next_surface_temperature_c": next_state.surface_temperature_c,
                "split": split,
            }
        )
        state = next_state
    attempt = {
        **initial_row.to_dict(),
        "split": split,
        "teacher_accepted": accepted and len(rows) == config.teacher_data.trajectory_steps,
        "completed_step_count": len(rows),
        "rejection_reason": rejection_reason,
        "mean_teacher_solve_time_s": float(np.mean(solve_times)) if solve_times else 0.0,
        "maximum_teacher_solve_time_s": float(np.max(solve_times)) if solve_times else 0.0,
    }
    return rows if attempt["teacher_accepted"] else [], attempt


def _dataset_metrics(
    attempts: pd.DataFrame,
    dataset: pd.DataFrame,
    config: PhaseSixRConfig,
    consistency: dict[str, Any],
) -> dict[str, Any]:
    accepted = int(attempts["teacher_accepted"].sum())
    acceptance = float(attempts["teacher_accepted"].mean())
    split_samples = {str(key): int(value) for key, value in dataset["split"].value_counts().items()}
    split_trajectories = {
        str(key): int(value)
        for key, value in dataset[["trajectory_id", "split"]].drop_duplicates()["split"].value_counts().items()
    }
    checks = {
        "teacher_consistency": consistency["success"],
        "minimum_teacher_acceptance": acceptance >= config.teacher_data.minimum_teacher_acceptance_fraction,
        "trajectory_split_isolation": bool(dataset.groupby("trajectory_id")["split"].nunique().max() == 1),
        "all_splits_nonempty": all(split_samples.get(name, 0) > 0 for name in ("train", "validation", "test")),
        "full_state_columns_present": set(ROLLING_STATE_FEATURES).issubset(dataset.columns),
        "only_first_action_per_solve": bool((dataset["teacher_used_fallback"] == False).all()),
    }
    return {
        "attempted_trajectory_count": int(len(attempts)),
        "accepted_trajectory_count": accepted,
        "teacher_acceptance_fraction": acceptance,
        "sample_count": int(len(dataset)),
        "split_sample_counts": split_samples,
        "split_trajectory_counts": split_trajectories,
        "active_constraint_counts": {
            "voltage": int(dataset["active_voltage_constraint"].sum()),
            "temperature": int(dataset["active_temperature_constraint"].sum()),
            "current_upper": int(dataset["active_current_upper_constraint"].sum()),
            "current_change": int(dataset["active_current_change_constraint"].sum()),
        },
        "mean_teacher_solve_time_ms": float(dataset["teacher_solve_time_s"].mean() * 1000.0),
        "maximum_teacher_solve_time_ms": float(dataset["teacher_solve_time_s"].max() * 1000.0),
        "checks": checks,
        "success": bool(all(checks.values())),
    }


def run_phase_six_r_teacher(config: PhaseSixRConfig, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    data_dir = root / "data" / "phase6r_corrected_policy_distillation"
    metrics_dir = root / "outputs" / "metrics"
    data_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    phase3, parameters, ocv_function = _load_context(config, root)
    model = ReducedBatteryModel(phase3, ocv_function, parameters)
    design = design_initial_states(config, phase3.battery.ambient_temperature_c)
    split_map = assign_trajectory_splits(design["trajectory_id"].tolist(), config)
    design["split"] = design["trajectory_id"].map(split_map)
    design.to_csv(data_dir / "initial_state_design.csv", index=False)

    audit, consistency = run_consistency_audit(design, model, phase3, config)
    audit.to_csv(data_dir / "teacher_consistency_audit.csv", index=False)
    if not consistency["success"]:
        raise RuntimeError("Phase 6R teacher consistency gate failed.")

    attempts_path = data_dir / "teacher_attempts.csv"
    dataset_path = data_dir / "rolling_first_action_dataset.csv"
    attempts = pd.read_csv(attempts_path).to_dict("records") if attempts_path.exists() else []
    rows = pd.read_csv(dataset_path).to_dict("records") if dataset_path.exists() else []
    completed = {str(record["trajectory_id"]) for record in attempts}
    for _, initial_row in design.iterrows():
        trajectory_id = str(initial_row["trajectory_id"])
        if trajectory_id in completed:
            continue
        trajectory_rows, attempt = _trajectory_rows(
            initial_row,
            split_map[trajectory_id],
            model,
            phase3,
            config,
        )
        rows.extend(trajectory_rows)
        attempts.append(attempt)
        completed.add(trajectory_id)
        if len(completed) % config.teacher_data.checkpoint_interval_trajectories == 0:
            pd.DataFrame.from_records(attempts).to_csv(attempts_path, index=False)
            pd.DataFrame.from_records(rows).to_csv(dataset_path, index=False)
            accepted_count = sum(bool(record["teacher_accepted"]) for record in attempts)
            print(
                f"completed {len(completed)}/{len(design)} rolling trajectories; accepted={accepted_count}",
                flush=True,
            )
    attempt_frame = pd.DataFrame.from_records(attempts)
    dataset = pd.DataFrame.from_records(rows)
    metrics = _dataset_metrics(attempt_frame, dataset, config, consistency)
    payload = {
        "status": "completed" if metrics["success"] else "teacher_data_gate_failed",
        "configuration": asdict(config),
        "teacher_consistency": consistency,
        "teacher_dataset": metrics,
    }
    (metrics_dir / "phase6r_teacher_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload

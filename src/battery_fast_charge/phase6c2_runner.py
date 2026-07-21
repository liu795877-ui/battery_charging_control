"""Generate targeted-boundary and closed-loop DAgger MPC labels for Phase 6C-2."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .closed_loop import _correct_reduced_state_from_dfn, initial_reduced_state
from .mpc import ReducedBatteryModel
from .paper_method import PaperTrajectoryMPC, hammersley_points, row_to_state
from .phase6_config import FEATURE_NAMES
from .phase6b_runner import _load_context
from .phase6c1_runner import file_sha256
from .phase6c2_config import PhaseSixC2Config


TARGETED_SOURCE = "targeted_boundary_sampling"
DAGGER_SOURCE = "closed_loop_DAgger"


def generate_targeted_boundary_design(config: PhaseSixC2Config) -> pd.DataFrame:
    count = config.sampling.targeted_boundary_trajectory_count
    unit = hammersley_points(count, len(FEATURE_NAMES))
    values = np.column_stack(
        [
            low + unit[:, index] * (high - low)
            for index, (low, high) in enumerate(config.sampling.targeted_ranges.values())
        ]
    )
    frame = pd.DataFrame(values, columns=FEATURE_NAMES)
    frame.insert(0, "sampling_method", TARGETED_SOURCE)
    frame.insert(0, "initial_state_id", [f"phase6c_targeted_{index:04d}" for index in range(count)])
    frame["coverage_distance_standardized"] = np.nan
    return frame


def replay_pure_dnn_states(
    trajectory: pd.DataFrame,
    model: ReducedBatteryModel,
    phase3: Any,
) -> pd.DataFrame:
    state = initial_reduced_state(phase3)
    records: list[dict[str, float]] = []
    control_rows = trajectory[trajectory["time_s"] > 0.0]
    for _, row in control_rows.iterrows():
        records.append(
            {
                "state_soc": state.soc,
                "state_polarization_fast_v": state.polarization_fast_v,
                "state_polarization_slow_v": state.polarization_slow_v,
                "state_average_temperature_c": model.average_temperature(state),
                "state_previous_current_a": state.previous_current_a,
            }
        )
        current = float(row["charge_current_a"])
        predicted, _ = model.step(state, current)
        state = _correct_reduced_state_from_dfn(
            predicted,
            {
                "soc": float(row["soc"]),
                "average_temperature_c": float(row["average_temperature_c"]),
            },
            model,
            current,
        )
    return pd.DataFrame.from_records(records)


def generate_dagger_design(
    replay_states: pd.DataFrame,
    frozen_training: pd.DataFrame,
    config: PhaseSixC2Config,
) -> pd.DataFrame:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    sampling = config.sampling
    random = np.random.default_rng(config.random_seed)
    base_indices = random.integers(0, len(replay_states), size=sampling.dagger_candidate_count)
    candidates = replay_states.iloc[base_indices][list(FEATURE_NAMES)].reset_index(drop=True).copy()
    for feature in FEATURE_NAMES:
        sigma = sampling.dagger_jitter_standard_deviation[feature]
        candidates[feature] += random.normal(0.0, sigma, size=len(candidates))
        low, high = sampling.global_state_bounds[feature]
        candidates[feature] = candidates[feature].clip(low, high)

    scaler = StandardScaler().fit(frozen_training[list(FEATURE_NAMES)])
    reference = scaler.transform(frozen_training[list(FEATURE_NAMES)])
    candidate_values = scaler.transform(candidates[list(FEATURE_NAMES)])
    neighbor = NearestNeighbors(n_neighbors=1).fit(reference)
    distance, _ = neighbor.kneighbors(candidate_values)
    candidates["coverage_distance_standardized"] = distance[:, 0]
    candidates = candidates.sort_values(
        ["coverage_distance_standardized", "state_soc"], ascending=[False, True]
    ).head(sampling.closed_loop_dagger_trajectory_count).reset_index(drop=True)
    candidates.insert(0, "sampling_method", DAGGER_SOURCE)
    candidates.insert(
        0,
        "initial_state_id",
        [f"phase6c_dagger_{index:04d}" for index in range(len(candidates))],
    )
    return candidates


def assign_new_splits(design: pd.DataFrame, config: PhaseSixC2Config) -> dict[str, str]:
    random = np.random.default_rng(config.random_seed + 1)
    mapping: dict[str, str] = {}
    for _, source_frame in design.groupby("sampling_method", sort=True):
        ids = source_frame["initial_state_id"].to_numpy(dtype=object).copy()
        random.shuffle(ids)
        validation_count = max(1, int(round(len(ids) * config.sampling.new_validation_fraction)))
        for value in ids[:validation_count]:
            mapping[str(value)] = "phase6c_validation"
        for value in ids[validation_count:]:
            mapping[str(value)] = "phase6c_train"
    return mapping


def _unfold_accepted_trajectory(
    initial_row: pd.Series,
    split: str,
    result: Any,
    model: ReducedBatteryModel,
    phase3: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    state = row_to_state(initial_row)
    for step_index, current in enumerate(result.currents_a):
        next_state, output = model.step(state, float(current))
        change = float(current - state.previous_current_a)
        rows.append(
            {
                "trajectory_id": str(initial_row["initial_state_id"]),
                "sampling_method": str(initial_row["sampling_method"]),
                "coverage_distance_standardized": float(initial_row["coverage_distance_standardized"])
                if np.isfinite(initial_row["coverage_distance_standardized"])
                else np.nan,
                "step_index": step_index,
                "state_soc": state.soc,
                "state_polarization_fast_v": state.polarization_fast_v,
                "state_polarization_slow_v": state.polarization_slow_v,
                "state_average_temperature_c": model.average_temperature(state),
                "state_previous_current_a": state.previous_current_a,
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
                "split": split,
            }
        )
        state = next_state
    return rows


def _metrics(attempts: pd.DataFrame, dataset: pd.DataFrame, config: PhaseSixC2Config) -> dict[str, Any]:
    accepted = int(attempts["teacher_accepted"].sum())
    acceptance = float(attempts["teacher_accepted"].mean())
    source_metrics: dict[str, Any] = {}
    for source, source_attempts in attempts.groupby("sampling_method"):
        source_data = dataset[dataset["sampling_method"] == source]
        source_metrics[str(source)] = {
            "attempted_trajectory_count": int(len(source_attempts)),
            "accepted_trajectory_count": int(source_attempts["teacher_accepted"].sum()),
            "acceptance_fraction": float(source_attempts["teacher_accepted"].mean()),
            "unfolded_sample_count": int(len(source_data)),
        }
    active_counts = {
        "voltage": int(dataset["active_voltage_constraint"].sum()),
        "temperature": int(dataset["active_temperature_constraint"].sum()),
        "current_upper": int(dataset["active_current_upper_constraint"].sum()),
        "current_change": int(dataset["active_current_change_constraint"].sum()),
    }
    checks = {
        "minimum_acceptance_fraction": acceptance >= config.acceptance.minimum_teacher_acceptance_fraction,
        "minimum_accepted_trajectory_count": accepted >= config.acceptance.minimum_accepted_trajectory_count,
        "trajectory_split_isolation": bool(dataset.groupby("trajectory_id")["split"].nunique().max() == 1),
        "new_splits_only": set(dataset["split"]) == {"phase6c_train", "phase6c_validation"},
        "both_sources_present": set(dataset["sampling_method"]) == {TARGETED_SOURCE, DAGGER_SOURCE},
    }
    return {
        "attempted_trajectory_count": int(len(attempts)),
        "accepted_trajectory_count": accepted,
        "teacher_acceptance_fraction": acceptance,
        "unfolded_sample_count": int(len(dataset)),
        "split_sample_counts": {str(k): int(v) for k, v in dataset["split"].value_counts().items()},
        "split_trajectory_counts": {
            str(k): int(v)
            for k, v in dataset[["trajectory_id", "split"]].drop_duplicates()["split"].value_counts().items()
        },
        "source_metrics": source_metrics,
        "active_constraint_counts": active_counts,
        "mean_teacher_solve_time_ms": float(attempts["teacher_solve_time_s"].mean() * 1000.0),
        "maximum_teacher_solve_time_ms": float(attempts["teacher_solve_time_s"].max() * 1000.0),
        "checks": checks,
        "success": bool(all(checks.values())),
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["teacher_data"]
    lines = [
        "# Phase 6C-2: targeted MPC teacher data",
        "",
        "## Frozen-set protection",
        "",
        f"- Phase 6B dataset SHA-256: `{payload['frozen_baseline_audit']['dataset_sha256']}`",
        f"- Frozen test trajectories: {payload['frozen_baseline_audit']['test_trajectory_count']}",
        "- The original validation and test assignments were not modified.",
        "",
        "## New teacher data",
        "",
        f"- Attempted trajectories: {metrics['attempted_trajectory_count']}",
        f"- Accepted trajectories: {metrics['accepted_trajectory_count']}",
        f"- Acceptance fraction: {100 * metrics['teacher_acceptance_fraction']:.2f}%",
        f"- Unfolded samples: {metrics['unfolded_sample_count']}",
        f"- New split samples: {metrics['split_sample_counts']}",
        f"- Active constraints: {metrics['active_constraint_counts']}",
        "",
        "New samples are labeled by source as `targeted_boundary_sampling` or `closed_loop_DAgger` and enter only the Phase 6C training/new-validation sets.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase_six_c2(config: PhaseSixC2Config, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    data_dir = root / "data" / "phase6c_constraint_regime_learning" / "c2_targeted_teacher"
    metrics_dir = root / "outputs" / "metrics"
    data_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    frozen_path = root / config.frozen_baseline.dataset
    pure_path = root / config.frozen_baseline.pure_dnn_trajectory
    frozen_digest = file_sha256(frozen_path)
    pure_digest = file_sha256(pure_path)
    if frozen_digest.lower() != config.frozen_baseline.dataset_sha256.lower():
        raise RuntimeError("Frozen Phase 6B dataset hash changed; refusing Phase 6C-2 generation.")
    if pure_digest.lower() != config.frozen_baseline.pure_dnn_trajectory_sha256.lower():
        raise RuntimeError("Frozen pure-DNN trajectory hash changed; refusing DAgger generation.")
    frozen = pd.read_csv(frozen_path)
    frozen_audit = {
        "dataset_sha256": frozen_digest,
        "pure_dnn_trajectory_sha256": pure_digest,
        "sample_count": int(len(frozen)),
        "test_sample_count": int((frozen["split"] == "test").sum()),
        "test_trajectory_count": int(frozen.loc[frozen["split"] == "test", "trajectory_id"].nunique()),
        "trajectory_split_leak_count": int((frozen.groupby("trajectory_id")["split"].nunique() != 1).sum()),
    }

    phase3, parameters, ocv_function = _load_context(config, root)
    model = ReducedBatteryModel(phase3, ocv_function, parameters)
    targeted = generate_targeted_boundary_design(config)
    replay = replay_pure_dnn_states(pd.read_csv(pure_path), model, phase3)
    dagger = generate_dagger_design(replay, frozen[frozen["split"] == "train"], config)
    design = pd.concat([targeted, dagger], ignore_index=True)
    split_map = assign_new_splits(design, config)
    design["assigned_split"] = design["initial_state_id"].map(split_map)
    design.to_csv(data_dir / "initial_state_design.csv", index=False)
    replay.to_csv(data_dir / "pure_dnn_replayed_states.csv", index=False)

    attempts_path = data_dir / "teacher_attempts.csv"
    dataset_path = data_dir / "targeted_teacher_dataset.csv"
    attempts = pd.read_csv(attempts_path).to_dict("records") if attempts_path.exists() else []
    rows = pd.read_csv(dataset_path).to_dict("records") if dataset_path.exists() else []
    completed = {str(record["initial_state_id"]) for record in attempts}
    optimizer = PaperTrajectoryMPC(model, phase3, config.sampling.trajectory_steps)
    for index, (_, initial_row) in enumerate(design.iterrows(), start=1):
        initial_id = str(initial_row["initial_state_id"])
        if initial_id in completed:
            continue
        try:
            result = optimizer.solve(row_to_state(initial_row))
            accepted = bool(result.optimizer_success and result.prediction_feasible)
            attempt = {
                **initial_row.to_dict(),
                "teacher_optimizer_success": result.optimizer_success,
                "teacher_prediction_feasible": result.prediction_feasible,
                "teacher_accepted": accepted,
                "teacher_status": result.status,
                "teacher_objective": result.objective_value,
                "teacher_solve_time_s": result.solve_time_s,
                "teacher_minimum_constraint_margin": result.minimum_constraint_margin,
            }
            if accepted:
                rows.extend(
                    _unfold_accepted_trajectory(
                        initial_row, split_map[initial_id], result, model, phase3
                    )
                )
        except Exception as error:  # preserve a failed audit row and continue the design
            attempt = {
                **initial_row.to_dict(),
                "teacher_optimizer_success": False,
                "teacher_prediction_feasible": False,
                "teacher_accepted": False,
                "teacher_status": f"exception: {type(error).__name__}: {error}",
                "teacher_objective": np.nan,
                "teacher_solve_time_s": 0.0,
                "teacher_minimum_constraint_margin": np.nan,
            }
        attempts.append(attempt)
        completed.add(initial_id)
        if len(completed) % config.sampling.checkpoint_interval == 0 or index == len(design):
            pd.DataFrame.from_records(attempts).to_csv(attempts_path, index=False)
            pd.DataFrame.from_records(rows).to_csv(dataset_path, index=False)
            accepted_count = sum(bool(record["teacher_accepted"]) for record in attempts)
            print(
                f"completed {len(completed)}/{len(design)} teacher trajectories; accepted={accepted_count}",
                flush=True,
            )

    attempt_frame = pd.DataFrame.from_records(attempts)
    dataset = pd.DataFrame.from_records(rows)
    metrics = _metrics(attempt_frame, dataset, config)
    payload = {
        "status": "completed" if metrics["success"] else "teacher_data_gate_failed",
        "configuration": asdict(config),
        "frozen_baseline_audit": frozen_audit,
        "teacher_data": metrics,
    }
    (metrics_dir / "phase6c2_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(root / "outputs" / "phase6c2_report.md", payload)
    return payload

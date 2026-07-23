"""Run nominal and parameter-oracle MPC on the frozen Phase 5A scenario set."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .closed_loop import _cap_current_at_target, initial_reduced_state
from .identification import build_ocv_function
from .mpc import ConstrainedMPC, ReducedBatteryModel
from .phase3_config import load_phase_three_config
from .phase5a_config import load_phase_five_a_config
from .phase5b0_config import PhaseFiveBZeroConfig, load_phase_five_b_zero_config
from .robustness import _estimated_state, generate_reduced_stress_scenarios, perturb_identified_parameters


CONTROLLERS = ("nominal_mpc", "oracle_mpc")


def classify_scenario(nominal_feasible: bool, oracle_feasible: bool, ann_feasible: bool) -> str:
    if not nominal_feasible and oracle_feasible:
        return "nominal_teacher_failed_oracle_teacher_feasible"
    if nominal_feasible and ann_feasible:
        return "teacher_and_ann_feasible"
    if nominal_feasible and not ann_feasible:
        return "teacher_feasible_ann_failed"
    if not oracle_feasible and not ann_feasible:
        return "teacher_and_ann_infeasible"
    return "ann_feasible_teachers_failed_unresolved"


def infeasibility_reasons(summary: dict[str, Any]) -> str:
    reasons: list[str] = []
    if not summary["completion_success"]:
        reasons.append("target_not_reached")
    if not summary["physical_safe"]:
        for name in ("voltage", "temperature", "current", "current_change"):
            if summary[f"{name}_limit_exceeded"]:
                reasons.append(f"{name}_violation")
    if summary["optimizer_success_fraction"] < summary["required_optimizer_success_fraction"]:
        reasons.append("optimizer_success_fraction")
    if summary["fallback_count"] > summary["maximum_allowed_fallback_count"]:
        reasons.append("mpc_fallback")
    return ";".join(reasons) if reasons else "none"


def _context(root: Path, config: PhaseFiveBZeroConfig):
    phase3 = load_phase_three_config(root / config.source_phase3_config)
    phase5a = load_phase_five_a_config(root / config.source_phase5a_config)
    parameters = json.loads((root / phase3.artifacts.identified_parameters).read_text(encoding="utf-8"))
    ocv_frame = pd.read_csv(root / phase3.artifacts.ocv_curve)
    return phase3, phase5a, parameters, build_ocv_function(ocv_frame)


def simulate_mpc_scenario(
    root: Path,
    config: PhaseFiveBZeroConfig,
    scenario: pd.Series,
    scenario_index: int,
    controller_kind: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    phase3, phase5a, nominal_parameters, ocv = _context(root, config)
    stress = phase5a.reduced_stress_test
    controller_battery = replace(
        phase3.battery,
        initial_soc=float(scenario["initial_soc"]),
        initial_temperature_c=float(scenario["ambient_temperature_c"]),
        ambient_temperature_c=float(scenario["ambient_temperature_c"]),
    )
    nominal_controller_config = replace(
        phase3,
        battery=controller_battery,
        control=replace(phase3.control, maximum_simulation_time_s=config.scope.maximum_charge_time_s),
    )
    true_battery = replace(
        controller_battery,
        nominal_capacity_ah=phase3.battery.nominal_capacity_ah * float(scenario["capacity_multiplier"]),
    )
    true_config = replace(nominal_controller_config, battery=true_battery)
    true_parameters = perturb_identified_parameters(nominal_parameters, scenario)
    if controller_kind == "nominal_mpc":
        controller_config = nominal_controller_config
        controller_parameters = nominal_parameters
    elif controller_kind == "oracle_mpc":
        controller_config = true_config
        controller_parameters = true_parameters
    else:
        raise ValueError(f"Unknown Phase 5B-0 controller: {controller_kind}")
    controller_model = ReducedBatteryModel(controller_config, ocv, controller_parameters)
    true_model = ReducedBatteryModel(true_config, ocv, true_parameters)
    controller = ConstrainedMPC(controller_model, controller_config)
    true_state = initial_reduced_state(true_config)
    random = np.random.default_rng(config.random_seed + 1009 * scenario_index)
    noise = {"soc": 0.0, "temperature_c": 0.0, "polarization_fast_v": 0.0, "polarization_slow_v": 0.0}
    sigmas = {
        "soc": stress.soc_noise_standard_deviation,
        "temperature_c": stress.temperature_noise_standard_deviation_c,
        "polarization_fast_v": stress.polarization_noise_standard_deviation_v,
        "polarization_slow_v": stress.polarization_noise_standard_deviation_v,
    }
    rho = 0.95
    innovation = np.sqrt(1.0 - rho**2)
    source = f"phase5b0_{controller_kind}"
    records: list[dict[str, Any]] = [{
        "scenario_id": str(scenario["scenario_id"]), "controller": controller_kind,
        "time_s": 0.0, "true_soc": true_state.soc, "estimated_soc": true_state.soc,
        "charge_current_a": 0.0, "terminal_voltage_v": true_model.ocv(true_state.soc),
        "true_average_temperature_c": true_model.average_temperature(true_state),
        "control_decision_updated": False, "optimizer_success": True,
        "prediction_feasible": True, "used_fallback": False, "solve_time_s": 0.0,
        "active_voltage_constraint": False, "active_temperature_constraint": False,
        "active_current_upper_constraint": False, "active_current_change_constraint": False,
        "source": source,
    }]
    result = None
    steps_until_reoptimization = 0
    controller_complete = False
    maximum_steps = int(np.ceil(config.scope.maximum_charge_time_s / phase3.control.control_interval_s))
    for step in range(1, maximum_steps + 1):
        for key, sigma in sigmas.items():
            noise[key] = rho * noise[key] + innovation * sigma * float(scenario["noise_scale"]) * random.normal()
        estimate = _estimated_state(true_state, controller_model, scenario, noise)
        if estimate.soc >= phase3.battery.target_soc - phase3.validation.target_soc_tolerance:
            controller_complete = True
            break
        decision_updated = result is None or steps_until_reoptimization <= 0
        if decision_updated:
            previous = estimate.previous_current_a
            result = controller.solve(estimate)
            hits_slew = abs(result.current_a - previous) >= 0.95 * phase3.constraints.maximum_current_change_a_per_step
            steps_until_reoptimization = 1 if hits_slew else phase3.control.control_block_steps
        assert result is not None
        requested = float(result.current_a)
        applied, _ = _cap_current_at_target(requested, estimate.soc, controller_config)
        true_state, output = true_model.step(true_state, applied)
        margin = config.activity_margins
        records.append({
            "scenario_id": str(scenario["scenario_id"]), "controller": controller_kind,
            "time_s": step * phase3.control.control_interval_s, "true_soc": true_state.soc,
            "estimated_soc": estimate.soc, "charge_current_a": applied,
            "terminal_voltage_v": output.terminal_voltage_v,
            "true_average_temperature_c": output.average_temperature_c,
            "control_decision_updated": decision_updated,
            "optimizer_success": result.optimizer_success, "prediction_feasible": result.prediction_feasible,
            "used_fallback": result.used_fallback, "solve_time_s": result.solve_time_s if decision_updated else 0.0,
            "active_voltage_constraint": bool(decision_updated and result.predicted_maximum_voltage_v >= controller_config.constraints.mpc_maximum_voltage_v - margin.voltage_v),
            "active_temperature_constraint": bool(decision_updated and result.predicted_maximum_temperature_c >= controller_config.constraints.mpc_maximum_temperature_c - margin.temperature_c),
            "active_current_upper_constraint": bool(decision_updated and requested >= controller_config.constraints.maximum_current_a - margin.current_a),
            "active_current_change_constraint": bool(decision_updated and abs(requested - estimate.previous_current_a) >= controller_config.constraints.maximum_current_change_a_per_step - margin.current_change_a),
            "source": source,
        })
        steps_until_reoptimization -= 1
        if true_state.soc >= 0.815:
            break
    frame = pd.DataFrame.from_records(records)
    changes = frame["charge_current_a"].diff().abs().fillna(0.0)
    optimization = frame[frame["control_decision_updated"]]
    tolerance = phase3.validation.physical_constraint_tolerance
    terminal_error = float(frame["true_soc"].iloc[-1] - phase3.battery.target_soc)
    completion = bool(controller_complete and abs(terminal_error) <= config.feasibility.terminal_true_soc_tolerance)
    violations = {
        "voltage_limit_exceeded": bool(frame["terminal_voltage_v"].max() > phase3.constraints.physical_maximum_voltage_v + tolerance),
        "temperature_limit_exceeded": bool(frame["true_average_temperature_c"].max() > phase3.constraints.physical_maximum_temperature_c + tolerance),
        "current_limit_exceeded": bool(frame["charge_current_a"].max() > phase3.constraints.maximum_current_a + tolerance or frame["charge_current_a"].min() < -tolerance),
        "current_change_limit_exceeded": bool(changes.max() > phase3.constraints.maximum_current_change_a_per_step + tolerance),
    }
    physical_safe = not any(violations.values())
    optimizer_fraction = float(optimization["optimizer_success"].mean()) if len(optimization) else 0.0
    fallback_count = int(optimization["used_fallback"].sum())
    teacher_feasible = bool(
        completion
        and (physical_safe or not config.feasibility.require_physical_safety)
        and optimizer_fraction >= config.feasibility.minimum_optimizer_success_fraction
        and fallback_count <= config.feasibility.maximum_fallback_count
    )
    summary = {**scenario.to_dict(), "controller": controller_kind,
        "controller_declared_complete": controller_complete, "completion_success": completion,
        "physical_safe": physical_safe, "teacher_feasible": teacher_feasible,
        "charge_time_min": float(frame["time_s"].iloc[-1] / 60.0),
        "final_true_soc": float(frame["true_soc"].iloc[-1]), "terminal_true_soc_error": terminal_error,
        "maximum_voltage_v": float(frame["terminal_voltage_v"].max()),
        "maximum_temperature_c": float(frame["true_average_temperature_c"].max()),
        "maximum_current_a": float(frame["charge_current_a"].max()),
        "maximum_current_change_a": float(changes.max()),
        "optimization_count": int(len(optimization)), "optimizer_success_fraction": optimizer_fraction,
        "fallback_count": fallback_count,
        "mean_solve_time_ms": float(optimization["solve_time_s"].mean() * 1000.0) if len(optimization) else 0.0,
        "maximum_solve_time_ms": float(optimization["solve_time_s"].max() * 1000.0) if len(optimization) else 0.0,
        "required_optimizer_success_fraction": config.feasibility.minimum_optimizer_success_fraction,
        "maximum_allowed_fallback_count": config.feasibility.maximum_fallback_count,
        **violations,
    }
    for name in ("voltage", "temperature", "current_upper", "current_change"):
        column = f"active_{name}_constraint"
        summary[f"{column}_count"] = int(optimization[column].sum())
        summary[f"{column}_fraction"] = float(optimization[column].mean()) if len(optimization) else 0.0
    summary["infeasibility_reasons"] = infeasibility_reasons(summary)
    return summary, frame


def _worker(root: str, config_path: str, scenario: dict[str, Any], index: int, controller: str):
    config = load_phase_five_b_zero_config(config_path)
    return simulate_mpc_scenario(Path(root), config, pd.Series(scenario), index, controller)


def _scenario_table(runs: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    indexed = runs.set_index(["scenario_id", "controller"])
    records: list[dict[str, Any]] = []
    ann_indexed = ann.set_index("scenario_id")
    for scenario_id in sorted(runs["scenario_id"].unique()):
        nominal = indexed.loc[(scenario_id, "nominal_mpc")]
        oracle = indexed.loc[(scenario_id, "oracle_mpc")]
        ann_row = ann_indexed.loc[scenario_id]
        ann_feasible = bool(ann_row["completion_success"] and ann_row["physical_safe"])
        records.append({
            "scenario_id": scenario_id, "scenario_kind": nominal["scenario_kind"],
            "ambient_temperature_c": nominal["ambient_temperature_c"],
            "resistance_multiplier": nominal["resistance_multiplier"],
            "heat_capacity_multiplier": nominal["heat_capacity_multiplier"],
            "thermal_resistance_multiplier": nominal["thermal_resistance_multiplier"],
            "ann_feasible": ann_feasible,
            "nominal_mpc_feasible": bool(nominal["teacher_feasible"]),
            "oracle_mpc_feasible": bool(oracle["teacher_feasible"]),
            "nominal_charge_time_min": nominal["charge_time_min"],
            "oracle_charge_time_min": oracle["charge_time_min"],
            "nominal_final_soc": nominal["final_true_soc"], "oracle_final_soc": oracle["final_true_soc"],
            "nominal_infeasibility_reasons": nominal["infeasibility_reasons"],
            "oracle_infeasibility_reasons": oracle["infeasibility_reasons"],
            "scenario_class": classify_scenario(bool(nominal["teacher_feasible"]), bool(oracle["teacher_feasible"]), ann_feasible),
        })
    return pd.DataFrame.from_records(records)


def run_phase_five_b_zero(config: PhaseFiveBZeroConfig, project_root: str | Path, config_path: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_path = str(Path(config_path).resolve())
    data_dir = root / "data" / "phase5b_mpc_feasibility"
    trajectory_dir = data_dir / "trajectories"
    metrics_dir = root / "outputs" / "metrics"
    data_dir.mkdir(parents=True, exist_ok=True); trajectory_dir.mkdir(parents=True, exist_ok=True); metrics_dir.mkdir(parents=True, exist_ok=True)
    _, phase5a, _, _ = _context(root, config)
    scenarios = generate_reduced_stress_scenarios(phase5a)
    if len(scenarios) != config.scope.expected_reduced_scenario_count:
        raise RuntimeError("Frozen Phase 5A scenario count changed.")
    ann = pd.read_csv(root / config.source_phase5a_summary)
    if set(ann["scenario_id"]) != set(scenarios["scenario_id"]):
        raise RuntimeError("Phase 5A ANN summary does not match the frozen scenario set.")
    runs_path = data_dir / "controller_run_summary.csv"
    records = pd.read_csv(runs_path).to_dict("records") if runs_path.exists() else []
    completed = {(str(row["scenario_id"]), str(row["controller"])) for row in records}
    tasks = [(index, scenario.to_dict(), controller) for index, scenario in scenarios.iterrows() for controller in CONTROLLERS if (str(scenario["scenario_id"]), controller) not in completed]
    if tasks:
        with ProcessPoolExecutor(max_workers=config.execution.worker_count) as pool:
            futures = {pool.submit(_worker, str(root), config_path, scenario, index, controller): (scenario["scenario_id"], controller) for index, scenario, controller in tasks}
            for future in as_completed(futures):
                scenario_id, controller = futures[future]
                summary, trajectory = future.result()
                records.append(summary)
                if config.execution.save_all_trajectories:
                    trajectory.to_csv(trajectory_dir / f"{scenario_id}__{controller}.csv", index=False)
                pd.DataFrame.from_records(records).sort_values(["scenario_id", "controller"]).to_csv(runs_path, index=False)
                print(f"completed {scenario_id} / {controller}: feasible={summary['teacher_feasible']}", flush=True)
    runs = pd.DataFrame.from_records(records).sort_values(["scenario_id", "controller"])
    table = _scenario_table(runs, ann)
    table.to_csv(data_dir / "scenario_feasibility_table.csv", index=False)
    mask = table[["scenario_id", "nominal_mpc_feasible", "oracle_mpc_feasible", "scenario_class"]].copy()
    mask["teacher_feasible_for_ann_evaluation"] = mask["nominal_mpc_feasible"]
    mask.to_csv(data_dir / "teacher_feasible_scenario_mask.csv", index=False)
    activity_columns = [column for column in runs.columns if column.startswith("active_") and column.endswith("_fraction")]
    activity = runs.groupby("controller")[activity_columns].mean().reset_index()
    activity.to_csv(data_dir / "teacher_constraint_activity.csv", index=False)
    reason_counts = runs.groupby(["controller", "infeasibility_reasons"]).size().rename("scenario_count").reset_index()
    reason_counts.to_csv(data_dir / "infeasibility_reason_counts.csv", index=False)
    controller_summary = runs.groupby("controller").agg(
        scenario_count=("scenario_id", "count"), feasible_count=("teacher_feasible", "sum"),
        completion_fraction=("completion_success", "mean"), physical_safety_fraction=("physical_safe", "mean"),
        mean_charge_time_min=("charge_time_min", "mean"), maximum_charge_time_min=("charge_time_min", "max"),
        mean_solve_time_ms=("mean_solve_time_ms", "mean"), maximum_solve_time_ms=("maximum_solve_time_ms", "max"),
        fallback_scenario_count=("fallback_count", lambda values: int((values > 0).sum())),
    ).reset_index()
    controller_summary["feasible_fraction"] = controller_summary["feasible_count"] / controller_summary["scenario_count"]
    controller_summary.to_csv(data_dir / "teacher_completion_and_safety_summary.csv", index=False)
    class_counts = table["scenario_class"].value_counts().rename_axis("scenario_class").reset_index(name="scenario_count")
    class_counts.to_csv(data_dir / "scenario_class_counts.csv", index=False)
    payload = {
        "status": "completed", "configuration": asdict(config),
        "controller_summary": controller_summary.to_dict("records"),
        "scenario_class_counts": class_counts.to_dict("records"),
        "teacher_feasible_mask_count": int(mask["teacher_feasible_for_ann_evaluation"].sum()),
        "unresolved_class_count": int((table["scenario_class"] == "ann_feasible_teachers_failed_unresolved").sum()),
        "ready_for_phase5b1_sampling_design": True,
        "dfn_anchors_run": False,
    }
    (metrics_dir / "phase5b0_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload

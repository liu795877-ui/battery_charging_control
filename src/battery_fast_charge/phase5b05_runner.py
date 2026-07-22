"""Representative Phase 5B-0.5 closed-loop recovery recheck with checkpoints."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ann_model import TinyANN
from .closed_loop import initial_reduced_state
from .identification import build_ocv_function
from .mpc import ReducedBatteryModel
from .phase3_config import load_phase_three_config
from .phase5a_config import load_phase_five_a_config
from .phase5b05_config import PhaseFiveBZeroFiveConfig
from .phase5b05_mpc import (
    FAILURE_HARD_SAFETY_SLEW_CONFLICT,
    FAILURE_NUMERICAL_RECOVERED,
    FAILURE_PREDICTION_INFEASIBLE,
    RecoverableConstrainedMPC,
)
from .robustness import _estimated_state, generate_reduced_stress_scenarios, perturb_identified_parameters


CONTROLLERS = ("nominal_mpc_recovery", "oracle_mpc_recovery")


def scenario_row_by_id(scenarios: pd.DataFrame, scenario_id: str) -> pd.Series:
    """Return a scenario row without losing its identifier during lookup."""
    matches = scenarios.loc[scenarios["scenario_id"].astype(str) == str(scenario_id)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one stress scenario for {scenario_id!r}, found {len(matches)}"
        )
    return matches.iloc[0].copy()


def select_representative_scenarios(
    table: pd.DataFrame, config: PhaseFiveBZeroFiveConfig
) -> pd.DataFrame:
    selection = config.representative_selection
    selected: dict[str, set[str]] = {}

    def add(frame: pd.DataFrame, count: int, label: str) -> None:
        for scenario_id in frame.sort_values("scenario_id")["scenario_id"].head(count):
            selected.setdefault(str(scenario_id), set()).add(label)

    add(table[table["nominal_mpc_feasible"].astype(bool)], selection.teacher_feasible_count, "teacher_feasible")
    add(
        table[table["scenario_class"] == "ann_feasible_teachers_failed_unresolved"],
        selection.unresolved_count,
        "unresolved",
    )
    add(
        table[table["scenario_class"] == "teacher_and_ann_infeasible"],
        selection.teacher_and_ann_infeasible_count,
        "teacher_and_ann_infeasible",
    )
    if selection.include_nominal:
        selected.setdefault("nominal", set()).add("nominal")
    if selection.include_hot_extreme:
        selected.setdefault("corner_hot_resistive_optimistic", set()).add("hot_extreme")
    if selection.include_cold_extreme:
        selected.setdefault("corner_cold_resistive", set()).add("cold_extreme")
    rows = table[table["scenario_id"].isin(selected)].copy()
    rows["selection_labels"] = rows["scenario_id"].map(
        lambda value: ";".join(sorted(selected[str(value)]))
    )
    return rows.sort_values("scenario_id").reset_index(drop=True)


def _context(root: Path, config: PhaseFiveBZeroFiveConfig):
    phase3 = load_phase_three_config(root / config.source_phase3_config)
    phase5a = load_phase_five_a_config(root / config.source_phase5a_config)
    parameters = json.loads((root / phase3.artifacts.identified_parameters).read_text(encoding="utf-8"))
    ocv = build_ocv_function(pd.read_csv(root / phase3.artifacts.ocv_curve))
    ann = TinyANN.load(root / config.ann_model)
    return phase3, phase5a, parameters, ocv, ann


def simulate_recovery_scenario(
    root: Path,
    config: PhaseFiveBZeroFiveConfig,
    scenario: pd.Series,
    scenario_index: int,
    controller_kind: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    phase3, phase5a, nominal_parameters, ocv, ann = _context(root, config)
    stress = phase5a.reduced_stress_test
    battery = replace(
        phase3.battery,
        initial_soc=float(scenario["initial_soc"]),
        initial_temperature_c=float(scenario["ambient_temperature_c"]),
        ambient_temperature_c=float(scenario["ambient_temperature_c"]),
    )
    nominal_config = replace(
        phase3,
        battery=battery,
        control=replace(phase3.control, maximum_simulation_time_s=config.execution.maximum_simulation_time_s),
    )
    true_battery = replace(
        battery,
        nominal_capacity_ah=phase3.battery.nominal_capacity_ah * float(scenario["capacity_multiplier"]),
    )
    true_config = replace(nominal_config, battery=true_battery)
    true_parameters = perturb_identified_parameters(nominal_parameters, scenario)
    if controller_kind == "nominal_mpc_recovery":
        controller_config, controller_parameters = nominal_config, nominal_parameters
    elif controller_kind == "oracle_mpc_recovery":
        controller_config, controller_parameters = true_config, true_parameters
    else:
        raise ValueError(f"Unknown Phase 5B-0.5 controller: {controller_kind}")
    controller_model = ReducedBatteryModel(controller_config, ocv, controller_parameters)
    true_model = ReducedBatteryModel(true_config, ocv, true_parameters)
    controller = RecoverableConstrainedMPC(
        controller_model, controller_config, ann, config.recovery.one_step_scan_points
    )
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
    records: list[dict[str, Any]] = [{
        "scenario_id": str(scenario["scenario_id"]), "controller": controller_kind,
        "time_s": 0.0, "true_soc": true_state.soc, "estimated_soc": true_state.soc,
        "charge_current_a": 0.0, "terminal_voltage_v": true_model.ocv(true_state.soc),
        "true_average_temperature_c": true_model.average_temperature(true_state),
        "control_decision_updated": False, "decision_source": "initial",
        "failure_type": "none", "optimizer_success": True,
        "selected_sequence_feasible": True, "used_emergency_fallback": False,
        "hard_safety_slew_conflict": False, "solve_time_s": 0.0,
    }]
    result = None
    steps_until_reoptimization = 0
    elapsed_steps = 0
    controller_complete = False
    maximum_steps = int(np.ceil(config.execution.maximum_simulation_time_s / phase3.control.control_interval_s))
    for step in range(1, maximum_steps + 1):
        for key, sigma in sigmas.items():
            noise[key] = rho * noise[key] + innovation * sigma * float(scenario["noise_scale"]) * random.normal()
        estimate = _estimated_state(true_state, controller_model, scenario, noise)
        if estimate.soc >= phase3.battery.target_soc - phase3.validation.target_soc_tolerance:
            controller_complete = True
            break
        decision_updated = result is None or steps_until_reoptimization <= 0
        if decision_updated:
            result = controller.solve_with_recovery(estimate, elapsed_steps)
            hits_slew = abs(result.current_a - estimate.previous_current_a) >= 0.95 * phase3.constraints.maximum_current_change_a_per_step
            steps_until_reoptimization = 1 if hits_slew else phase3.control.control_block_steps
            elapsed_steps = 0
        assert result is not None
        previous_current = true_state.previous_current_a
        applied = float(result.current_a)
        true_state, output = true_model.step(true_state, applied)
        current_change = abs(applied - previous_current)
        records.append({
            "scenario_id": str(scenario["scenario_id"]), "controller": controller_kind,
            "time_s": step * phase3.control.control_interval_s, "true_soc": true_state.soc,
            "estimated_soc": estimate.soc, "charge_current_a": applied,
            "current_change_a": current_change,
            "terminal_voltage_v": output.terminal_voltage_v,
            "true_average_temperature_c": output.average_temperature_c,
            "control_decision_updated": decision_updated,
            "decision_source": result.source, "failure_type": result.failure_type,
            "optimizer_success": result.optimizer_success,
            "selected_sequence_feasible": result.selected_sequence_feasible,
            "used_emergency_fallback": result.used_emergency_fallback,
            "hard_safety_slew_conflict": result.hard_safety_slew_conflict,
            "solve_time_s": result.solve_time_s if decision_updated else 0.0,
        })
        steps_until_reoptimization -= 1
        elapsed_steps += 1
        if true_state.soc >= 0.815:
            break
    frame = pd.DataFrame.from_records(records)
    control = frame.iloc[1:]
    decisions = frame[frame["control_decision_updated"]]
    tolerance = phase3.validation.physical_constraint_tolerance
    terminal_error = float(frame["true_soc"].iloc[-1] - phase3.battery.target_soc)
    completion = bool(controller_complete and abs(terminal_error) <= stress.terminal_true_soc_tolerance)
    voltage_bad = bool(frame["terminal_voltage_v"].max() > phase3.constraints.physical_maximum_voltage_v + tolerance)
    temperature_bad = bool(frame["true_average_temperature_c"].max() > phase3.constraints.physical_maximum_temperature_c + tolerance)
    current_bad = bool(frame["charge_current_a"].max() > phase3.constraints.maximum_current_a + tolerance or frame["charge_current_a"].min() < -tolerance)
    slew_bad = bool(control["current_change_a"].max() > phase3.constraints.maximum_current_change_a_per_step + tolerance)
    physical_safe = not (voltage_bad or temperature_bad or current_bad or slew_bad)
    recovery_rows = control[control["decision_source"] != "slsqp"]
    nonconflict_recovery = recovery_rows[~recovery_rows["hard_safety_slew_conflict"].astype(bool)]
    fallback_slew_violation_count = int(
        (nonconflict_recovery["current_change_a"] > phase3.constraints.maximum_current_change_a_per_step + tolerance).sum()
    )
    conflict_count = int(decisions["hard_safety_slew_conflict"].sum())
    operational_feasible = bool(completion and physical_safe and conflict_count == 0 and fallback_slew_violation_count == 0)
    summary = {
        **scenario.to_dict(), "controller": controller_kind,
        "controller_declared_complete": controller_complete, "completion_success": completion,
        "physical_safe": physical_safe, "operational_feasible": operational_feasible,
        "charge_time_min": float(frame["time_s"].iloc[-1] / 60.0),
        "final_true_soc": float(frame["true_soc"].iloc[-1]), "terminal_true_soc_error": terminal_error,
        "maximum_voltage_v": float(frame["terminal_voltage_v"].max()),
        "maximum_temperature_c": float(frame["true_average_temperature_c"].max()),
        "maximum_current_a": float(frame["charge_current_a"].max()),
        "maximum_current_change_a": float(control["current_change_a"].max()),
        "voltage_limit_exceeded": voltage_bad, "temperature_limit_exceeded": temperature_bad,
        "current_limit_exceeded": current_bad, "current_change_limit_exceeded": slew_bad,
        "optimization_count": int(len(decisions)),
        "optimizer_success_fraction": float(decisions["optimizer_success"].mean()) if len(decisions) else 0.0,
        "numerical_failure_recovered_count": int((decisions["failure_type"] == FAILURE_NUMERICAL_RECOVERED).sum()),
        "prediction_domain_infeasible_count": int((decisions["failure_type"] == FAILURE_PREDICTION_INFEASIBLE).sum()),
        "hard_safety_slew_conflict_count": conflict_count,
        "fallback_slew_violation_count": fallback_slew_violation_count,
        "shifted_previous_count": int((decisions["decision_source"] == "shifted_previous_feasible").sum()),
        "projected_ann_sequence_count": int((decisions["decision_source"] == "projected_ann_sequence").sum()),
        "conservative_slew_down_count": int((decisions["decision_source"] == "conservative_slew_down").sum()),
        "slope_safe_emergency_count": int((decisions["decision_source"] == "slope_safe_emergency").sum()),
        "hard_safety_emergency_count": int((decisions["decision_source"] == "hard_safety_emergency").sum()),
        "mean_solve_time_ms": float(decisions["solve_time_s"].mean() * 1000.0) if len(decisions) else 0.0,
        "maximum_solve_time_ms": float(decisions["solve_time_s"].max() * 1000.0) if len(decisions) else 0.0,
    }
    return summary, frame


def run_phase_five_b_zero_five(
    config: PhaseFiveBZeroFiveConfig, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    data_dir = root / "data" / "phase5b05_mpc_recovery"
    trajectory_dir = data_dir / "trajectories"
    metrics_dir = root / "outputs" / "metrics"
    data_dir.mkdir(parents=True, exist_ok=True); trajectory_dir.mkdir(parents=True, exist_ok=True); metrics_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(root / config.source_phase5b0_table)
    representatives = select_representative_scenarios(table, config)
    representatives.to_csv(data_dir / "representative_scenarios.csv", index=False)
    phase5a = load_phase_five_a_config(root / config.source_phase5a_config)
    scenarios = generate_reduced_stress_scenarios(phase5a)
    runs_path = data_dir / "recovery_run_summary.csv"
    records = pd.read_csv(runs_path).to_dict("records") if runs_path.exists() else []
    completed = {(str(row["scenario_id"]), str(row["controller"])) for row in records}
    for _, selected in representatives.iterrows():
        scenario_id = str(selected["scenario_id"])
        scenario = scenario_row_by_id(scenarios, scenario_id)
        scenario_index = int(scenarios.index[scenarios["scenario_id"] == scenario_id][0])
        for controller in CONTROLLERS:
            if (scenario_id, controller) in completed:
                continue
            summary, trajectory = simulate_recovery_scenario(
                root, config, scenario, scenario_index, controller
            )
            summary["selection_labels"] = selected["selection_labels"]
            summary["baseline_feasible"] = bool(
                selected["nominal_mpc_feasible"]
                if controller == "nominal_mpc_recovery"
                else selected["oracle_mpc_feasible"]
            )
            records.append(summary)
            completed.add((scenario_id, controller))
            if config.execution.save_all_trajectories:
                trajectory.to_csv(trajectory_dir / f"{scenario_id}__{controller}.csv", index=False)
            pd.DataFrame.from_records(records).sort_values(["scenario_id", "controller"]).to_csv(runs_path, index=False)
            print(
                f"completed {scenario_id} / {controller}: operational_feasible={summary['operational_feasible']}",
                flush=True,
            )
    runs = pd.DataFrame.from_records(records).sort_values(["scenario_id", "controller"])
    summary = runs.groupby("controller").agg(
        scenario_count=("scenario_id", "count"),
        baseline_feasible_count=("baseline_feasible", "sum"),
        recovery_feasible_count=("operational_feasible", "sum"),
        fallback_slew_violation_count=("fallback_slew_violation_count", "sum"),
        hard_safety_slew_conflict_count=("hard_safety_slew_conflict_count", "sum"),
        numerical_failure_recovered_count=("numerical_failure_recovered_count", "sum"),
        prediction_domain_infeasible_count=("prediction_domain_infeasible_count", "sum"),
        mean_solve_time_ms=("mean_solve_time_ms", "mean"),
        maximum_solve_time_ms=("maximum_solve_time_ms", "max"),
    ).reset_index()
    summary["feasible_gain"] = summary["recovery_feasible_count"] - summary["baseline_feasible_count"]
    summary.to_csv(data_dir / "recovery_controller_summary.csv", index=False)
    nominal = summary[summary["controller"] == "nominal_mpc_recovery"].iloc[0]
    oracle = summary[summary["controller"] == "oracle_mpc_recovery"].iloc[0]
    checks = {
        "fallback_slew_violation_zero": int(summary["fallback_slew_violation_count"].sum())
        <= config.acceptance.maximum_fallback_slew_violation_count,
        "matched_nominal_feasible_gain": int(nominal["feasible_gain"])
        >= config.acceptance.minimum_matched_nominal_feasible_gain,
        "oracle_not_weaker_than_nominal": int(oracle["recovery_feasible_count"])
        >= int(nominal["recovery_feasible_count"]),
        "failure_types_auditable": bool(
            runs["numerical_failure_recovered_count"].notna().all()
            and runs["prediction_domain_infeasible_count"].notna().all()
            and runs["hard_safety_slew_conflict_count"].notna().all()
        ),
    }
    payload = {
        "status": "completed", "configuration": asdict(config),
        "representative_scenario_count": int(representatives["scenario_id"].nunique()),
        "controller_summary": summary.to_dict("records"), "checks": checks,
        "representative_gate_passed": bool(all(checks.values())),
        "full_69_recheck_required_before_phase5b1": config.acceptance.require_full_69_before_phase5b1,
        "proceed_to_phase5b1": False,
    }
    (metrics_dir / "phase5b05_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload

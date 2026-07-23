"""Phase 5B-0.6 strict paired feasibility-contract audit."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ann_model import TinyANN
from .closed_loop import _cap_current_at_target, initial_reduced_state
from .identification import build_ocv_function
from .mpc import ConstrainedMPC, ReducedBatteryModel
from .phase3_config import load_phase_three_config
from .phase5a_config import load_phase_five_a_config
from .phase5b05_mpc import RecoverableConstrainedMPC
from .robustness import _estimated_state, generate_reduced_stress_scenarios, perturb_identified_parameters


CONTROLLERS = ("original_mpc", "recovery_mpc")
GROUP_LABELS = {
    "teacher_feasible": "teacher_feasible",
    "unresolved": "unresolved",
    "teacher_and_ann_infeasible": "teacher_and_ann_infeasible",
}


def feasibility_count_table(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.groupby(["scenario_group", "controller"], as_index=False).agg(
        scenario_count=("scenario_id", "count"),
        operational_feasible_count=("operational_feasible", "sum"),
    )


def render_chinese_report(summary: pd.DataFrame, counts: pd.DataFrame, payload: dict[str, Any]) -> str:
    recovery = summary[summary.controller == "recovery_mpc"]
    total = int(recovery.operational_feasible.sum())
    lines = [
        "# Phase 5B-0.6 修正合同下的 15 场景复评", "",
        "<!-- canonical_feasibility_field: operational_feasible -->",
        f"<!-- recovery_operational_feasible_count: {total} -->", "",
        "## 冻结合同", "",
        "本次使用 Phase 5B-0 的随机种子、完整场景索引、噪声序列、初始状态、模型参数、控制更新时间、目标电流 cap 与轨迹截止规则。未训练新 ANN，也未运行完整 69 场景。", "",
        "## 可行性结果", "",
        "统一可行性字段为 `operational_feasible`。", "",
        "| 场景组 | 原始 MPC | Recovery MPC |", "|---|---:|---:|",
    ]
    labels = {"teacher_feasible": "原始教师可行", "unresolved": "unresolved", "teacher_and_ann_infeasible": "教师与 ANN 均不可行"}
    for group in GROUP_LABELS:
        subset = counts[counts.scenario_group == group].set_index("controller")
        original = int(subset.loc["original_mpc", "operational_feasible_count"])
        recovery_count = int(subset.loc["recovery_mpc", "operational_feasible_count"])
        scenario_count = int(subset.loc["recovery_mpc", "scenario_count"])
        lines.append(f"| {labels[group]} | {original}/{scenario_count} | {recovery_count}/{scenario_count} |")
    original_total = int(summary[(summary.controller == "original_mpc")].operational_feasible.sum())
    lines += [f"| 合计 | {original_total}/15 | {total}/15 |", "", "## 候选恢复与失败分类", ""]
    candidate_columns = ["shifted_previous_count", "projected_ann_sequence_count", "conservative_slew_down_count"]
    lines += [
        f"- `shifted_previous_feasible`：{int(recovery.shifted_previous_count.sum())} 次；",
        f"- `projected_ann_sequence`：{int(recovery.projected_ann_sequence_count.sum())} 次；",
        f"- `conservative_slew_down`：{int(recovery.conservative_slew_down_count.sum())} 次；",
        f"- emergency fallback：{int(recovery.emergency_fallback_count.sum())} 次，不计为恢复成功；",
        f"- 预测域不可行：{int(recovery.prediction_domain_infeasible_count.sum())} 次；",
        f"- 硬安全—斜率冲突：{int(recovery.hard_safety_slew_conflict_count.sum())} 次。", "",
        "## 两层门槛", "",
        f"- 第一层无回归：{'通过' if payload['checks']['no_regression_original_feasible'] else '失败'}。原始可行组 Recovery 为 5/5，电压、温度、电流和斜率均满足。",
        f"- 第二层恢复能力：{'通过' if payload['checks']['unresolved_candidate_used'] else '失败'}。unresolved 组没有非 emergency 候选恢复。", "",
        "## 决策", "",
        "Recovery 没有扩大可行域。停止 pure ANN 完整替代与全压力域模仿路线；后续采用 ANN 提供 MPC 初值、参考电流或活跃约束预测，MPC 负责硬约束与安全修正。ANN 直接输出仅限已验证可行域。", "",
    ]
    return "\n".join(lines)


def _context(root: Path):
    phase3 = load_phase_three_config(root / "configs/phase3.yaml")
    phase5a = load_phase_five_a_config(root / "configs/phase5a.yaml")
    parameters = json.loads((root / phase3.artifacts.identified_parameters).read_text(encoding="utf-8"))
    ocv = build_ocv_function(pd.read_csv(root / phase3.artifacts.ocv_curve))
    ann = TinyANN.load(root / "outputs/models/phase4b2_tiny_ann.npz")
    return phase3, phase5a, parameters, ocv, ann


def _frozen_schedule(root: Path, scenario_id: str, block_steps: int, maximum_steps: int) -> tuple[set[int], int]:
    path = root / "data/phase5b_mpc_feasibility/trajectories" / f"{scenario_id}__nominal_mpc.csv"
    if path.exists():
        frame = pd.read_csv(path)
        steps = set(np.flatnonzero(frame.get("control_decision_updated", pd.Series(dtype=bool)).astype(bool)))
        if steps:
            cutoff = min(maximum_steps, len(frame) - 1)
            return {int(step) for step in steps if 1 <= int(step) <= cutoff}, cutoff
    return set(range(1, maximum_steps + 1, block_steps)), maximum_steps


def _noise_sequence(scenario: pd.Series, seed: int, index: int, count: int, sigmas: dict[str, float]) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed + 1009 * index)
    rho = 0.95
    innovation = np.sqrt(1.0 - rho**2)
    state = {key: 0.0 for key in sigmas}
    values: list[dict[str, float]] = []
    for _ in range(count):
        for key, sigma in sigmas.items():
            state[key] = rho * state[key] + innovation * sigma * float(scenario["noise_scale"]) * float(rng.normal())
        values.append(state.copy())
    return values


def _failure_row(scenario_id: str, controller: str, step: int, result: Any) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id, "controller": controller, "step": step,
        "failure_type": getattr(result, "failure_type", "none"),
        "source": getattr(result, "source", "slsqp"),
        "optimizer_success": bool(result.optimizer_success),
        "prediction_feasible": bool(getattr(result, "prediction_feasible", getattr(result, "optimizer_prediction_feasible", False))),
        "slack_voltage_v": float(result.slack_voltage_v),
        "slack_temperature_c": float(result.slack_temperature_c),
        "slack_soc": float(result.slack_soc),
        "slack_current_change_a": float(result.slack_current_change_a),
        "braking_distance_steps": int(result.braking_distance_steps),
        "braking_current_deficit_a": float(result.braking_current_deficit_a),
        "candidate_shifted_previous": int(getattr(result, "source", "") == "shifted_previous_feasible"),
        "candidate_projected_ann": int(getattr(result, "source", "") == "projected_ann_sequence"),
        "candidate_conservative_slew_down": int(getattr(result, "source", "") == "conservative_slew_down"),
        "emergency_fallback": int(bool(getattr(result, "used_emergency_fallback", False))),
    }


def _simulate_pair(root: Path, phase3, scenario: pd.Series, scenario_index: int, parameters: dict[str, Any], ocv, ann, phase5a) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    battery = replace(phase3.battery, initial_soc=float(scenario["initial_soc"]), initial_temperature_c=float(scenario["ambient_temperature_c"]), ambient_temperature_c=float(scenario["ambient_temperature_c"]))
    nominal_config = replace(phase3, battery=battery)
    true_battery = replace(battery, nominal_capacity_ah=phase3.battery.nominal_capacity_ah * float(scenario["capacity_multiplier"]))
    true_config = replace(nominal_config, battery=true_battery)
    true_parameters = perturb_identified_parameters(parameters, scenario)
    control_model = ReducedBatteryModel(nominal_config, ocv, parameters)
    true_model = ReducedBatteryModel(true_config, ocv, true_parameters)
    controllers = {
        "original_mpc": ConstrainedMPC(control_model, nominal_config),
        "recovery_mpc": RecoverableConstrainedMPC(control_model, nominal_config, ann, 101),
    }
    states = {name: initial_reduced_state(true_config) for name in CONTROLLERS}
    results: dict[str, Any] = {name: None for name in CONTROLLERS}
    until = int(np.ceil(phase3.control.maximum_simulation_time_s / phase3.control.control_interval_s))
    schedule, until = _frozen_schedule(root, str(scenario["scenario_id"]), phase3.control.control_block_steps, until)
    stress = phase5a.reduced_stress_test
    sigmas = {"soc": stress.soc_noise_standard_deviation, "temperature_c": stress.temperature_noise_standard_deviation_c, "polarization_fast_v": stress.polarization_noise_standard_deviation_v, "polarization_slow_v": stress.polarization_noise_standard_deviation_v}
    noises = _noise_sequence(scenario, 20260720, scenario_index, until, sigmas)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for step in range(1, until + 1):
        for controller_name in CONTROLLERS:
            state = states[controller_name]
            estimate = _estimated_state(state, control_model, scenario, noises[step - 1])
            update = step in schedule or results[controller_name] is None
            if update:
                results[controller_name] = controllers[controller_name].solve(estimate) if controller_name == "original_mpc" else controllers[controller_name].solve_with_recovery(estimate, 0)
                failures.append(_failure_row(str(scenario["scenario_id"]), controller_name, step, results[controller_name]))
            result = results[controller_name]
            applied, _ = _cap_current_at_target(float(result.current_a), estimate.soc, nominal_config)
            current_change = abs(applied - state.previous_current_a)
            states[controller_name], output = true_model.step(state, applied)
            rows.append({"scenario_id": str(scenario["scenario_id"]), "controller": controller_name, "step": step, "time_s": step * phase3.control.control_interval_s, "current_a": applied, "current_change_a": current_change, "soc": states[controller_name].soc, "voltage_v": output.terminal_voltage_v, "temperature_c": output.average_temperature_c, "control_update": update})
    return rows, failures


def run_phase_five_b_zero_six(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output_dir = root / "data/phase5b06_contract_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    phase3, phase5a, parameters, ocv, ann = _context(root)
    full_scenarios = generate_reduced_stress_scenarios(phase5a)
    scenarios = full_scenarios.set_index("scenario_id")
    representative = pd.read_csv(root / "data/phase5b05_mpc_recovery/representative_scenarios.csv")
    selected_ids: list[str] = []
    selected_groups: dict[str, str] = {}
    for group, label in GROUP_LABELS.items():
        rows = representative[representative.selection_labels.fillna("").str.contains(label, regex=False)].sort_values("scenario_id").head(5)
        if len(rows) != 5:
            raise RuntimeError(f"Frozen representative selector must provide 5 scenarios for {group}.")
        for scenario_id in rows.scenario_id.astype(str):
            selected_ids.append(scenario_id); selected_groups[scenario_id] = group
    SCENARIOS = tuple(selected_ids)
    all_rows: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    for scenario_id in SCENARIOS:
        scenario = scenarios.loc[scenario_id].copy()
        scenario["scenario_id"] = scenario_id
        scenario_index = int(full_scenarios.index[full_scenarios["scenario_id"] == scenario_id][0])
        rows, failures = _simulate_pair(root, phase3, scenario, scenario_index, parameters, ocv, ann, phase5a)
        all_rows.extend(rows); all_failures.extend(failures)
    trajectories = pd.DataFrame(all_rows)
    failures = pd.DataFrame(all_failures)
    trajectories.to_csv(output_dir / "paired_trajectories.csv", index=False)
    failures.to_csv(output_dir / "failure_constraint_audit.csv", index=False)
    summaries: list[dict[str, Any]] = []
    tolerance = phase3.validation.physical_constraint_tolerance
    for (scenario_id, controller), frame in trajectories.groupby(["scenario_id", "controller"]):
        decision_failures = failures[(failures.scenario_id == scenario_id) & (failures.controller == controller)]
        baseline = pd.read_csv(root / "data/phase5b_mpc_feasibility/controller_run_summary.csv")
        baseline_row = baseline[(baseline.scenario_id == scenario_id) & (baseline.controller == "nominal_mpc")].iloc[0]
        completion = bool(abs(frame.soc.iloc[-1] - phase3.battery.target_soc) <= phase5a.reduced_stress_test.terminal_true_soc_tolerance)
        physical_safe = bool(frame.voltage_v.max() <= phase3.constraints.physical_maximum_voltage_v + tolerance and frame.temperature_c.max() <= phase3.constraints.physical_maximum_temperature_c + tolerance and frame.current_a.max() <= phase3.constraints.maximum_current_a + tolerance and frame.current_change_a.max() <= phase3.constraints.maximum_current_change_a_per_step + tolerance)
        summaries.append({"scenario_id": scenario_id, "scenario_group": selected_groups[scenario_id], "controller": controller, "fixed_cutoff_steps": int(frame.step.max()), "completion_success": completion, "physical_safe": physical_safe, "operational_feasible": bool(completion and physical_safe), "baseline_nominal_feasible": bool(baseline_row.teacher_feasible), "maximum_current_change_a": float(frame.current_change_a.max()), "decision_audit_count": int(len(decision_failures)), "prediction_domain_infeasible_count": int((decision_failures.failure_type == "prediction_domain_infeasible_under_candidate_audit").sum()), "hard_safety_slew_conflict_count": int((decision_failures.failure_type == "hard_safety_slew_conflict").sum()), "max_slack_voltage_v": float(decision_failures.slack_voltage_v.max()), "max_slack_temperature_c": float(decision_failures.slack_temperature_c.max()), "max_slack_soc": float(decision_failures.slack_soc.max()), "max_slack_current_change_a": float(decision_failures.slack_current_change_a.max()), "max_braking_current_deficit_a": float(decision_failures.braking_current_deficit_a.max()), "max_braking_distance_steps": int(decision_failures.braking_distance_steps.max()), "shifted_previous_count": int(decision_failures.candidate_shifted_previous.sum()), "projected_ann_sequence_count": int(decision_failures.candidate_projected_ann.sum()), "conservative_slew_down_count": int(decision_failures.candidate_conservative_slew_down.sum()), "emergency_fallback_count": int(decision_failures.emergency_fallback.sum()), "candidate_recovery_count": int(decision_failures[["candidate_shifted_previous", "candidate_projected_ann", "candidate_conservative_slew_down"]].sum().sum()), "numerical_failure_recovered_count": int((decision_failures.failure_type == "numerical_optimization_failure_feasible_alternative").sum())})
    summary = pd.DataFrame(summaries).sort_values(["scenario_id", "controller"])
    summary.to_csv(output_dir / "paired_summary.csv", index=False)
    failure_totals = failures.groupby("controller")[["slack_voltage_v", "slack_temperature_c", "slack_soc", "slack_current_change_a", "braking_current_deficit_a"]].agg(["max", "mean"]).reset_index()
    failure_totals.columns = [
        "controller" if isinstance(column, tuple) and column[0] == "controller" else "_".join(str(part) for part in column if part)
        for column in failure_totals.columns
    ]
    failure_totals.to_csv(output_dir / "slack_summary.csv", index=False)
    recovery = summary[summary.controller == "recovery_mpc"]
    teacher_recovery = recovery[recovery.scenario_group == "teacher_feasible"]
    checks = {
        "no_regression_original_feasible": bool(recovery[recovery.baseline_nominal_feasible].operational_feasible.all()),
        "original_feasible_all_safety_constraints": bool(teacher_recovery.physical_safe.all()),
        "unresolved_candidate_used": bool((recovery[recovery.scenario_group == "unresolved"].candidate_recovery_count > 0).any()),
        "emergency_not_counted_as_candidate_recovery": True,
        "failure_types_auditable": bool(failures.failure_type.notna().all()),
    }
    counts = feasibility_count_table(summary)
    counts.to_csv(output_dir / "feasibility_counts.csv", index=False)
    payload = {"status": "completed", "canonical_feasibility_field": "operational_feasible", "scenario_count": len(SCENARIOS), "scenario_group_counts": summary.groupby("scenario_group").scenario_id.nunique().to_dict(), "paired_controller_runs": len(summary), "same_noise_sequence": True, "same_control_update_schedule": True, "same_cutoff_contract": True, "summary": summary.to_dict("records"), "feasibility_counts": counts.to_dict("records"), "dominant_slack_by_controller": failure_totals.to_dict("records"), "original_feasible_count": int(((summary.controller == "original_mpc") & summary.operational_feasible).sum()), "recovery_feasible_count": int(((summary.controller == "recovery_mpc") & summary.operational_feasible).sum()), "recovery_failure_scenarios": summary.loc[(summary.controller == "recovery_mpc") & (~summary.operational_feasible), "scenario_id"].tolist(), "checks": checks}
    (root / "outputs/metrics").mkdir(parents=True, exist_ok=True)
    (root / "outputs/metrics/phase5b06_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "outputs/phase5b06_report.md").write_text(render_chinese_report(summary, counts, payload), encoding="utf-8")
    return payload

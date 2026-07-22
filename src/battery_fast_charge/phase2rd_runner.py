"""Phase 2R-D：pure DNN 路线最终多起点、完整记忆和邻域敏感性判别。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import nbformat as nbf
from nbclient import NotebookClient
import numpy as np
import pandas as pd
import yaml

from .mpc import ConstrainedMPC, ReducedBatteryModel, ReducedState
from .phase6b_runner import _load_context
from .phase6r_config import load_phase_six_r_config
from .phase6r_teacher import assign_trajectory_splits, design_initial_states, row_to_rolling_state, state_features
from .phase2r_state_audit import _local_variance_metrics


@dataclass(frozen=True)
class PhaseTwoRDConfig:
    name: str
    random_seed: int
    phase6r_config: str
    trajectory_count: int
    trajectory_steps: int
    trajectory_checkpoint: int
    minimum_acceptance: float
    state_count: int
    warm_start_count: int
    state_checkpoint: int
    relative_objective_tolerance: float
    absolute_objective_tolerance: float
    action_range_a: float
    maximum_ambiguous_fraction: float
    neighbor_counts: tuple[int, ...]
    local_std_limit_a: float
    neighbor_p95_limit_a: float
    minimum_mode_samples: int


def load_phase_two_rd_config(path: str | Path) -> PhaseTwoRDConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    full, multi, local = raw["full_sequence_dataset"], raw["multistart"], raw["local_variance"]
    return PhaseTwoRDConfig(
        name=str(raw["study"]["name"]), random_seed=int(raw["study"]["random_seed"]),
        phase6r_config=str(raw["sources"]["phase6r_config"]),
        trajectory_count=int(full["trajectory_count"]), trajectory_steps=int(full["trajectory_steps"]),
        trajectory_checkpoint=int(full["checkpoint_interval_trajectories"]), minimum_acceptance=float(full["minimum_acceptance_fraction"]),
        state_count=int(multi["representative_state_count"]), warm_start_count=int(multi["warm_start_count"]),
        state_checkpoint=int(multi["checkpoint_interval_states"]), relative_objective_tolerance=float(multi["near_optimal_relative_tolerance"]),
        absolute_objective_tolerance=float(multi["near_optimal_absolute_tolerance"]), action_range_a=float(multi["meaningful_first_action_range_a"]),
        maximum_ambiguous_fraction=float(multi["maximum_ambiguous_state_fraction"]),
        neighbor_counts=tuple(int(v) for v in local["neighbor_counts"]), local_std_limit_a=float(local["sufficient_local_standard_deviation_a"]),
        neighbor_p95_limit_a=float(local["sufficient_p95_neighbor_label_difference_a"]), minimum_mode_samples=int(local["minimum_mode_sample_count"]),
    )


def _state_from_row(row: pd.Series) -> ReducedState:
    return ReducedState(float(row.state_soc), float(row.state_polarization_fast_v), float(row.state_polarization_slow_v), float(row.state_core_temperature_c), float(row.state_surface_temperature_c), float(row.state_previous_current_a))


def _mode(voltage: bool, temperature: bool, slew: bool) -> str:
    active = [name for name, flag in (("voltage", voltage), ("temperature", temperature), ("slew", slew)) if flag]
    return "+".join(active) if active else "interior"


def _full_sequence_trajectory(initial: pd.Series, split: str, model: ReducedBatteryModel, phase3: Any, steps: int):
    controller = ConstrainedMPC(model, phase3); state = row_to_rolling_state(initial)
    previous_plan: np.ndarray | None = None; rows: list[dict[str, Any]] = []; rejection = ""
    for step in range(steps):
        available = previous_plan is not None
        stored = np.full(controller.number_of_blocks, state.previous_current_a) if previous_plan is None else previous_plan.copy()
        result = controller.solve(state); plan = controller.last_optimal_block_currents_a
        if not result.optimizer_success or not result.prediction_feasible or result.used_fallback or plan is None:
            rejection = result.status; break
        current = float(result.current_a); next_state, _ = model.step(state, current)
        voltage = result.predicted_maximum_voltage_v >= phase3.constraints.mpc_maximum_voltage_v - 0.01
        temperature = result.predicted_maximum_temperature_c >= phase3.constraints.mpc_maximum_temperature_c - 0.10
        slew = abs(current - state.previous_current_a) >= phase3.constraints.maximum_current_change_a_per_step - 0.05
        row = {
            "trajectory_id": str(initial.trajectory_id), "step_index": step, **state_features(state, phase3.battery.ambient_temperature_c),
            "state_average_temperature_c": model.average_temperature(state), "teacher_current_a": current,
            "previous_plan_available": float(available), "previous_plan_first_a": float(stored[0]), "previous_plan_mean_a": float(stored.mean()),
            "previous_plan_last_a": float(stored[-1]), "previous_plan_range_a": float(np.ptp(stored)),
            "active_voltage": float(voltage), "active_temperature": float(temperature), "active_slew": float(slew),
            "active_mode": _mode(voltage, temperature, slew), "split": split,
        }
        row.update({f"previous_plan_block_{i:02d}_a": float(value) for i, value in enumerate(stored)})
        row.update({f"current_plan_block_{i:02d}_a": float(value) for i, value in enumerate(plan)})
        rows.append(row); previous_plan = plan.copy(); state = next_state
    accepted = len(rows) == steps
    return (rows if accepted else []), {"trajectory_id": str(initial.trajectory_id), "split": split, "teacher_accepted": accepted, "completed_step_count": len(rows), "rejection_reason": rejection}


def _generate_full_sequence_dataset(config: PhaseTwoRDConfig, root: Path, data_dir: Path):
    phase6r = load_phase_six_r_config(root / config.phase6r_config)
    if (config.trajectory_count, config.trajectory_steps) != (phase6r.teacher_data.initial_trajectory_count, phase6r.teacher_data.trajectory_steps):
        raise ValueError("2R-D 必须保持 Phase 6R 的 240×8 合同。")
    phase3, parameters, ocv = _load_context(phase6r, root); model = ReducedBatteryModel(phase3, ocv, parameters)
    design = design_initial_states(phase6r, phase3.battery.ambient_temperature_c); split = assign_trajectory_splits(design.trajectory_id.tolist(), phase6r)
    attempt_path, dataset_path = data_dir / "teacher_attempts.csv", data_dir / "full_control_sequence_teacher.csv"
    attempts = pd.read_csv(attempt_path).to_dict("records") if attempt_path.exists() else []
    rows = pd.read_csv(dataset_path).to_dict("records") if dataset_path.exists() else []
    completed = {str(v["trajectory_id"]) for v in attempts}
    for _, initial in design.iterrows():
        if str(initial.trajectory_id) in completed: continue
        result_rows, attempt = _full_sequence_trajectory(initial, split[str(initial.trajectory_id)], model, phase3, config.trajectory_steps)
        rows.extend(result_rows); attempts.append(attempt); completed.add(str(initial.trajectory_id))
        if len(completed) % config.trajectory_checkpoint == 0:
            pd.DataFrame(attempts).to_csv(attempt_path, index=False); pd.DataFrame(rows).to_csv(dataset_path, index=False)
            print(f"D2 checkpoint {len(completed)}/{len(design)}", flush=True)
    attempts_frame, dataset = pd.DataFrame(attempts), pd.DataFrame(rows)
    attempts_frame.to_csv(attempt_path, index=False); dataset.to_csv(dataset_path, index=False)
    return dataset, attempts_frame, model, phase3


def _representative_states(dataset: pd.DataFrame, count: int) -> pd.DataFrame:
    ordered = dataset.sort_values(["active_mode", "state_soc", "state_previous_current_a", "state_average_temperature_c"]).reset_index(drop=True)
    indices = np.unique(np.linspace(0, len(ordered) - 1, count).round().astype(int))
    selected = ordered.iloc[indices].copy()
    if len(selected) < count:
        extra = ordered.drop(index=indices).head(count - len(selected)); selected = pd.concat([selected, extra])
    selected = selected.head(count).reset_index(drop=True); selected.insert(0, "state_id", [f"state_{i:03d}" for i in range(len(selected))])
    return selected


def _slew_path(previous: float, targets: np.ndarray, maximum: float, delta: float) -> np.ndarray:
    values = []; current = previous
    for target in targets:
        current = float(np.clip(target, current - delta, current + delta)); current = float(np.clip(current, 0.0, maximum)); values.append(current)
    return np.asarray(values)


def _warm_starts(row: pd.Series, number_of_blocks: int, count: int, maximum: float, delta: float, seed: int):
    rng = np.random.default_rng(seed); previous = float(row.state_previous_current_a)
    native = np.asarray([row[f"previous_plan_block_{i:02d}_a"] for i in range(number_of_blocks)], dtype=float)
    starts: list[tuple[str, np.ndarray | None]] = [("default_ramp", None), ("native_previous_plan", native)]
    starts.append(("conservative_down", _slew_path(previous, np.zeros(number_of_blocks), maximum, delta)))
    starts.append(("aggressive_up", _slew_path(previous, np.full(number_of_blocks, maximum), maximum, delta)))
    levels = np.linspace(0.0, maximum, 4)
    for index, level in enumerate(levels): starts.append((f"level_{index}", _slew_path(previous, np.full(number_of_blocks, level), maximum, delta)))
    while len(starts) < count:
        targets = rng.uniform(0.0, maximum, number_of_blocks)
        starts.append((f"random_walk_{len(starts):02d}", _slew_path(previous, targets, maximum, delta)))
    return starts[:count]


def _run_multistart(selected: pd.DataFrame, model: ReducedBatteryModel, phase3: Any, config: PhaseTwoRDConfig, data_dir: Path):
    path = data_dir / "multistart_solutions.csv"; records = pd.read_csv(path).to_dict("records") if path.exists() else []
    completed = {str(v["state_id"]) for v in records if sum(str(r["state_id"]) == str(v["state_id"]) for r in records) >= config.warm_start_count}
    for _, row in selected.iterrows():
        state_id = str(row.state_id)
        if state_id in completed: continue
        state = _state_from_row(row)
        starts = _warm_starts(row, phase3.control.number_of_control_blocks, config.warm_start_count, phase3.constraints.maximum_current_a, phase3.constraints.maximum_current_change_a_per_step, config.random_seed + int(state_id[-3:]))
        for start_index, (kind, warm) in enumerate(starts):
            controller = ConstrainedMPC(model, phase3); controller.set_initial_block_currents_a(warm); result = controller.solve(state)
            plan = controller.last_optimal_block_currents_a
            voltage = result.predicted_maximum_voltage_v >= phase3.constraints.mpc_maximum_voltage_v - 0.01
            temperature = result.predicted_maximum_temperature_c >= phase3.constraints.mpc_maximum_temperature_c - 0.10
            slew = abs(result.current_a - state.previous_current_a) >= phase3.constraints.maximum_current_change_a_per_step - 0.05
            records.append({"state_id": state_id, "trajectory_id": row.trajectory_id, "step_index": int(row.step_index), "warm_start_index": start_index, "warm_start_kind": kind,
                "first_action_a": result.current_a, "objective_value": result.objective_value, "optimizer_success": result.optimizer_success,
                "prediction_feasible": result.prediction_feasible, "used_fallback": result.used_fallback, "active_mode": _mode(voltage, temperature, slew),
                "status": result.status, "minimum_constraint_margin": result.minimum_constraint_margin,
                **{f"solution_block_{i:02d}_a": float(value) for i, value in enumerate(plan if plan is not None else np.full(phase3.control.number_of_control_blocks, np.nan))}})
        if (int(state_id[-3:]) + 1) % config.state_checkpoint == 0:
            pd.DataFrame(records).to_csv(path, index=False); print(f"D1 checkpoint {int(state_id[-3:])+1}/{len(selected)}", flush=True)
    frame = pd.DataFrame(records); frame.to_csv(path, index=False); return frame


def _summarize_multistart(frame: pd.DataFrame, config: PhaseTwoRDConfig):
    rows = []
    for state_id, group in frame.groupby("state_id"):
        successful = group[group.optimizer_success & group.prediction_feasible & ~group.used_fallback]
        if successful.empty:
            action_range = near_range = objective_range = np.nan; near_count = 0
        else:
            best = float(successful.objective_value.min()); tolerance = max(config.absolute_objective_tolerance, abs(best) * config.relative_objective_tolerance)
            near = successful[successful.objective_value <= best + tolerance]
            action_range = float(successful.first_action_a.max() - successful.first_action_a.min())
            near_range = float(near.first_action_a.max() - near.first_action_a.min()); objective_range = float(successful.objective_value.max() - best); near_count = len(near)
        rows.append({"state_id": state_id, "successful_feasible_count": len(successful), "feasible_count": int(group.prediction_feasible.sum()), "fallback_count": int(group.used_fallback.sum()),
            "optimizer_success_count": int(group.optimizer_success.sum()), "first_action_range_a": action_range, "near_optimal_first_action_range_a": near_range,
            "objective_range": objective_range, "near_optimal_solution_count": near_count, "active_mode_count": int(group.active_mode.nunique()), "status_count": int(group.status.nunique()),
            "near_optimal_multivalued": bool(np.isfinite(near_range) and near_range > config.action_range_a),
            "warm_start_sensitive": bool((np.isfinite(action_range) and action_range > config.action_range_a) or group.prediction_feasible.nunique() > 1 or group.optimizer_success.nunique() > 1)})
    table = pd.DataFrame(rows); ambiguous = float(table.near_optimal_multivalued.mean())
    summary = {"state_count": len(table), "warm_starts_per_state": config.warm_start_count, "near_optimal_multivalued_state_count": int(table.near_optimal_multivalued.sum()),
        "near_optimal_multivalued_fraction": ambiguous, "warm_start_sensitive_state_count": int(table.warm_start_sensitive.sum()),
        "all_starts_feasible_state_count": int((table.feasible_count == config.warm_start_count).sum()), "maximum_near_optimal_first_action_range_a": float(table.near_optimal_first_action_range_a.max()),
        "control_law_multivalued": bool(ambiguous > config.maximum_ambiguous_fraction)}
    return table, summary


def _variance_audit(dataset: pd.DataFrame, config: PhaseTwoRDConfig):
    base = ["state_soc", "state_polarization_fast_v", "state_polarization_slow_v", "state_average_temperature_c", "state_previous_current_a"]
    summary_features = [*base, "previous_plan_first_a", "previous_plan_mean_a", "previous_plan_last_a", "previous_plan_range_a", "previous_plan_available"]
    blocks = [f"previous_plan_block_{i:02d}_a" for i in range(12)]
    feature_sets = [("five_state", base), ("five_state_plus_summary", summary_features), ("five_state_plus_full_previous_sequence", [*base, *blocks, "previous_plan_available"])]
    groups = [("all", dataset)]
    groups.extend([(name, dataset[dataset.active_mode == name]) for name in ["interior", "voltage", "temperature", "slew", "voltage+temperature", "voltage+slew", "temperature+slew", "voltage+temperature+slew"]])
    rows = []
    for group_name, group in groups:
        for k in config.neighbor_counts:
            if len(group) < max(config.minimum_mode_samples, k + 2):
                for feature_name, features in feature_sets: rows.append({"mode_group": group_name, "sample_count": len(group), "neighbor_count": k, "feature_set": feature_name, "features": ",".join(features), "auditable": False})
                continue
            for feature_name, features in feature_sets:
                metrics = _local_variance_metrics(group, features, k)
                rows.append({"mode_group": group_name, "sample_count": len(group), "neighbor_count": k, "feature_set": feature_name, "features": ",".join(features), "auditable": True, **metrics,
                    "locally_sufficient": bool(metrics["mean_local_standard_deviation_a"] <= config.local_std_limit_a and metrics["nearest_neighbor_label_difference_p95_a"] <= config.neighbor_p95_limit_a)})
    table = pd.DataFrame(rows)
    overall = table[(table.mode_group == "all") & table.auditable]
    full25 = overall[(overall.neighbor_count == 25) & (overall.feature_set == "five_state_plus_full_previous_sequence")].iloc[0]
    robustness = overall[overall.feature_set == "five_state_plus_full_previous_sequence"]
    mode_k25 = table[(table.neighbor_count == 25) & (table.mode_group != "all")]
    summary = {"full_sequence_k25_mean_local_standard_deviation_a": float(full25.mean_local_standard_deviation_a),
        "full_sequence_k25_neighbor_label_difference_p95_a": float(full25.nearest_neighbor_label_difference_p95_a),
        "full_sequence_k25_sufficient": bool(full25.locally_sufficient), "full_sequence_all_k_sufficient": bool(robustness.locally_sufficient.all()),
        "auditable_mode_groups_at_k25": sorted(mode_k25[mode_k25.auditable].mode_group.unique().tolist()),
        "unauditable_mode_groups_at_k25": sorted(mode_k25[~mode_k25.auditable].mode_group.unique().tolist())}
    return table, summary


def _write_report(path: Path, payload: dict[str, Any]):
    d1, d23, decision = payload["d1_multistart"], payload["d2_d3_variance"], payload["decision"]
    path.write_text(f"""# Phase 2R-D：pure DNN 路线最终判别报告

## D1 相同状态多起点

- 状态数：{d1['state_count']}；每状态 warm start：{d1['warm_starts_per_state']}。
- 近最优第一动作多值状态：{d1['near_optimal_multivalued_state_count']}（{100*d1['near_optimal_multivalued_fraction']:.1f}%）。
- warm-start 敏感状态：{d1['warm_start_sensitive_state_count']}。
- 最大近最优第一动作极差：{d1['maximum_near_optimal_first_action_range_a']:.4f} A。
- 判定：{'存在不可忽略的近最优多值性' if d1['control_law_multivalued'] else '未发现超过门槛的系统性近最优多值性'}。

## D2/D3 完整序列、邻域与模式

- 完整上一序列，K=25：局部标准差 {d23['full_sequence_k25_mean_local_standard_deviation_a']:.4f} A；最近邻标签差 P95 {d23['full_sequence_k25_neighbor_label_difference_p95_a']:.4f} A。
- K=25 达标：{d23['full_sequence_k25_sufficient']}；K=5/10/25/50 全部达标：{d23['full_sequence_all_k_sufficient']}。
- K=25 可审计模式：{', '.join(d23['auditable_mode_groups_at_k25']) or '无'}。
- K=25 样本不足模式：{', '.join(d23['unauditable_mode_groups_at_k25']) or '无'}。

## 最终决策

**{decision['conclusion']}**

理由：{decision['reason']} 本阶段不训练 ANN；求解后活跃模式只用于诊断，不作为 pure DNN 的在线输入。
""", encoding="utf-8")


def _write_notebook(path: Path):
    nb = nbf.v4.new_notebook(); nb.metadata["kernelspec"] = {"display_name":"Python 3","language":"python","name":"python3"}
    nb.cells = [nbf.v4.new_markdown_cell("# Phase 2R-D：pure DNN 最终判别\n\n只读取冻结结果，不重新求解 MPC。"),
        nbf.v4.new_code_cell("""from pathlib import Path\nimport json, pandas as pd\nfrom IPython.display import display\nROOT=Path.cwd(); OUT=ROOT/'outputs'/'phase2rd_final_discrimination'\nif not OUT.exists(): ROOT=ROOT.parent; OUT=ROOT/'outputs'/'phase2rd_final_discrimination'\nm=json.loads((OUT/'metrics.json').read_text(encoding='utf-8')); d1=pd.read_csv(OUT/'multistart_state_summary.csv'); v=pd.read_csv(OUT/'local_variance_sensitivity.csv'); print(m['decision'])"""),
        nbf.v4.new_markdown_cell("## D1：完全相同状态的多起点求解"), nbf.v4.new_code_cell("""display(pd.DataFrame([m['d1_multistart']])); display(d1.sort_values('near_optimal_first_action_range_a',ascending=False).head(10))"""),
        nbf.v4.new_markdown_cell("## D2/D3：完整序列与 K/模式敏感性"), nbf.v4.new_code_cell("""view=v[(v.mode_group=='all') & (v.auditable==True)][['neighbor_count','feature_set','mean_local_standard_deviation_a','nearest_neighbor_label_difference_p95_a','locally_sufficient']]; display(view)"""),
        nbf.v4.new_code_cell("""modes=v[(v.mode_group!='all') & (v.auditable==True) & (v.feature_set=='five_state_plus_full_previous_sequence')]; display(modes[['mode_group','sample_count','neighbor_count','mean_local_standard_deviation_a','nearest_neighbor_label_difference_p95_a','locally_sufficient']])"""),
        nbf.v4.new_markdown_cell("## 最终判别"), nbf.v4.new_code_cell("""display(pd.DataFrame([m['decision']])); assert m['status']=='completed'""")]
    path.parent.mkdir(parents=True, exist_ok=True); nbf.write(nb,path)
    executed=nbf.read(path,as_version=4); NotebookClient(executed,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(path.parents[1])}}).execute(); nbf.write(executed,path)


def run_phase_two_rd(config: PhaseTwoRDConfig, project_root: str | Path):
    root=Path(project_root).resolve(); data=root/'data'/'phase2rd_final_discrimination'; out=root/'outputs'/'phase2rd_final_discrimination'; data.mkdir(parents=True,exist_ok=True); out.mkdir(parents=True,exist_ok=True)
    dataset, attempts, model, phase3 = _generate_full_sequence_dataset(config,root,data)
    selected_path=data/'representative_states_100.csv'; selected=pd.read_csv(selected_path) if selected_path.exists() else _representative_states(dataset,config.state_count)
    selected.to_csv(selected_path,index=False); solutions=_run_multistart(selected,model,phase3,config,data)
    d1_table,d1_summary=_summarize_multistart(solutions,config); d1_table.to_csv(out/'multistart_state_summary.csv',index=False)
    variance,d23_summary=_variance_audit(dataset,config); variance.to_csv(out/'local_variance_sensitivity.csv',index=False)
    acceptance=float(attempts.teacher_accepted.mean()); contract=acceptance>=config.minimum_acceptance
    continue_pure=bool(contract and not d1_summary['control_law_multivalued'] and d23_summary['full_sequence_k25_sufficient'] and d23_summary['full_sequence_all_k_sufficient'])
    if continue_pure: conclusion='pure DNN 路线仍值得做一次受限的多种子输入扩充实验'; reason='多起点未显示系统性多值，且完整控制记忆对不同 K 均达到单值性门槛。'
    elif d1_summary['control_law_multivalued']: conclusion='停止 pure DNN 直接替代路线'; reason='相同完整状态下仍存在不可忽略的近最优第一动作多值性。'
    else: conclusion='停止 pure DNN 直接替代路线'; reason='即使加入完整上一控制序列，局部单值性仍未稳健达到门槛。'
    payload={'study_name':config.name,'status':'completed' if contract else 'teacher_contract_failed','teacher':{'attempted_trajectories':len(attempts),'accepted_trajectories':int(attempts.teacher_accepted.sum()),'acceptance_fraction':acceptance,'sample_count':len(dataset),'contract_success':contract},
        'd1_multistart':d1_summary,'d2_d3_variance':d23_summary,'decision':{'continue_pure_dnn':continue_pure,'conclusion':conclusion,'reason':reason}}
    (out/'metrics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8'); _write_report(out/'PHASE2R-D最终判别报告.md',payload)
    _write_notebook(root/'notebooks'/'phase2rd_final_pure_dnn_discrimination_results.ipynb'); print(json.dumps({'status':payload['status'],'decision':payload['decision']},ensure_ascii=False),flush=True); return payload

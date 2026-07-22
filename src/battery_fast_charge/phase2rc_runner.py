"""Phase 2R-C：前瞻式原生控制记忆与控制律单值性审计。"""

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

from .mpc import ConstrainedMPC, ReducedBatteryModel
from .phase6b_runner import _load_context
from .phase6r_config import load_phase_six_r_config
from .phase6r_teacher import assign_trajectory_splits, design_initial_states, row_to_rolling_state, state_features
from .phase2r_state_audit import _local_variance_metrics


@dataclass(frozen=True)
class PhaseTwoRCConfig:
    name: str
    random_seed: int
    phase6r_config: str
    phase2r_config: str
    trajectory_count: int
    trajectory_steps: int
    checkpoint_interval: int
    minimum_acceptance_fraction: float
    neighbor_count: int
    significant_reduction: float
    sufficient_local_std_a: float
    sufficient_neighbor_p95_a: float


def load_phase_two_rc_config(path: str | Path) -> PhaseTwoRCConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return PhaseTwoRCConfig(
        name=str(raw["study"]["name"]),
        random_seed=int(raw["study"]["random_seed"]),
        phase6r_config=str(raw["sources"]["phase6r_config"]),
        phase2r_config=str(raw["sources"]["phase2r_config"]),
        trajectory_count=int(raw["teacher_data"]["trajectory_count"]),
        trajectory_steps=int(raw["teacher_data"]["trajectory_steps"]),
        checkpoint_interval=int(raw["teacher_data"]["checkpoint_interval_trajectories"]),
        minimum_acceptance_fraction=float(raw["teacher_data"]["minimum_acceptance_fraction"]),
        neighbor_count=int(raw["state_audit"]["neighbor_count"]),
        significant_reduction=float(raw["state_audit"]["significant_variance_reduction_fraction"]),
        sufficient_local_std_a=float(raw["state_audit"]["sufficient_local_standard_deviation_a"]),
        sufficient_neighbor_p95_a=float(raw["state_audit"]["sufficient_p95_neighbor_label_difference_a"]),
    )


def _trajectory(initial: pd.Series, split: str, model: ReducedBatteryModel, phase3: Any, steps: int):
    controller = ConstrainedMPC(model, phase3)
    state = row_to_rolling_state(initial)
    previous_plan: np.ndarray | None = None
    rows: list[dict[str, Any]] = []
    solve_times: list[float] = []
    rejection = ""
    for step_index in range(steps):
        if previous_plan is None:
            previous_first = previous_mean = previous_last = float(state.previous_current_a)
            previous_range = 0.0
            previous_plan_available = 0.0
        else:
            previous_first = float(previous_plan[0])
            previous_mean = float(np.mean(previous_plan))
            previous_last = float(previous_plan[-1])
            previous_range = float(np.ptp(previous_plan))
            previous_plan_available = 1.0
        result = controller.solve(state)
        solve_times.append(float(result.solve_time_s))
        plan = controller.last_optimal_block_currents_a
        if (
            not result.optimizer_success
            or not result.prediction_feasible
            or result.used_fallback
            or plan is None
        ):
            rejection = str(result.status)
            break
        current = float(result.current_a)
        next_state, _ = model.step(state, current)
        voltage_margin = float(phase3.constraints.mpc_maximum_voltage_v - result.predicted_maximum_voltage_v)
        temperature_margin = float(phase3.constraints.mpc_maximum_temperature_c - result.predicted_maximum_temperature_c)
        slew_margin = float(phase3.constraints.maximum_current_change_a_per_step - abs(current - state.previous_current_a))
        rows.append({
            "trajectory_id": str(initial["trajectory_id"]),
            "step_index": step_index,
            **state_features(state, phase3.battery.ambient_temperature_c),
            "state_average_temperature_c": model.average_temperature(state),
            "teacher_current_a": current,
            "teacher_solve_time_s": float(result.solve_time_s),
            "previous_plan_available": previous_plan_available,
            "previous_plan_first_a": previous_first,
            "previous_plan_mean_a": previous_mean,
            "previous_plan_last_a": previous_last,
            "previous_plan_range_a": previous_range,
            "current_plan_first_a": float(plan[0]),
            "current_plan_mean_a": float(np.mean(plan)),
            "current_plan_last_a": float(plan[-1]),
            "voltage_margin_v": voltage_margin,
            "temperature_margin_c": temperature_margin,
            "slew_margin_a": slew_margin,
            "mode_voltage_active": float(voltage_margin <= 0.01),
            "mode_temperature_active": float(temperature_margin <= 0.10),
            "mode_slew_active": float(slew_margin <= 0.05),
            "predicted_terminal_soc": float(result.predicted_terminal_soc),
            "target_cap_active": float(result.predicted_terminal_soc >= phase3.battery.target_soc - 0.002),
            "split": split,
        })
        previous_plan = np.asarray(plan, dtype=float).copy()
        state = next_state
    accepted = len(rows) == steps
    attempt = {
        "trajectory_id": str(initial["trajectory_id"]),
        "split": split,
        "teacher_accepted": accepted,
        "completed_step_count": len(rows),
        "rejection_reason": rejection,
        "mean_solve_time_s": float(np.mean(solve_times)) if solve_times else 0.0,
        "maximum_solve_time_s": float(np.max(solve_times)) if solve_times else 0.0,
    }
    return (rows if accepted else []), attempt


def _audit(dataset: pd.DataFrame, config: PhaseTwoRCConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = ["state_soc", "state_polarization_fast_v", "state_polarization_slow_v", "state_average_temperature_c", "state_previous_current_a"]
    feature_sets = [
        ("current_dnn_5", base),
        ("plus_previous_plan_first", [*base, "previous_plan_first_a"]),
        ("plus_previous_plan_mean_last", [*base, "previous_plan_mean_a", "previous_plan_last_a"]),
        ("plus_previous_plan_full_summary", [*base, "previous_plan_first_a", "previous_plan_mean_a", "previous_plan_last_a", "previous_plan_range_a", "previous_plan_available"]),
        ("plus_online_pre_solve_memory", [*base, "previous_plan_first_a", "previous_plan_mean_a", "previous_plan_last_a", "previous_plan_range_a", "previous_plan_available", "step_index"]),
        ("diagnostic_post_solve_modes", [*base, "mode_voltage_active", "mode_temperature_active", "mode_slew_active", "target_cap_active"]),
    ]
    records = []
    for name, features in feature_sets:
        records.append({"feature_set": name, "features": ",".join(features), **_local_variance_metrics(dataset, features, config.neighbor_count)})
    table = pd.DataFrame(records)
    baseline = float(table.loc[table.feature_set == "current_dnn_5", "mean_conditional_variance_a2"].iloc[0])
    table["variance_reduction_vs_current_dnn_fraction"] = 1.0 - table.mean_conditional_variance_a2 / baseline
    table["significant_reduction"] = table.variance_reduction_vs_current_dnn_fraction >= config.significant_reduction
    deployable = table[table.feature_set == "plus_previous_plan_full_summary"].iloc[0]
    summary = {
        "baseline_mean_local_standard_deviation_a": float(table.iloc[0].mean_local_standard_deviation_a),
        "baseline_neighbor_label_difference_p95_a": float(table.iloc[0].nearest_neighbor_label_difference_p95_a),
        "native_memory_mean_local_standard_deviation_a": float(deployable.mean_local_standard_deviation_a),
        "native_memory_neighbor_label_difference_p95_a": float(deployable.nearest_neighbor_label_difference_p95_a),
        "native_memory_variance_reduction_fraction": float(deployable.variance_reduction_vs_current_dnn_fraction),
        "native_memory_significant": bool(deployable.significant_reduction),
        "native_memory_locally_sufficient": bool(
            deployable.mean_local_standard_deviation_a <= config.sufficient_local_std_a
            and deployable.nearest_neighbor_label_difference_p95_a <= config.sufficient_neighbor_p95_a
        ),
        "thresholds": {
            "mean_local_standard_deviation_a": config.sufficient_local_std_a,
            "neighbor_label_difference_p95_a": config.sufficient_neighbor_p95_a,
            "variance_reduction_fraction": config.significant_reduction,
        },
    }
    return table, summary


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    teacher, state = payload["teacher"], payload["state_audit"]
    conclusion = "达到局部单值性门槛" if state["native_memory_locally_sufficient"] else "仍未达到局部单值性门槛"
    text = f"""# Phase 2R-C：前瞻式原生控制记忆审计报告

## 结论

- 原生记录教师数据：{teacher['accepted_trajectory_count']}/{teacher['attempted_trajectory_count']} 条轨迹接受，共 {teacher['sample_count']} 个样本。
- 五状态基线平均局部标准差：{state['baseline_mean_local_standard_deviation_a']:.4f} A。
- 加入上一最优序列原生摘要后：{state['native_memory_mean_local_standard_deviation_a']:.4f} A，方差下降 {100*state['native_memory_variance_reduction_fraction']:.1f}%。
- 最近邻标签差 P95：{state['native_memory_neighbor_label_difference_p95_a']:.4f} A。
- 判定：**{conclusion}**。

## 合同边界

本阶段只运行 25 ℃、固定参数、同模型滚动 MPC，不训练 ANN，也不运行 Phase 5A 压力场景。上一最优序列在每次 MPC 求解前直接记录，不使用事后重放。MPC 活跃模式和 target-cap 为求解后诊断变量，不可冒充求解前可部署输入。

## 下一步

{'可把原生控制记忆作为候选输入进入小规模多种子 ANN 消融，但仍须保留冻结测试。' if state['native_memory_locally_sufficient'] else '停止直接训练扩充输入 ANN；先检查完整控制器记忆、求解器分支和邻域采样密度，或转向 ANN 初值/参考 + MPC 安全修正。'}
"""
    path.write_text(text, encoding="utf-8")


def _write_notebook(path: Path, output_dir: Path) -> None:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell("# Phase 2R-C：前瞻式原生控制记忆审计\n\n只读取冻结结果，不重新运行 MPC。"),
        nbf.v4.new_code_cell("""from pathlib import Path\nimport json, pandas as pd\nfrom IPython.display import display\nROOT=Path.cwd(); OUT=ROOT/'outputs'/'phase2rc_prospective_memory_audit'\nif not OUT.exists(): ROOT=ROOT.parent; OUT=ROOT/'outputs'/'phase2rc_prospective_memory_audit'\nm=json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))\nt=pd.read_csv(OUT/'state_conditional_variance_metrics.csv')\nprint(m['status']); display(pd.DataFrame([m['teacher']]))"""),
        nbf.v4.new_markdown_cell("## 条件方差对照"),
        nbf.v4.new_code_cell("""display(t[['feature_set','mean_local_standard_deviation_a','nearest_neighbor_label_difference_p95_a','variance_reduction_vs_current_dnn_fraction','significant_reduction']].style.format({'mean_local_standard_deviation_a':'{:.4f}','nearest_neighbor_label_difference_p95_a':'{:.4f}','variance_reduction_vs_current_dnn_fraction':'{:.1%}'}))"""),
        nbf.v4.new_markdown_cell("## 阶段判定"),
        nbf.v4.new_code_cell("""s=m['state_audit']; display(pd.DataFrame([s])); assert m['teacher']['contract_success']"""),
        nbf.v4.new_markdown_cell("原生摘要是否足以支持下一阶段 ANN 输入消融，以 `native_memory_locally_sufficient` 为准；求解后模式只作诊断。"),
    ]
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)
    executed = nbf.read(path, as_version=4)
    NotebookClient(
        executed,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parents[1])}},
    ).execute()
    nbf.write(executed, path)


def run_phase_two_rc(config: PhaseTwoRCConfig, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    data_dir = root / "data" / "phase2rc_prospective_memory_audit"
    output_dir = root / "outputs" / "phase2rc_prospective_memory_audit"
    data_dir.mkdir(parents=True, exist_ok=True); output_dir.mkdir(parents=True, exist_ok=True)
    phase6r = load_phase_six_r_config(root / config.phase6r_config)
    if config.trajectory_count != phase6r.teacher_data.initial_trajectory_count or config.trajectory_steps != phase6r.teacher_data.trajectory_steps:
        raise ValueError("Phase 2R-C 必须保持 Phase 6R 的 240×8 冻结教师合同。")
    phase3, parameters, ocv = _load_context(phase6r, root)
    model = ReducedBatteryModel(phase3, ocv, parameters)
    design = design_initial_states(phase6r, phase3.battery.ambient_temperature_c)
    split_map = assign_trajectory_splits(design.trajectory_id.tolist(), phase6r)
    attempts_path = data_dir / "teacher_attempts.csv"; dataset_path = data_dir / "native_control_memory_teacher.csv"
    attempts = pd.read_csv(attempts_path).to_dict("records") if attempts_path.exists() else []
    rows = pd.read_csv(dataset_path).to_dict("records") if dataset_path.exists() else []
    completed = {str(row["trajectory_id"]) for row in attempts}
    for _, initial in design.iterrows():
        trajectory_id = str(initial["trajectory_id"])
        if trajectory_id in completed: continue
        trajectory_rows, attempt = _trajectory(initial, split_map[trajectory_id], model, phase3, config.trajectory_steps)
        rows.extend(trajectory_rows); attempts.append(attempt); completed.add(trajectory_id)
        if len(completed) % config.checkpoint_interval == 0:
            pd.DataFrame(attempts).to_csv(attempts_path, index=False)
            pd.DataFrame(rows).to_csv(dataset_path, index=False)
            print(f"checkpoint {len(completed)}/{len(design)}", flush=True)
    attempts_frame = pd.DataFrame(attempts); dataset = pd.DataFrame(rows)
    attempts_frame.to_csv(attempts_path, index=False); dataset.to_csv(dataset_path, index=False)
    audit_table, state_summary = _audit(dataset, config)
    audit_table.to_csv(output_dir / "state_conditional_variance_metrics.csv", index=False)
    acceptance = float(attempts_frame.teacher_accepted.mean())
    teacher_summary = {
        "attempted_trajectory_count": int(len(attempts_frame)),
        "accepted_trajectory_count": int(attempts_frame.teacher_accepted.sum()),
        "acceptance_fraction": acceptance,
        "sample_count": int(len(dataset)),
        "native_previous_plan_rows": int(dataset.previous_plan_available.sum()),
        "contract_success": bool(acceptance >= config.minimum_acceptance_fraction and len(dataset) > 0),
    }
    status = "completed" if teacher_summary["contract_success"] else "teacher_contract_failed"
    payload = {"study_name": config.name, "status": status, "teacher": teacher_summary, "state_audit": state_summary}
    (output_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(output_dir / "PHASE2R-C前瞻式控制记忆审计报告.md", payload)
    notebook_path = root / "notebooks" / "phase2rc_prospective_control_memory_results.ipynb"
    _write_notebook(notebook_path, output_dir)
    print(json.dumps({"status": status, "metrics": str(output_dir / 'metrics.json'), "notebook": str(notebook_path)}, ensure_ascii=False), flush=True)
    return payload

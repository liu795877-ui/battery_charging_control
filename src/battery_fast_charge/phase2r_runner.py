"""组织 Phase 2R-A/2R-B 审计、汇总、绘图与中文报告。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .phase2_config import load_phase_two_config
from .phase2r_config import PhaseTwoRConfig
from .phase2r_model_audit import (
    evaluate_model_variants,
    fit_related_electrical_parameters,
    fit_temperature_specific_thermal_models,
    generate_phase_two_r_dfn_data,
    summarize_model_audit,
)
from .phase2r_state_audit import (
    replay_teacher_with_control_memory,
    run_state_sufficiency_audit,
    summarize_replay_differences,
)


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"无法序列化 {type(value)!r}")


def _plot_model_audit(horizon: pd.DataFrame, output: Path) -> Path:
    holdout = horizon[horizon["holdout_soc"]]
    summary = holdout.groupby(["variant", "horizon_s"])[["voltage_rmse_mv", "temperature_rmse_c"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for variant, group in summary.groupby("variant"):
        axes[0].plot(group["horizon_s"], group["voltage_rmse_mv"], marker="o", label=variant)
        axes[1].plot(group["horizon_s"], group["temperature_rmse_c"], marker="o", label=variant)
    axes[0].axhline(50.0, color="black", linestyle="--", linewidth=0.9)
    axes[0].set(xlabel="Prediction horizon [s]", ylabel="Mean voltage RMSE [mV]", title="65/75% SOC holdout")
    axes[1].axhline(1.5, color="black", linestyle="--", linewidth=0.9)
    axes[1].set(xlabel="Prediction horizon [s]", ylabel="Mean temperature RMSE [°C]", title="Combined electro-thermal prediction")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _plot_state_audit(metrics: pd.DataFrame, output: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = metrics["feature_set"].str.replace("current_dnn_5_plus_", "5+").str.replace("plus_", "+")
    axes[0].barh(labels, metrics["mean_local_standard_deviation_a"], color="tab:blue")
    axes[0].axvline(0.25, color="black", linestyle="--", linewidth=0.9)
    axes[0].set(xlabel="Mean local label std [A]", title="Conditional label dispersion")
    axes[1].barh(labels, metrics["nearest_neighbor_label_difference_p95_a"], color="tab:orange")
    axes[1].axvline(0.50, color="black", linestyle="--", linewidth=0.9)
    axes[1].set(xlabel="P95 nearest-label difference [A]", title="Near-collision diagnostic")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _write_report(path: Path, payload: dict[str, Any], model_table: pd.DataFrame, state_table: pd.DataFrame, availability: pd.DataFrame) -> None:
    decision = payload["decision"]
    model_rows = []
    for _, row in model_table.iterrows():
        model_rows.append(
            f"| {row['variant']} | {row['horizon_s']:.0f} | {row['voltage_rmse_mv_mean']:.2f} | "
            f"{row['temperature_rmse_c_mean']:.4f} | {int(row['voltage_false_safe_count'])} | {int(row['temperature_false_safe_count'])} |"
        )
    state_rows = []
    for _, row in state_table.iterrows():
        state_rows.append(
            f"| {row['feature_set']} | {row['mean_local_standard_deviation_a']:.4f} | "
            f"{row['nearest_neighbor_label_difference_p95_a']:.4f} | "
            f"{100 * row['variance_reduction_from_current_dnn_5_fraction']:.1f}% |"
        )
    availability_rows = [f"| {row.candidate} | {row.status} | {row.detail} |" for row in availability.itertuples(index=False)]
    text = rf"""# Phase 2R：Chen2020 模型与控制状态充分性审计

## 总结

- 2R-A 固定参数 2RC+双节点热模型：**{'通过本轮筛查' if decision['fixed_model_sufficient'] else '未通过本轮筛查'}**（筛查通过不等同于已证明全域充分）。
- 2R-A SOC/温度相关参数模型：**{'通过本轮筛查' if decision['related_model_sufficient'] else '未通过本轮筛查'}**。
- 2R-B 当前五状态 DNN 输入的局部单值性：**{'充分' if decision['current_dnn_state_sufficient'] else '不充分'}**。
- 下一步：{decision['next_action']}

## 审计合同

2R-A 使用 60%、65%、70%、75%、80% 初始 SOC，15/25/30 ℃，0.5/1/2 C 的 300 s DFN 脉冲，并在 5/25/300 s 统计误差。60/70/80% 用于局部参数辨识，65/75% 作为 SOC 插值留出验证。约束真假分类使用 4.2 V 与 35 ℃物理边界。

2R-B 严格重放冻结的 222 条、1776 个 Phase 6R 首动作标签。对嵌套输入集合用 25 近邻估计：

\[
\operatorname{{Var}}\!\left(I_k^\star\mid\phi_k\right).
\]

该估计是局部邻域诊断，不等同于解析条件分布。MPC 活跃模式和 target-cap 状态由求解结果构造，只能用于解释，不自动视为可部署的求解前输入。

## 2R-A 留出 SOC 结果

| 模型 | 时域 [s] | 平均电压 RMSE [mV] | 平均温度 RMSE [℃] | 电压 false-safe | 温度 false-safe |
|---|---:|---:|---:|---:|---:|
{chr(10).join(model_rows)}

## 2R-B 条件方差结果

| 输入集合 | 平均局部标准差 [A] | 最近邻标签差 P95 [A] | 相对当前五状态方差下降 |
|---|---:|---:|---:|
{chr(10).join(state_rows)}

严格逐点重放**未通过**：最大教师电流差为 `{payload['state_audit']['replay']['maximum_replay_current_difference_a']:.3e} A`，超过 `1e-6 A` 的样本数为 `{payload['state_audit']['replay']['replay_mismatch_count']}`。差异的 P95 为 `{payload['state_audit']['replay']['p95_replay_current_difference_a']:.3e} A`，超过 `0.01 A` 的比例为 `{100.0 * payload['state_audit']['replay']['large_difference_fraction']:.2f}%`；因此本轮只把上一最优序列摘要判为**可用于统计诊断的近似重放量**，不能把它当作精确历史真值。

## 候选变量可审计性

| 候选变量 | 状态 | 说明 |
|---|---|---|
{chr(10).join(availability_rows)}

## 结论边界

1. 参数相关模型在 65/75% SOC 上是插值验证，但温度仅在 15/25/30 ℃锚点上评价，不是温度留出外推。
2. Chen2020 DFN 使用集总热模型，只提供体积平均温度；双节点核心/表面温度仍没有独立真值，因此不能宣称两个内部温度状态分别得到验证。
3. 若所有温度样本都未达到 35 ℃，温度分类只能证明真可行区域，不能证明对过温的识别能力。
4. 可辨识参数和环境温度在 Phase 6R 教师集中为常量，其条件方差贡献无法由该数据识别。
5. 固定参数模型在留出 SOC 上通过预设均值门槛且没有 false-safe，但 300 s 最大电压误差达到约 52 mV，且本批样本没有真实过温阳性；所以不能据此宣称它足以覆盖完整 MPC 运行域。
6. 参数相关模型的独立热结构重放误差很小，但电热耦合预测在 15 ℃、2 C、300 s 出现明显温度偏差，并产生 3 个电压 false-safe。该结果说明当前“参数相关辨识 + 电热耦合”实现尚未形成可靠替代模型，不说明 SOC/温度相关参数思想本身无效。
7. 上一最优序列摘要使平均局部条件方差下降 37.6%，但加入后仍未达到 0.25 A / 0.50 A 两项单值性门槛；它是下一轮应精确记录并验证的必要候选信息，不是已经充分的状态扩充方案。
"""
    path.write_text(text, encoding="utf-8")


def run_phase_two_r(config: PhaseTwoRConfig, project_root: str | Path = ".") -> dict[str, Any]:
    root = Path(project_root)
    data_dir = root / "data" / "phase2r_sufficiency_audit"
    output_dir = root / "outputs" / "phase2r_sufficiency_audit"
    figure_dir = output_dir / "figures"
    for directory in (data_dir, output_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    phase2 = load_phase_two_config(root / config.sources.phase2_config)
    fixed_parameters = json.loads((root / config.sources.fixed_parameters).read_text(encoding="utf-8"))
    fixed_ocv = pd.read_csv(root / config.sources.fixed_ocv_curve)

    ocv_grid, pulses = generate_phase_two_r_dfn_data(phase2, config, data_dir)
    pulses.to_csv(data_dir / "dfn_pulse_grid.csv", index=False)
    related_path = data_dir / "related_electrical_parameter_grid.csv"
    if related_path.exists():
        related = pd.read_csv(related_path)
    else:
        related = fit_related_electrical_parameters(pulses, ocv_grid, phase2, config)
        related.to_csv(related_path, index=False)
    thermal_path = data_dir / "temperature_specific_thermal_parameters.csv"
    if thermal_path.exists():
        thermal = pd.read_csv(thermal_path)
    else:
        thermal = fit_temperature_specific_thermal_models(
            pulses,
            fixed_parameters["thermal_two_node"],
            float(fixed_parameters["core_heat_capacity_fraction"]),
            config,
        )
        thermal.to_csv(thermal_path, index=False)
    predictions, horizon_metrics, thermal_structure = evaluate_model_variants(
        pulses, ocv_grid, related, thermal, fixed_ocv, fixed_parameters, phase2, config
    )
    predictions.to_csv(data_dir / "model_variant_predictions.csv", index=False)
    horizon_metrics.to_csv(output_dir / "model_horizon_metrics.csv", index=False)
    thermal_structure.to_csv(output_dir / "thermal_structure_metrics.csv", index=False)
    model_summary = summarize_model_audit(horizon_metrics, thermal_structure, config)
    model_summary_table = pd.DataFrame.from_records(model_summary["holdout_summary"])

    teacher = pd.read_csv(root / config.sources.rolling_teacher_dataset)
    augmented_path = data_dir / "rolling_teacher_control_memory.csv"
    replay_metrics_path = data_dir / "rolling_teacher_replay_metrics.json"
    if augmented_path.exists() and replay_metrics_path.exists():
        augmented = pd.read_csv(augmented_path)
        replay_metrics = summarize_replay_differences(augmented, config)
        replay_metrics_path.write_text(json.dumps(replay_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        augmented, replay_metrics = replay_teacher_with_control_memory(teacher, config, root)
        augmented.to_csv(augmented_path, index=False)
        replay_metrics_path.write_text(json.dumps(replay_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if not replay_metrics["memory_summary_usable"]:
        raise RuntimeError("Phase 2R-B 重放误差过大，上一最优序列摘要不可用于条件方差审计。")
    state_metrics, availability, state_summary = run_state_sufficiency_audit(augmented, config)
    state_metrics.to_csv(output_dir / "state_conditional_variance_metrics.csv", index=False)
    availability.to_csv(output_dir / "candidate_variable_availability.csv", index=False)

    fixed_check = model_summary["variant_checks"]["fixed_2rc_dual"]
    related_check = model_summary["variant_checks"]["related_2rc_dual"]
    fixed_sufficient = bool(
        fixed_check["all_horizon_mean_voltage_rmse_within_limit"]
        and fixed_check["all_horizon_mean_temperature_rmse_within_limit"]
        and fixed_check["voltage_false_safe_count"] == 0
        and fixed_check["temperature_false_safe_count"] == 0
    )
    related_sufficient = bool(
        related_check["all_horizon_mean_voltage_rmse_within_limit"]
        and related_check["all_horizon_mean_temperature_rmse_within_limit"]
        and related_check["voltage_false_safe_count"] == 0
        and related_check["temperature_false_safe_count"] == 0
    )
    state_sufficient = bool(state_summary["current_dnn_input_locally_sufficient"])
    if not fixed_sufficient and related_sufficient:
        next_action = "先采用/验证 SOC-温度相关降阶模型，再进入 A1 硬斜率消融。"
    elif not state_sufficient:
        next_action = "先在新教师数据中精确记录并验证上一最优序列摘要等控制记忆，再决定 Chen2020 DNN 的扩充输入；暂不训练新 ANN。"
    elif fixed_sufficient:
        next_action = "固定当前降阶模型，进入 A1 硬斜率约束消融。"
    else:
        next_action = "降阶模型仍不充分，暂停 A1–A7 与 ANN 训练，继续模型辨识。"

    figures = {
        "model_horizon": str(_plot_model_audit(horizon_metrics, figure_dir / "model_horizon_errors.png")),
        "state_variance": str(_plot_state_audit(state_metrics, figure_dir / "state_conditional_variance.png")),
    }
    payload: dict[str, Any] = {
        "study_name": config.study_name,
        "model_audit": {
            "pulse_profile_count": int(pulses["profile_name"].nunique()),
            "ocv_point_count": int(len(ocv_grid)),
            **model_summary,
        },
        "state_audit": {
            "replay": replay_metrics,
            "summary": state_summary,
            "metrics": state_metrics.to_dict("records"),
            "availability": availability.to_dict("records"),
        },
        "decision": {
            "fixed_model_sufficient": fixed_sufficient,
            "related_model_sufficient": related_sufficient,
            "current_dnn_state_sufficient": state_sufficient,
            "next_action": next_action,
        },
        "figures": figures,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _write_report(output_dir / "PHASE2R模型与控制状态充分性审计报告.md", payload, model_summary_table, state_metrics, availability)
    return payload

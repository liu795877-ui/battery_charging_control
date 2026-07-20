"""组织第三阶段 B 的公平基线、教师数据生成、质量审计和结果落盘。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .filtered_baseline import (
    filtered_baseline_metrics,
    simulate_filtered_baseline_dfn,
    simulate_filtered_baseline_reduced,
)
from .identification import build_ocv_function
from .mpc import ReducedBatteryModel
from .phase3_config import load_phase_three_config
from .phase3b_config import PhaseThreeBConfig
from .phase3b_plotting import (
    plot_fair_baseline_comparison,
    plot_teacher_dataset_coverage,
)
from .teacher_data import (
    dataset_quality_metrics,
    generate_reachable_rollout,
    label_teacher_states,
    sample_reachable_states,
)


def _load_model(
    config: PhaseThreeBConfig, project_root: Path
) -> tuple[Any, ReducedBatteryModel]:
    """读取第三阶段A配置及其已经验证的第二阶段模型产物。"""
    phase3 = load_phase_three_config(project_root / config.source_phase3_config)
    parameter_path = project_root / phase3.artifacts.identified_parameters
    validation_path = project_root / phase3.artifacts.validation_metrics
    ocv_path = project_root / phase3.artifacts.ocv_curve
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation.get("success", False):
        raise RuntimeError("第二阶段验证闸门未通过，禁止生成MPC教师数据。")
    parameters = json.loads(parameter_path.read_text(encoding="utf-8"))
    ocv = pd.read_csv(ocv_path)
    return phase3, ReducedBatteryModel(
        phase3, build_ocv_function(ocv), parameters
    )


def _comparison_frame(
    baseline_metrics: dict[str, Any], mpc_metrics: dict[str, Any]
) -> pd.DataFrame:
    """只比较在同一个DFN和同一物理约束下运行的两种控制方法。"""
    return pd.DataFrame.from_records(
        [
            {
                "controller": "Filtered 1C baseline",
                "charge_time_min": baseline_metrics["charge_time_min"],
                "maximum_voltage_v": baseline_metrics["maximum_voltage_v"],
                "maximum_temperature_c": baseline_metrics[
                    "maximum_temperature_c"
                ],
                "maximum_current_a": baseline_metrics["maximum_current_a"],
                "success": baseline_metrics["success"],
            },
            {
                "controller": "Constrained MPC",
                "charge_time_min": mpc_metrics["charge_time_min"],
                "maximum_voltage_v": mpc_metrics["maximum_voltage_v"],
                "maximum_temperature_c": mpc_metrics[
                    "maximum_temperature_c"
                ],
                "maximum_current_a": mpc_metrics["maximum_current_a"],
                "success": mpc_metrics["success"],
            },
        ]
    )


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    """不依赖tabulate地生成简单Markdown表格。"""
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for _, row in frame.iterrows():
        values = []
        for header in headers:
            value = row[header]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_report(
    path: Path,
    quality: dict[str, Any],
    comparison: pd.DataFrame,
    baseline_metrics: dict[str, Any],
    overall_success: bool,
) -> None:
    """写出面向后续DNN训练决策的中文报告。"""
    active = quality["active_constraint_counts"]
    baseline_time = float(comparison.iloc[0]["charge_time_min"])
    mpc_time = float(comparison.iloc[1]["charge_time_min"])
    time_statement = (
        f"本次MPC比公平基线快{baseline_time - mpc_time:.2f} min。"
        if mpc_time < baseline_time
        else f"本次MPC比公平基线慢{mpc_time - baseline_time:.2f} min，"
        "不能声称MPC已经缩短充电时间。"
    )
    lines = [
        "# 第三阶段 B 报告：MPC教师数据生成与质量审计",
        "",
        "## 结论",
        "",
        f"第三阶段B第一版验收：{'通过' if overall_success else '未通过'}。",
        "",
        "## 数据协议",
        "",
        "- 所有状态来自从10% SOC、25 ℃、零极化初态逐步推进的受约束轨迹；",
        "- 每条轨迹在每个10% SOC分箱内分层取样；",
        "- 数据按整条轨迹划分为训练、验证和测试集；",
        "- 优化失败、预测不可行或使用回退的样本只保留在审计表，不进入训练标签。",
        "",
        "## 教师数据质量",
        "",
        f"- 候选状态：{quality['candidate_count']}；接受标签：{quality['accepted_count']}；拒绝：{quality['rejected_count']}；",
        f"- 教师接受率：{100 * quality['teacher_acceptance_fraction']:.2f}%；",
        f"- 数据划分：{quality['split_counts']}；轨迹划分：{quality['trajectory_split_counts']}；",
        f"- SOC分箱计数：{quality['soc_bin_counts']}；",
        f"- 活跃约束计数：电压{active['voltage']}、温度{active['temperature']}、电流{active['current_upper']}、电流变化率{active['current_change']}；",
        f"- 平均教师求解时间：{quality['mean_teacher_solve_time_ms']:.1f} ms；最大：{quality['maximum_teacher_solve_time_ms']:.1f} ms。",
        "",
        "## 同约束DFN基线",
        "",
        *_markdown_table(comparison),
        "",
        f"公平基线安全回退次数：{baseline_metrics['safety_override_count']}。",
        time_statement,
        "",
        "## 进入DNN训练前仍需遵守",
        "",
        "- 第一版DNN输入只使用SOC、两支极化电压、平均温度和上一时刻电流；",
        "- 核心与表面温度仅作为教师内部审计状态，不能作为已验证物理测量量；",
        "- 两支极化电压也是模型内部状态；未来接入BMS前必须增加状态观测器并单独验证；",
        "- 训练后必须在未见过的整条轨迹及Chen2020 DFN上做闭环约束验证；",
        "- 电流回归误差小不等于闭环安全，必须报告约束违反率、充电时间差和终端成功率。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase_three_b(
    config: PhaseThreeBConfig, project_root: str | Path
) -> dict[str, Any]:
    """执行第三阶段B全流程并返回数据质量与公平基线结果。"""
    project_root = Path(project_root)
    data_dir = project_root / "data" / "phase3b"
    metrics_dir = project_root / "outputs" / "metrics"
    figures_dir = project_root / "outputs" / "figures"
    for directory in (data_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    phase3, model = _load_model(config, project_root)

    rollout_frames = [
        generate_reachable_rollout(model, phase3, rollout)
        for rollout in config.dataset.rollouts
    ]
    rollouts = pd.concat(rollout_frames, ignore_index=True)
    rollouts.to_csv(data_dir / "reachable_rollouts.csv", index=False)
    candidates = sample_reachable_states(rollouts, config)
    attempts, accepted = label_teacher_states(
        candidates, model, phase3, config
    )
    attempts.to_csv(data_dir / "teacher_label_attempts.csv", index=False)
    accepted.to_csv(data_dir / "teacher_dataset.csv", index=False)
    quality = dataset_quality_metrics(attempts, accepted, config)

    baseline_reduced = simulate_filtered_baseline_reduced(
        model,
        phase3,
        config.fair_baseline.desired_current_a,
        config.fair_baseline.maximum_simulation_time_s,
    )
    baseline_reduced.to_csv(
        data_dir / "filtered_1c_reduced_closed_loop.csv", index=False
    )
    baseline_dfn = simulate_filtered_baseline_dfn(
        model,
        phase3,
        config.fair_baseline.desired_current_a,
        config.fair_baseline.maximum_simulation_time_s,
    )
    baseline_dfn.to_csv(
        data_dir / "filtered_1c_dfn_closed_loop.csv", index=False
    )
    baseline_metrics = filtered_baseline_metrics(baseline_dfn, phase3)

    phase3_metrics_path = project_root / "outputs" / "metrics" / "phase3_mpc_metrics.json"
    phase3_metrics = json.loads(phase3_metrics_path.read_text(encoding="utf-8"))
    mpc_metrics = phase3_metrics["dfn_closed_loop"]
    comparison = _comparison_frame(baseline_metrics, mpc_metrics)
    comparison.to_csv(
        metrics_dir / "phase3b_same_constraint_comparison.csv", index=False
    )

    dataset_figure = plot_teacher_dataset_coverage(
        accepted, figures_dir / "phase3b_teacher_dataset_coverage.png"
    )
    mpc_dfn = pd.read_csv(
        project_root / "data" / "phase3" / "mpc_chen2020_dfn_closed_loop.csv"
    )
    baseline_figure = plot_fair_baseline_comparison(
        baseline_dfn,
        mpc_dfn,
        phase3,
        figures_dir / "phase3b_same_constraint_comparison.png",
    )
    # 数据进入DNN训练的前提不仅是标签质量和公平基线成功，原始MPC教师本身
    # 也必须已在DFN上通过闭环约束验证。
    overall_success = bool(
        quality["success"]
        and baseline_metrics["success"]
        and mpc_metrics["success"]
    )
    payload = {
        "dataset_quality": quality,
        "fair_baseline": baseline_metrics,
        "mpc_reference": mpc_metrics,
        "success": overall_success,
        "ready_for_dnn_training": overall_success,
    }
    metrics_path = metrics_dir / "phase3b_metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = project_root / "outputs" / "phase3b_report.md"
    _write_report(
        report_path, quality, comparison, baseline_metrics, overall_success
    )
    return {
        "metrics": payload,
        "comparison": comparison,
        "outputs": {
            "dataset": str(data_dir / "teacher_dataset.csv"),
            "attempts": str(data_dir / "teacher_label_attempts.csv"),
            "metrics": str(metrics_path),
            "dataset_figure": str(dataset_figure),
            "baseline_figure": str(baseline_figure),
            "report": str(report_path),
        },
    }

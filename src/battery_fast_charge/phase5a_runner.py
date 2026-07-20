"""组织阶段5A降阶压力测试、DFN温度锚点和准入判断。"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ann_model import TinyANN
from .identification import build_ocv_function
from .phase3_config import load_phase_three_config
from .phase4_config import load_phase_four_a_config
from .phase5a_config import PhaseFiveAConfig
from .phase5a_plotting import (
    plot_dfn_temperature_anchors,
    plot_reduced_stress_summary,
)
from .robustness import run_dfn_temperature_anchors, run_reduced_stress_test


def _markdown_table(frame: pd.DataFrame) -> str:
    """生成不依赖tabulate的小型Markdown表格。"""
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
    return "\n".join(lines)


def _reduced_metrics(
    summary: pd.DataFrame, config: PhaseFiveAConfig
) -> dict[str, Any]:
    """汇总完成率、物理安全率和最坏工况。"""
    criteria = config.success_criteria
    completion = float(summary["completion_success"].mean())
    safety = float(summary["physical_safe"].mean())
    maximum_time = float(summary["charge_time_min"].max())
    maximum_material = float(summary["material_intervention_fraction"].max())
    checks = {
        "completion_fraction": completion
        >= criteria.minimum_reduced_completion_fraction,
        "physical_safety_fraction": safety
        >= criteria.minimum_reduced_physical_safety_fraction,
        "maximum_charge_time": maximum_time
        <= criteria.maximum_reduced_charge_time_min,
        "maximum_material_intervention": maximum_material
        <= criteria.maximum_reduced_material_intervention_fraction,
    }

    def worst(column: str, ascending: bool = False) -> dict[str, Any]:
        row = summary.sort_values(column, ascending=ascending).iloc[0]
        return {
            "scenario_id": str(row["scenario_id"]),
            "value": float(row[column]),
        }

    return {
        "scenario_count": int(len(summary)),
        "completion_count": int(summary["completion_success"].sum()),
        "completion_fraction": completion,
        "physical_safety_count": int(summary["physical_safe"].sum()),
        "physical_safety_fraction": safety,
        "maximum_charge_time_min": maximum_time,
        "maximum_material_intervention_fraction": maximum_material,
        "worst_cases": {
            "charge_time": worst("charge_time_min"),
            "voltage": worst("maximum_voltage_v"),
            "temperature": worst("maximum_temperature_c"),
            "material_intervention": worst("material_intervention_fraction"),
            "terminal_soc_undercharge": worst(
                "terminal_true_soc_error", ascending=True
            ),
            "current_change": worst("maximum_current_change_a"),
        },
        "failure_scenarios": summary.loc[
            ~(summary["completion_success"] & summary["physical_safe"]),
            [
                "scenario_id",
                "completion_success",
                "physical_safe",
                "terminal_true_soc_error",
                "maximum_voltage_v",
                "maximum_temperature_c",
                "maximum_current_change_a",
            ],
        ].to_dict("records"),
        "checks": checks,
        "success": bool(all(checks.values())),
    }


def _dfn_metrics(summary: pd.DataFrame, config: PhaseFiveAConfig) -> dict[str, Any]:
    """检查三个DFN锚点是否全部完成且无物理越界。"""
    criteria = config.success_criteria
    physical_safe = ~(
        summary["voltage_limit_exceeded"]
        | summary["temperature_limit_exceeded"]
        | summary["current_limit_exceeded"]
        | summary["current_change_limit_exceeded"]
    )
    all_complete = bool(summary["reached_target_soc"].all())
    all_safe = bool(physical_safe.all())
    maximum_time = float(summary["charge_time_min"].fillna(np.inf).max())
    maximum_material = float(
        summary["material_safety_filter_intervention_fraction"].max()
    )
    checks = {
        "all_anchors_complete": (
            all_complete if criteria.require_all_dfn_anchors_complete else True
        ),
        "all_anchors_physically_safe": (
            all_safe if criteria.require_all_dfn_anchors_physically_safe else True
        ),
        "maximum_charge_time": maximum_time
        <= criteria.maximum_dfn_anchor_charge_time_min,
        "maximum_material_intervention": maximum_material
        <= criteria.maximum_dfn_material_intervention_fraction,
    }
    return {
        "anchor_count": int(len(summary)),
        "all_anchors_complete": all_complete,
        "all_anchors_physically_safe": all_safe,
        "maximum_charge_time_min": maximum_time,
        "maximum_material_intervention_fraction": maximum_material,
        "checks": checks,
        "success": bool(all(checks.values())),
    }


def _write_report(
    path: Path,
    payload: dict[str, Any],
    reduced_summary: pd.DataFrame,
    dfn_summary: pd.DataFrame,
) -> None:
    """写出通过/失败均可审计的阶段5A中文报告。"""
    reduced_columns = [
        "scenario_id",
        "completion_success",
        "physical_safe",
        "charge_time_min",
        "maximum_voltage_v",
        "maximum_temperature_c",
        "terminal_true_soc_error",
    ]
    dfn_columns = [
        "anchor_temperature_c",
        "reached_target_soc",
        "charge_time_min",
        "maximum_voltage_v",
        "maximum_temperature_c",
        "material_safety_filter_intervention_fraction",
        "success",
    ]
    worst_ids = {
        item["scenario_id"]
        for item in payload["reduced_stress_test"]["worst_cases"].values()
    }
    worst_table = reduced_summary[
        reduced_summary["scenario_id"].isin(worst_ids)
    ][reduced_columns]
    lines = [
        "# 第五阶段 A 报告：有界鲁棒性验证",
        "",
        "## 结论",
        "",
        f"阶段5A验收：{'通过' if payload['success'] else '未通过'}。",
        "",
        "## 方法边界",
        "",
        "- 降阶层使用名义ANN和名义安全过滤器控制参数扰动对象，并注入有偏、相关的状态估计误差；",
        "- 场景是有界压力样本，不代表真实制造概率分布，也不能换算失效率；",
        "- DFN层只复核15、25、30 ℃三个温度锚点。",
        "",
        "## 降阶压力测试",
        "",
        f"- 场景数：{payload['reduced_stress_test']['scenario_count']}；完成率：{100 * payload['reduced_stress_test']['completion_fraction']:.2f}%；物理安全率：{100 * payload['reduced_stress_test']['physical_safety_fraction']:.2f}%；",
        f"- 最长时间：{payload['reduced_stress_test']['maximum_charge_time_min']:.2f} min；最坏实质介入：{100 * payload['reduced_stress_test']['maximum_material_intervention_fraction']:.2f}%；",
        "",
        _markdown_table(worst_table),
        "",
        "## Chen2020 DFN温度锚点",
        "",
        _markdown_table(dfn_summary[dfn_columns]),
        "",
        "## 下一步",
        "",
        "- 若本阶段未通过，应针对明确失败模式调整训练域、安全余量或状态估计方案，禁止直接进入BMS接口；",
        "- 即使通过，仍需析锂/老化约束、观测器实现和HIL验证。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase_five_a(
    config: PhaseFiveAConfig, project_root: str | Path
) -> dict[str, Any]:
    """运行降阶广覆盖压力测试和DFN温度锚点。"""
    project_root = Path(project_root)
    data_dir = project_root / "data" / "phase5a"
    metrics_dir = project_root / "outputs" / "metrics"
    figures_dir = project_root / "outputs" / "figures"
    for directory in (data_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    phase3 = load_phase_three_config(project_root / config.source_phase3_config)
    phase4 = load_phase_four_a_config(project_root / config.source_phase4a_config)
    prior = json.loads(
        (project_root / config.phase4b2_metrics).read_text(encoding="utf-8")
    )
    if not prior.get("ready_for_robustness_validation", False):
        raise RuntimeError("阶段4B-2未通过，禁止进入鲁棒性验证。")
    parameters = json.loads(
        (project_root / phase3.artifacts.identified_parameters).read_text(
            encoding="utf-8"
        )
    )
    ocv_frame = pd.read_csv(project_root / phase3.artifacts.ocv_curve)
    ocv_function = build_ocv_function(ocv_frame)
    ann = TinyANN.load(project_root / config.ann_model)

    reduced_summary, reduced_worst = run_reduced_stress_test(
        ann, parameters, ocv_function, phase3, phase4, config
    )
    reduced_summary.to_csv(data_dir / "reduced_stress_summary.csv", index=False)
    reduced_worst.to_csv(
        data_dir / "reduced_worst_case_trajectories.csv", index=False
    )
    dfn_summary, dfn_trajectories = run_dfn_temperature_anchors(
        ann, parameters, ocv_function, phase3, phase4, config
    )
    dfn_summary.to_csv(data_dir / "dfn_temperature_anchor_summary.csv", index=False)
    dfn_trajectories.to_csv(
        data_dir / "dfn_temperature_anchor_trajectories.csv", index=False
    )

    reduced_metrics = _reduced_metrics(reduced_summary, config)
    dfn_metrics = _dfn_metrics(dfn_summary, config)
    checks = {
        "reduced_stress_test": reduced_metrics["success"],
        "dfn_temperature_anchors": dfn_metrics["success"],
    }
    payload = {
        "configuration": asdict(config),
        "reduced_stress_test": reduced_metrics,
        "dfn_temperature_anchors": dfn_metrics,
        "checks": checks,
        "ready_for_observer_validation": bool(all(checks.values())),
        "success": bool(all(checks.values())),
    }
    metrics_path = metrics_dir / "phase5a_metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_reduced_stress_summary(
        reduced_summary, figures_dir / "phase5a_reduced_stress_summary.png"
    )
    plot_dfn_temperature_anchors(
        dfn_trajectories, figures_dir / "phase5a_dfn_temperature_anchors.png"
    )
    report_path = project_root / "outputs" / "phase5a_report.md"
    _write_report(report_path, payload, reduced_summary, dfn_summary)
    return {
        "metrics": payload,
        "reduced_summary": reduced_summary,
        "dfn_summary": dfn_summary,
    }

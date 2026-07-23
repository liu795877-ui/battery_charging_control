"""组织第三阶段 MPC 教师的仿真、指标、绘图和报告输出。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .closed_loop import (
    closed_loop_metrics,
    simulate_dfn_closed_loop,
    simulate_reduced_closed_loop,
)
from .identification import build_ocv_function
from .mpc import ReducedBatteryModel
from .phase3_config import PhaseThreeConfig
from .phase3_plotting import plot_phase_three_closed_loop


def _load_phase_two_artifacts(
    config: PhaseThreeConfig, project_root: Path
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """读取第二阶段已经固定的参数、OCV和验证结果。"""
    parameter_path = project_root / config.artifacts.identified_parameters
    metrics_path = project_root / config.artifacts.validation_metrics
    ocv_path = project_root / config.artifacts.ocv_curve
    missing = [path for path in (parameter_path, metrics_path, ocv_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少第二阶段输出：{missing}")
    parameters = json.loads(parameter_path.read_text(encoding="utf-8"))
    phase_two_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not phase_two_metrics.get("success", False):
        raise RuntimeError("第二阶段验证未通过，不能把该降阶模型用于 MPC。")
    return parameters, pd.read_csv(ocv_path), phase_two_metrics


def _comparison_table(
    reduced_metrics: dict[str, Any],
    dfn_metrics: dict[str, Any],
    project_root: Path,
) -> pd.DataFrame:
    """汇总现有 CC–CV 和本阶段两条 MPC 轨迹。"""
    records: list[dict[str, Any]] = []
    baseline_path = project_root / "outputs" / "metrics" / "baseline_summary.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        for _, row in baseline.iterrows():
            records.append(
                {
                    "controller": f"CC-CV {row['c_rate']:g}C",
                    "plant": "Chen2020 DFN",
                    "reached_target_soc": bool(row["reached_target_soc"]),
                    "charge_time_min": row["charge_time_min"],
                    "maximum_voltage_v": row["maximum_voltage_v"],
                    "maximum_temperature_c": row["maximum_temperature_c"],
                }
            )
    for label, metrics in (
        ("Constrained MPC", reduced_metrics),
        ("Constrained MPC", dfn_metrics),
    ):
        records.append(
            {
                "controller": label,
                "plant": metrics["source"],
                "reached_target_soc": metrics["reached_target_soc"],
                "charge_time_min": metrics["charge_time_min"],
                "maximum_voltage_v": metrics["maximum_voltage_v"],
                "maximum_temperature_c": metrics["maximum_temperature_c"],
            }
        )
    return pd.DataFrame.from_records(records)


def _write_report(
    path: Path,
    config: PhaseThreeConfig,
    reduced_metrics: dict[str, Any],
    dfn_metrics: dict[str, Any],
    comparison: pd.DataFrame,
) -> None:
    """生成可以脱离 notebook 阅读的中文阶段报告。"""
    # 不调用 DataFrame.to_markdown，避免仅为排版额外依赖 tabulate。
    comparison_headers = list(comparison.columns)
    comparison_markdown = [
        "| " + " | ".join(comparison_headers) + " |",
        "|" + "|".join("---" for _ in comparison_headers) + "|",
    ]
    for _, row in comparison.iterrows():
        values = []
        for column in comparison_headers:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        comparison_markdown.append("| " + " | ".join(values) + " |")

    lines = [
        "# 第三阶段 A 报告：约束 MPC 教师闭环验证",
        "",
        "## 本阶段问题",
        "",
        "在 25 ℃、10%–80% SOC 范围内，MPC 能否在电流、电压和平均温度约束下生成可行充电轨迹，并在独立 Chen2020 DFN 虚拟电池上保持约束？",
        "",
        "## 控制器边界",
        "",
        f"- 物理电压上限：{config.constraints.physical_maximum_voltage_v:.2f} V；MPC 内部上限：{config.constraints.mpc_maximum_voltage_v:.2f} V。",
        f"- 物理平均温度上限：{config.constraints.physical_maximum_temperature_c:.1f} ℃；MPC 内部上限：{config.constraints.mpc_maximum_temperature_c:.1f} ℃。",
        f"- 电流范围：0–{config.constraints.maximum_current_a:.1f} A；正常每 {config.control.control_interval_s:g} s 最大变化 {config.constraints.maximum_current_change_a_per_step:.1f} A。",
        "- 温度约束只针对已经验证的平均温度，不能解释为核心温度安全保证。",
        "",
        "## 闭环结果",
        "",
        "| 仿真对象 | 到达80% | 时间/min | 最高电压/V | 最高平均温度/℃ | 优化成功率 | 回退次数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| 降阶模型 | {reduced_metrics['reached_target_soc']} | "
            f"{reduced_metrics['charge_time_min'] or float('nan'):.2f} | "
            f"{reduced_metrics['maximum_voltage_v']:.4f} | "
            f"{reduced_metrics['maximum_temperature_c']:.3f} | "
            f"{100 * reduced_metrics['optimizer_success_fraction']:.1f}% | "
            f"{reduced_metrics['fallback_count']} |"
        ),
        (
            f"| Chen2020 DFN | {dfn_metrics['reached_target_soc']} | "
            f"{dfn_metrics['charge_time_min'] or float('nan'):.2f} | "
            f"{dfn_metrics['maximum_voltage_v']:.4f} | "
            f"{dfn_metrics['maximum_temperature_c']:.3f} | "
            f"{100 * dfn_metrics['optimizer_success_fraction']:.1f}% | "
            f"{dfn_metrics['fallback_count']} |"
        ),
        "",
        f"**第一版验收：{'通过' if dfn_metrics['success'] else '未通过'}。**",
        "",
        "## 与 CC–CV 的关系",
        "",
        *comparison_markdown,
        "",
        "## 解释限制",
        "",
        "- MPC 的最优性只相对于当前降阶模型、300 s 预测范围和 SOC 缺口代价函数成立。",
        "- 当前验证仍是虚拟电池仿真，不等同于实物安全验证。",
        "- 只有教师 MPC 在 DFN 上通过后，才应批量生成 DNN 监督学习数据。",
        "- 后续数据生成必须保存求解状态、可行性和约束活跃信息，不能把失败解当作最优标签。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase_three(config: PhaseThreeConfig, project_root: str | Path) -> dict[str, Any]:
    """运行降阶闭环和 DFN 闭环，并把全部可复现产物落盘。"""
    project_root = Path(project_root)
    data_dir = project_root / "data" / "phase3"
    metrics_dir = project_root / "outputs" / "metrics"
    figures_dir = project_root / "outputs" / "figures"
    for directory in (data_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    parameters, ocv_table, phase_two_metrics = _load_phase_two_artifacts(
        config, project_root
    )
    model = ReducedBatteryModel(
        config, build_ocv_function(ocv_table), parameters
    )
    reduced = simulate_reduced_closed_loop(model, config)
    reduced.to_csv(data_dir / "mpc_reduced_closed_loop.csv", index=False)
    reduced_metrics = closed_loop_metrics(reduced, config)

    dfn = simulate_dfn_closed_loop(model, config)
    dfn.to_csv(data_dir / "mpc_chen2020_dfn_closed_loop.csv", index=False)
    dfn_metrics = closed_loop_metrics(dfn, config)

    comparison = _comparison_table(reduced_metrics, dfn_metrics, project_root)
    comparison.to_csv(metrics_dir / "phase3_controller_comparison.csv", index=False)
    payload = {
        "phase2_gate": {
            "success": phase_two_metrics["success"],
            "scope_assessment": phase_two_metrics["scope_assessment"],
        },
        "reduced_closed_loop": reduced_metrics,
        "dfn_closed_loop": dfn_metrics,
        "success": dfn_metrics["success"],
    }
    metrics_path = metrics_dir / "phase3_mpc_metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    figure_path = plot_phase_three_closed_loop(
        reduced, dfn, config, figures_dir / "phase3_mpc_closed_loop.png"
    )
    report_path = project_root / "outputs" / "phase3_report.md"
    _write_report(report_path, config, reduced_metrics, dfn_metrics, comparison)
    return {
        "metrics": payload,
        "comparison": comparison,
        "outputs": {
            "metrics": str(metrics_path),
            "figure": str(figure_path),
            "report": str(report_path),
        },
    }

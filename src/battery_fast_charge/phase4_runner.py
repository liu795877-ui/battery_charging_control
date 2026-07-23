"""组织阶段4A训练、模型导出、闭环验证和结果落盘。"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ann_closed_loop import (
    ann_closed_loop_metrics,
    simulate_ann_dfn_closed_loop,
    simulate_ann_reduced_closed_loop,
)
from .ann_model import TinyANN, train_tiny_ann
from .identification import build_ocv_function
from .mpc import ReducedBatteryModel
from .phase3_config import load_phase_three_config
from .phase4_config import PhaseFourAConfig
from .phase4_plotting import plot_ann_closed_loop, plot_offline_imitation


def _load_reduced_model(
    config: PhaseFourAConfig, project_root: Path
) -> tuple[Any, ReducedBatteryModel]:
    """读取已通过第二阶段闸门的模型和第三阶段约束。"""
    phase3 = load_phase_three_config(project_root / config.source_phase3_config)
    validation = json.loads(
        (project_root / phase3.artifacts.validation_metrics).read_text(encoding="utf-8")
    )
    if not validation.get("success", False):
        raise RuntimeError("第二阶段验证未通过，禁止开展ANN闭环。")
    parameters = json.loads(
        (project_root / phase3.artifacts.identified_parameters).read_text(
            encoding="utf-8"
        )
    )
    ocv = pd.read_csv(project_root / phase3.artifacts.ocv_curve)
    return phase3, ReducedBatteryModel(
        phase3, build_ocv_function(ocv), parameters
    )


def _comparison_frame(
    ann_metrics: dict[str, Any],
    mpc_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> pd.DataFrame:
    """汇总同一Chen2020 DFN上的三种控制结果。"""
    records = []
    for name, metrics in (
        ("Filtered 1C baseline", baseline_metrics),
        ("Constrained MPC", mpc_metrics),
        ("Tiny ANN + safety filter", ann_metrics),
    ):
        records.append(
            {
                "controller": name,
                "charge_time_min": metrics["charge_time_min"],
                "maximum_voltage_v": metrics["maximum_voltage_v"],
                "maximum_temperature_c": metrics["maximum_temperature_c"],
                "maximum_current_a": metrics["maximum_current_a"],
                "success": metrics["success"],
            }
        )
    return pd.DataFrame.from_records(records)


def _markdown_table(frame: pd.DataFrame) -> str:
    """不增加tabulate依赖地生成报告中的小型Markdown表格。"""
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


def _write_report(
    path: Path,
    metrics: dict[str, Any],
    comparison: pd.DataFrame,
) -> None:
    """写出强调离线误差与闭环安全区别的中文报告。"""
    offline = metrics["offline_imitation"]
    test = offline["split_metrics"]["test"]
    dfn = metrics["ann_dfn_closed_loop"]
    lines = [
        "# 第四阶段 A 报告：小型ANN模仿与DFN闭环验证",
        "",
        "## 结论",
        "",
        f"阶段4A第一版验收：{'通过' if metrics['success'] else '未通过'}。",
        "",
        "## 网络与数据",
        "",
        f"- 网络结构：{offline['architecture']}，可训练参数：{offline['parameter_count']}；",
        f"- 选中L2正则化：{offline['selected_regularization_alpha']}；初始化种子：{offline['selected_initialization_seed']}；",
        "- 标准化只拟合112个训练样本，验证集用于选模，28个测试样本仅用于最终评价；",
        "- 运行时模型为非可执行NPZ权重，推理只依赖NumPy。",
        "",
        "## 离线模仿",
        "",
        f"- 测试MAE：{test['mae_a']:.4f} A；RMSE：{test['rmse_a']:.4f} A；最大误差：{test['maximum_absolute_error_a']:.4f} A；",
        f"- 测试R²：{test['r2']:.6f}；",
        f"- 线性基线测试RMSE：{offline['linear_baseline']['test_metrics']['rmse_a']:.4f} A。",
        "",
        "## 同约束DFN闭环",
        "",
        _markdown_table(comparison),
        "",
        f"- ANN安全过滤介入：{dfn['safety_filter_intervention_count']}次，占{100 * dfn['safety_filter_intervention_fraction']:.2f}%；",
        f"- ANN平均推理：{dfn['mean_ann_inference_time_ms']:.6f} ms；相对MPC平均求解加速：{metrics['inference_speedup_over_mpc']:.1f}倍；",
        f"- ANN与MPC充电时间相对差：{100 * metrics['dfn_time_gap_fraction']:.2f}%。",
        "",
        "## 解释边界",
        "",
        "- 离线误差小不代表裸ANN安全；本次DFN结果来自ANN加一步安全过滤器；",
        "- 安全层介入比例直接反映网络闭环状态偏离教师采样流形的程度，不能省略；",
        "- 当前教师本身未快于过滤1C，因此ANN目标是降低在线计算量，不是创造更快策略；",
        "- 结果仅覆盖Chen2020、25 ℃、10%到80% SOC，不构成实物BMS验证。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase_four_a(
    config: PhaseFourAConfig, project_root: str | Path
) -> dict[str, Any]:
    """执行小型ANN训练、权重导出和两级闭环验证。"""
    project_root = Path(project_root)
    data_dir = project_root / "data" / "phase4a"
    model_dir = project_root / "outputs" / "models"
    metrics_dir = project_root / "outputs" / "metrics"
    figures_dir = project_root / "outputs" / "figures"
    for directory in (data_dir, model_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    phase3, reduced_model = _load_reduced_model(config, project_root)
    teacher = pd.read_csv(project_root / config.teacher_dataset)
    ann, selection, predictions, offline_metrics = train_tiny_ann(teacher, config)
    model_path = ann.save(model_dir / "phase4a_tiny_ann.npz")
    reloaded = TinyANN.load(model_path)
    reference_features = teacher[list(config.features)].to_numpy(dtype=float)
    round_trip_error = float(
        np.max(np.abs(ann.predict(reference_features) - reloaded.predict(reference_features)))
    )

    selection.to_csv(data_dir / "hyperparameter_selection.csv", index=False)
    predictions.to_csv(data_dir / "offline_predictions.csv", index=False)
    reduced_frame = simulate_ann_reduced_closed_loop(
        reloaded, reduced_model, phase3, config
    )
    reduced_frame.to_csv(data_dir / "ann_reduced_closed_loop.csv", index=False)
    dfn_frame = simulate_ann_dfn_closed_loop(reloaded, reduced_model, phase3, config)
    dfn_frame.to_csv(data_dir / "ann_chen2020_dfn_closed_loop.csv", index=False)
    reduced_metrics = ann_closed_loop_metrics(reduced_frame, phase3)
    dfn_metrics = ann_closed_loop_metrics(dfn_frame, phase3)

    phase3_metrics = json.loads(
        (metrics_dir / "phase3_mpc_metrics.json").read_text(encoding="utf-8")
    )["dfn_closed_loop"]
    phase3b_metrics = json.loads(
        (metrics_dir / "phase3b_metrics.json").read_text(encoding="utf-8")
    )["fair_baseline"]
    mpc_time = float(phase3_metrics["charge_time_min"])
    ann_time = float(dfn_metrics["charge_time_min"]) if dfn_metrics["charge_time_min"] else np.inf
    time_gap_fraction = abs(ann_time - mpc_time) / mpc_time
    speedup = (
        float(phase3_metrics["mean_mpc_solve_time_ms"])
        / dfn_metrics["mean_ann_inference_time_ms"]
    )
    criteria = config.success_criteria
    test_metrics = offline_metrics["split_metrics"]["test"]
    temperature_metrics = offline_metrics["test_temperature_active_metrics"]
    checks = {
        "test_mae": test_metrics["mae_a"] <= criteria.maximum_test_mae_a,
        "test_rmse": test_metrics["rmse_a"] <= criteria.maximum_test_rmse_a,
        "temperature_active_mae": temperature_metrics["mae_a"]
        <= criteria.maximum_active_temperature_mae_a,
        "numpy_model_round_trip": round_trip_error <= 1.0e-12,
        "reduced_closed_loop": reduced_metrics["success"],
        "dfn_closed_loop": dfn_metrics["success"],
        "dfn_time_gap": time_gap_fraction
        <= criteria.maximum_dfn_time_gap_fraction,
        "inference_speedup": speedup
        >= criteria.minimum_inference_speedup_over_mpc,
    }
    comparison = _comparison_frame(dfn_metrics, phase3_metrics, phase3b_metrics)
    comparison.to_csv(metrics_dir / "phase4a_controller_comparison.csv", index=False)
    plot_offline_imitation(
        predictions, figures_dir / "phase4a_offline_imitation.png"
    )
    mpc_frame = pd.read_csv(
        project_root / "data" / "phase3" / "mpc_chen2020_dfn_closed_loop.csv"
    )
    baseline_frame = pd.read_csv(
        project_root / "data" / "phase3b" / "filtered_1c_dfn_closed_loop.csv"
    )
    plot_ann_closed_loop(
        dfn_frame,
        mpc_frame,
        baseline_frame,
        phase3,
        figures_dir / "phase4a_ann_dfn_comparison.png",
    )

    payload = {
        "configuration": asdict(config),
        "offline_imitation": offline_metrics,
        "model_round_trip_maximum_error_a": round_trip_error,
        "ann_reduced_closed_loop": reduced_metrics,
        "ann_dfn_closed_loop": dfn_metrics,
        "mpc_reference": phase3_metrics,
        "fair_baseline_reference": phase3b_metrics,
        "dfn_time_gap_fraction": float(time_gap_fraction),
        "inference_speedup_over_mpc": float(speedup),
        "checks": checks,
        # 安全过滤在DFN中频繁介入，因此当前只能认可带安全包装的纯仿真闭环。
        "standalone_ann_ready": False,
        "success": bool(all(checks.values())),
    }
    metrics_path = metrics_dir / "phase4a_metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = project_root / "outputs" / "phase4a_report.md"
    _write_report(report_path, payload, comparison)
    return {
        "metrics": payload,
        "comparison": comparison,
        "outputs": {
            "model": str(model_path),
            "metrics": str(metrics_path),
            "report": str(report_path),
        },
    }

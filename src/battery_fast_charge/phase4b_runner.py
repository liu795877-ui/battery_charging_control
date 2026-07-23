"""组织阶段4B-1策略诊断、混合教师闭环与准入判断。"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .closed_loop import _cap_current_at_target, initial_reduced_state
from .filtered_baseline import filtered_baseline_metrics
from .identification import build_ocv_function
from .mpc import ReducedBatteryModel
from .phase3_config import load_phase_three_config
from .phase4b_closed_loop import (
    hybrid_closed_loop_metrics,
    simulate_hybrid_dfn_closed_loop,
    simulate_hybrid_reduced_closed_loop,
)
from .phase4b_config import PhaseFourBConfig
from .phase4b_plotting import (
    plot_hybrid_teacher_comparison,
    plot_policy_sweep,
)
from .teacher_data import filter_feasible_current


def _load_model(
    config: PhaseFourBConfig, project_root: Path
) -> tuple[Any, ReducedBatteryModel]:
    """复用已经通过验证的第二阶段参数与第三阶段约束。"""
    phase3 = load_phase_three_config(project_root / config.source_phase3_config)
    validation = json.loads(
        (project_root / phase3.artifacts.validation_metrics).read_text(encoding="utf-8")
    )
    if not validation.get("success", False):
        raise RuntimeError("第二阶段验证未通过，禁止改造MPC教师。")
    parameters = json.loads(
        (project_root / phase3.artifacts.identified_parameters).read_text(
            encoding="utf-8"
        )
    )
    ocv = pd.read_csv(project_root / phase3.artifacts.ocv_curve)
    return phase3, ReducedBatteryModel(
        phase3, build_ocv_function(ocv), parameters
    )


def _simulate_reference_policy(
    model: ReducedBatteryModel,
    phase3,
    peak_current_a: float,
    switch_time_min: float,
    sustainable_current_a: float,
) -> dict[str, Any]:
    """在降阶模型上复核一条峰值-可持续电流参考的可行性。"""
    state = initial_reduced_state(phase3)
    records = [
        {
            "time_s": 0.0,
            "charge_current_a": 0.0,
            "soc": state.soc,
            "terminal_voltage_v": model.ocv(state.soc),
            "average_temperature_c": model.average_temperature(state),
            "safety_override": False,
            "source": "reduced_model",
        }
    ]
    maximum_steps = int(
        np.ceil(
            phase3.control.maximum_simulation_time_s
            / phase3.control.control_interval_s
        )
    )
    for step_index in range(1, maximum_steps + 1):
        elapsed_before_step_s = (step_index - 1) * phase3.control.control_interval_s
        desired = (
            peak_current_a
            if elapsed_before_step_s < 60.0 * switch_time_min
            else sustainable_current_a
        )
        filtered = filter_feasible_current(model, state, desired, phase3)
        current, _ = _cap_current_at_target(filtered.current_a, state.soc, phase3)
        state, output = model.step(state, current)
        records.append(
            {
                "time_s": step_index * phase3.control.control_interval_s,
                "charge_current_a": current,
                "soc": state.soc,
                "terminal_voltage_v": output.terminal_voltage_v,
                "average_temperature_c": output.average_temperature_c,
                "safety_override": filtered.safety_override,
                "source": "reduced_model",
            }
        )
        if state.soc >= phase3.battery.target_soc - phase3.validation.target_soc_tolerance:
            break
    frame = pd.DataFrame.from_records(records)
    metrics = filtered_baseline_metrics(frame, phase3)
    return {
        "peak_current_a": peak_current_a,
        "switch_time_min": switch_time_min,
        "sustainable_current_a": sustainable_current_a,
        **metrics,
    }


def _diagnostic_sweep(
    model: ReducedBatteryModel, phase3, config: PhaseFourBConfig
) -> pd.DataFrame:
    """运行配置中的窄范围参考策略搜索并保留全部组合。"""
    records = []
    for peak in config.diagnostic_sweep.peak_currents_a:
        for switch_time in config.diagnostic_sweep.switch_times_min:
            for sustainable in config.diagnostic_sweep.sustainable_currents_a:
                records.append(
                    _simulate_reference_policy(
                        model, phase3, peak, switch_time, sustainable
                    )
                )
    return pd.DataFrame.from_records(records).sort_values(
        ["success", "charge_time_min"], ascending=[False, True]
    )


def _comparison_frame(
    hybrid: dict[str, Any], original: dict[str, Any], baseline: dict[str, Any]
) -> pd.DataFrame:
    """只比较同一DFN和相同物理约束下的控制结果。"""
    records = []
    for name, metrics in (
        ("Filtered 1C baseline", baseline),
        ("Original constrained MPC", original),
        ("Hybrid thermal-budget teacher", hybrid),
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
    """生成无需额外依赖的小型Markdown表格。"""
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


def _write_report(path: Path, payload: dict[str, Any], comparison: pd.DataFrame) -> None:
    """写出教师改进、终端修正和下一阶段准入结论。"""
    dfn = payload["hybrid_dfn_closed_loop"]
    lines = [
        "# 第四阶段 B-1 报告：热预算MPC教师改进",
        "",
        "## 结论",
        "",
        f"阶段4B-1验收：{'通过' if payload['success'] else '未通过'}。",
        "",
        "## 方法修正",
        "",
        "- 原5 min短视野MPC会过早消耗热余量，并在终点把剩余电量摊到整个预测时域；",
        "- 热预算参考在SOC 17.67%或平均温度30.5 ℃触发后从8 A切换到5 A；",
        "- 启动阶段由参考调节器按每步 2 A 的限制爬升至 8 A；随后在 SOC 低于 20% 时每 5 s 求解热预算 MPC，达到 20% 后由一步可行参考调节器接管；",
        "- 预测到达80%后有效电流归零，实际执行仍由最后一步电量封顶。",
        "",
        "## 同约束DFN结果",
        "",
        _markdown_table(comparison),
        "",
        f"- 相对过滤1C时间改善：{100 * payload['improvement_fraction_over_filtered_1c']:.2f}%；",
        f"- 启动调节步数：{dfn['startup_governor_step_count']}；MPC步数：{dfn['mpc_step_count']}；终端调节步数：{dfn['terminal_governor_step_count']}；",
        f"- 优化成功率：{100 * dfn['optimizer_success_fraction']:.2f}%；回退：{dfn['fallback_count']}；",
        f"- 参考调节器安全覆盖次数：{dfn['reference_governor_safety_override_count']}。",
        "",
        "## 边界",
        "",
        "- 这是热预算MPC加终端参考调节器的混合教师，不是全时域全局最优性证明；",
        "- 改善只在Chen2020、25 ℃、10%到80% SOC和当前约束下成立；",
        "- 只有本阶段闸门通过后，才允许用该教师重新标注主动学习数据。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase_four_b(
    config: PhaseFourBConfig, project_root: str | Path
) -> dict[str, Any]:
    """执行诊断扫描、混合教师双层闭环和1%改进闸门。"""
    project_root = Path(project_root)
    data_dir = project_root / "data" / "phase4b"
    metrics_dir = project_root / "outputs" / "metrics"
    figures_dir = project_root / "outputs" / "figures"
    for directory in (data_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    phase3, model = _load_model(config, project_root)
    sweep = _diagnostic_sweep(model, phase3, config)
    sweep.to_csv(data_dir / "thermal_budget_policy_sweep.csv", index=False)
    reduced = simulate_hybrid_reduced_closed_loop(model, phase3, config)
    reduced.to_csv(data_dir / "hybrid_teacher_reduced_closed_loop.csv", index=False)
    dfn = simulate_hybrid_dfn_closed_loop(model, phase3, config)
    dfn.to_csv(data_dir / "hybrid_teacher_chen2020_dfn_closed_loop.csv", index=False)
    reduced_metrics = hybrid_closed_loop_metrics(reduced, phase3, config)
    dfn_metrics = hybrid_closed_loop_metrics(dfn, phase3, config)

    phase3_metrics = json.loads(
        (metrics_dir / "phase3_mpc_metrics.json").read_text(encoding="utf-8")
    )["dfn_closed_loop"]
    baseline_metrics = json.loads(
        (metrics_dir / "phase3b_metrics.json").read_text(encoding="utf-8")
    )["fair_baseline"]
    baseline_time = float(baseline_metrics["charge_time_min"])
    hybrid_time = float(dfn_metrics["charge_time_min"]) if dfn_metrics["charge_time_min"] else np.inf
    improvement = (baseline_time - hybrid_time) / baseline_time
    criteria = config.success_criteria
    checks = {
        "reduced_closed_loop": reduced_metrics["success"],
        "dfn_closed_loop": dfn_metrics["success"],
        "minimum_time_improvement": improvement
        >= criteria.minimum_improvement_fraction_over_filtered_1c,
        "zero_mpc_fallbacks": dfn_metrics["fallback_count"] == 0,
        "zero_reference_governor_safety_overrides": dfn_metrics[
            "reference_governor_safety_override_count"
        ]
        == 0,
    }
    comparison = _comparison_frame(dfn_metrics, phase3_metrics, baseline_metrics)
    comparison.to_csv(metrics_dir / "phase4b_controller_comparison.csv", index=False)
    plot_policy_sweep(sweep, figures_dir / "phase4b_policy_sweep.png")
    original_dfn = pd.read_csv(
        project_root / "data" / "phase3" / "mpc_chen2020_dfn_closed_loop.csv"
    )
    baseline_dfn = pd.read_csv(
        project_root / "data" / "phase3b" / "filtered_1c_dfn_closed_loop.csv"
    )
    plot_hybrid_teacher_comparison(
        dfn,
        original_dfn,
        baseline_dfn,
        phase3,
        figures_dir / "phase4b_hybrid_teacher_comparison.png",
    )
    payload = {
        "configuration": asdict(config),
        "diagnostic_sweep_best": sweep.iloc[0].to_dict(),
        "hybrid_reduced_closed_loop": reduced_metrics,
        "hybrid_dfn_closed_loop": dfn_metrics,
        "original_mpc_reference": phase3_metrics,
        "filtered_1c_reference": baseline_metrics,
        "improvement_fraction_over_filtered_1c": float(improvement),
        "checks": checks,
        "ready_for_active_data_aggregation": bool(all(checks.values())),
        "success": bool(all(checks.values())),
    }
    metrics_path = metrics_dir / "phase4b_metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = project_root / "outputs" / "phase4b_report.md"
    _write_report(report_path, payload, comparison)
    return {"metrics": payload, "comparison": comparison}

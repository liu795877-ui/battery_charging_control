"""阶段4B-1教师改进诊断和DFN对比图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .phase3_config import PhaseThreeConfig


def plot_policy_sweep(sweep: pd.DataFrame, output_path: str | Path) -> Path:
    """展示无紧急回退候选的时间、峰值电流和切换时刻。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid = sweep[
        sweep["success"] & (sweep["safety_override_count"] == 0)
    ]
    figure, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    scatter = axis.scatter(
        valid["switch_time_min"],
        valid["charge_time_min"],
        c=valid["peak_current_a"],
        s=45 + 25 * (valid["sustainable_current_a"] - 4.5),
        cmap="viridis",
        alpha=0.85,
    )
    axis.axhline(53.3333333333, color="black", linestyle=":", label="Filtered 1C DFN reference")
    axis.set(
        xlabel="High-current switch time [min]",
        ylabel="Reduced-model charge time [min]",
        title="Feasible thermal-budget reference sweep",
    )
    axis.legend()
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Peak current [A]")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_hybrid_teacher_comparison(
    hybrid_dfn: pd.DataFrame,
    mpc_dfn: pd.DataFrame,
    baseline_dfn: pd.DataFrame,
    config: PhaseThreeConfig,
    output_path: str | Path,
) -> Path:
    """用同一坐标比较混合教师、原MPC和过滤1C。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    styles = [
        (baseline_dfn, "Filtered 1C", "tab:blue", "-"),
        (mpc_dfn, "Original MPC", "tab:orange", "--"),
        (hybrid_dfn, "Hybrid thermal-budget teacher", "tab:green", "-."),
    ]
    variables = [
        ("charge_current_a", "Charge current [A]", config.constraints.maximum_current_a),
        ("terminal_voltage_v", "Terminal voltage [V]", config.constraints.physical_maximum_voltage_v),
        ("soc", "SOC [-]", config.battery.target_soc),
        ("average_temperature_c", "Average temperature [degC]", config.constraints.physical_maximum_temperature_c),
    ]
    for axis, (variable, ylabel, limit) in zip(axes.flat, variables):
        for frame, label, color, linestyle in styles:
            axis.plot(
                frame["time_s"] / 60.0,
                frame[variable],
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
            )
        axis.axhline(limit, color="black", linestyle=":", linewidth=1.4)
        axis.set(xlabel="Time [min]", ylabel=ylabel)
        axis.legend(fontsize=8)
    figure.suptitle("Phase 4B-1: same-constraint Chen2020 DFN teacher comparison")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path

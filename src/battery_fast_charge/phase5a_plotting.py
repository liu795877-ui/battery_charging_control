"""阶段5A有界压力场景和DFN温度锚点图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_reduced_stress_summary(summary: pd.DataFrame, path: str | Path) -> None:
    """显示温度/参数扰动下的时间、电压、温度和SOC终端误差。"""
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    categories = (
        (summary["completion_success"] & summary["physical_safe"], "complete + safe", "#54A24B", "o"),
        (~summary["completion_success"] & summary["physical_safe"], "incomplete, safe", "#F2CF5B", "^"),
        (summary["completion_success"] & ~summary["physical_safe"], "complete, unsafe", "#E45756", "x"),
        (~summary["completion_success"] & ~summary["physical_safe"], "incomplete + unsafe", "#B279A2", "s"),
    )
    panels = (
        (axes[0, 0], "ambient_temperature_c", "charge_time_min", 1.0),
        (axes[0, 1], "resistance_multiplier", "maximum_voltage_v", 1.0),
        (axes[1, 0], "ambient_temperature_c", "maximum_temperature_c", 1.0),
        (axes[1, 1], "soc_bias", "terminal_true_soc_error", 100.0),
    )
    for axis, x_column, y_column, scale in panels:
        for mask, label, color, marker in categories:
            axis.scatter(
                summary.loc[mask, x_column],
                scale * summary.loc[mask, y_column],
                color=color,
                marker=marker,
                alpha=0.85,
                label=label,
            )
    axes[0, 0].set(xlabel="Ambient temperature [degC]", ylabel="Charge time [min]")
    axes[0, 1].set(xlabel="Resistance multiplier [-]", ylabel="Maximum voltage [V]")
    axes[1, 0].set(xlabel="Ambient temperature [degC]", ylabel="Maximum temperature [degC]")
    axes[1, 1].set(xlabel="SOC estimate bias [-]", ylabel="Terminal true SOC error [%]")
    axes[0, 1].axhline(4.2, color="black", linestyle=":")
    axes[1, 0].axhline(35.0, color="black", linestyle=":")
    axes[1, 1].axhline(1.5, color="black", linestyle=":")
    axes[1, 1].axhline(-1.5, color="black", linestyle=":")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    axes[0, 0].legend(loc="best", fontsize=8)
    figure.suptitle("Phase 5A: bounded reduced-model stress test")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_dfn_temperature_anchors(
    trajectories: pd.DataFrame, path: str | Path
) -> None:
    """比较三个温度锚点的电流、电压、SOC和平均温度。"""
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    palette = {15.0: "#4C78A8", 25.0: "#54A24B", 30.0: "#E45756"}
    for temperature, frame in trajectories.groupby("anchor_temperature_c"):
        time_min = frame["time_s"] / 60.0
        label = f"{temperature:.0f} degC"
        color = palette.get(float(temperature))
        axes[0, 0].plot(time_min, frame["charge_current_a"], color=color, label=label)
        axes[0, 1].plot(time_min, frame["terminal_voltage_v"], color=color, label=label)
        axes[1, 0].plot(time_min, frame["soc"], color=color, label=label)
        axes[1, 1].plot(time_min, frame["average_temperature_c"], color=color, label=label)
    axes[0, 0].axhline(10.0, color="black", linestyle=":")
    axes[0, 1].axhline(4.2, color="black", linestyle=":")
    axes[1, 0].axhline(0.8, color="black", linestyle=":")
    axes[1, 1].axhline(35.0, color="black", linestyle=":")
    axes[0, 0].set(xlabel="Time [min]", ylabel="Charge current [A]")
    axes[0, 1].set(xlabel="Time [min]", ylabel="Terminal voltage [V]")
    axes[1, 0].set(xlabel="Time [min]", ylabel="SOC [-]")
    axes[1, 1].set(xlabel="Time [min]", ylabel="Average temperature [degC]")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    figure.suptitle("Phase 5A: Chen2020 DFN temperature anchors")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)

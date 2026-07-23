"""阶段4A离线模仿和ANN闭环验证图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .phase3_config import PhaseThreeConfig


def plot_offline_imitation(
    predictions: pd.DataFrame, output_path: str | Path
) -> Path:
    """展示教师-ANN一致性、SOC误差以及约束区域误差。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"train": "tab:blue", "validation": "tab:orange", "test": "tab:green"}
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    for split, frame in predictions.groupby("split"):
        axes[0].scatter(
            frame["teacher_current_a"],
            frame["ann_current_a"],
            s=28,
            alpha=0.8,
            label=split,
            color=colors[split],
        )
        axes[1].scatter(
            100.0 * frame["state_soc"],
            frame["ann_error_a"],
            s=28,
            alpha=0.8,
            label=split,
            color=colors[split],
        )
    bounds = [
        float(
            min(
                predictions["teacher_current_a"].min(),
                predictions["ann_current_a"].min(),
            )
        ),
        float(
            max(
                predictions["teacher_current_a"].max(),
                predictions["ann_current_a"].max(),
            )
        ),
    ]
    axes[0].plot(bounds, bounds, color="black", linestyle=":", label="ideal")
    axes[0].set(
        xlabel="MPC teacher current [A]",
        ylabel="ANN current [A]",
        title="Teacher-current imitation",
    )
    axes[0].legend()
    axes[1].axhline(0.0, color="black", linestyle=":")
    axes[1].set(
        xlabel="SOC [%]", ylabel="ANN error [A]", title="Error over SOC"
    )

    test = predictions[predictions["split"] == "test"].copy()
    categories = {
        "All test": np.ones(len(test), dtype=bool),
        "Temperature active": test["active_temperature_constraint"].astype(bool),
        "Voltage active": test["active_voltage_constraint"].astype(bool),
    }
    mae = [
        float(test.loc[mask, "ann_error_a"].abs().mean())
        for mask in categories.values()
    ]
    axes[2].bar(list(categories), mae, color=["tab:green", "tab:red", "tab:purple"])
    axes[2].set(ylabel="MAE [A]", title="Test errors near constraints")
    axes[2].tick_params(axis="x", rotation=18)
    figure.suptitle("Phase 4A: tiny ANN offline imitation")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path

def plot_ann_closed_loop(
    ann_dfn: pd.DataFrame,
    mpc_dfn: pd.DataFrame,
    baseline_dfn: pd.DataFrame,
    config: PhaseThreeConfig,
    output_path: str | Path,
) -> Path:
    """在同一DFN坐标中比较ANN、MPC和过滤1C基线。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    styles = [
        (baseline_dfn, "Filtered 1C", "tab:blue", "-"),
        (mpc_dfn, "Constrained MPC", "tab:orange", "--"),
        (ann_dfn, "Tiny ANN + safety filter", "tab:green", "-."),
    ]
    variables = [
        ("charge_current_a", "Charge current [A]", config.constraints.maximum_current_a),
        (
            "terminal_voltage_v",
            "Terminal voltage [V]",
            config.constraints.physical_maximum_voltage_v,
        ),
        ("soc", "SOC [-]", config.battery.target_soc),
        (
            "average_temperature_c",
            "Average temperature [degC]",
            config.constraints.physical_maximum_temperature_c,
        ),
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
        axis.axhline(limit, color="black", linestyle=":", linewidth=1.5)
        axis.set(xlabel="Time [min]", ylabel=ylabel)
        axis.legend(fontsize=8)
    figure.suptitle("Phase 4A: same-constraint Chen2020 DFN comparison")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path

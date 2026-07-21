"""Phase 6 数据覆盖、离线拟合和名义闭环审计图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_paper_dataset_audit(
    attempts: pd.DataFrame, dataset: pd.DataFrame, path: str | Path
) -> None:
    """同时显示初态接受区域、展开状态覆盖、标签分布和约束激活。"""
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = attempts["teacher_accepted"].map({True: "#54A24B", False: "#E45756"})
    axes[0, 0].scatter(
        attempts["state_soc"], attempts["state_average_temperature_c"],
        c=colors, alpha=0.65, s=24,
    )
    axes[0, 0].set(xlabel="Initial SOC [-]", ylabel="Initial average temperature [degC]", title="Initial-state feasibility")
    for split, color in (("train", "#4C78A8"), ("validation", "#F2CF5B"), ("test", "#B279A2")):
        frame = dataset[dataset["split"] == split]
        axes[0, 1].scatter(
            frame["state_soc"], frame["state_polarization_fast_v"] + frame["state_polarization_slow_v"],
            label=split, color=color, alpha=0.45, s=12,
        )
    axes[0, 1].set(xlabel="SOC [-]", ylabel="Total polarization [V]", title="Unfolded state coverage")
    axes[0, 1].legend()
    axes[1, 0].hist(dataset["teacher_current_a"], bins=30, color="#4C78A8", alpha=0.8)
    axes[1, 0].set(xlabel="MPC teacher current [A]", ylabel="Samples", title="Label distribution")
    counts = {
        "voltage": int(dataset["active_voltage_constraint"].sum()),
        "temperature": int(dataset["active_temperature_constraint"].sum()),
        "current": int(dataset["active_current_upper_constraint"].sum()),
        "slew": int(dataset["active_current_change_constraint"].sum()),
    }
    axes[1, 1].bar(
        list(counts),
        list(counts.values()),
        color=["#4C78A8", "#E45756", "#54A24B", "#F2CF5B"],
    )
    axes[1, 1].set(ylabel="Active samples", title="Constraint-active coverage")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle("Phase 6: paper-style MPC dataset audit")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_paper_dnn_offline(predictions: pd.DataFrame, path: str | Path) -> None:
    """显示测试集教师—DNN 一致性和误差随状态的分布。"""
    test = predictions[predictions["split"] == "test"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].scatter(test["teacher_current_a"], test["dnn_current_a"], alpha=0.55, s=18)
    low = min(test["teacher_current_a"].min(), test["dnn_current_a"].min())
    high = max(test["teacher_current_a"].max(), test["dnn_current_a"].max())
    axes[0].plot([low, high], [low, high], "k:")
    axes[0].set(xlabel="MPC teacher current [A]", ylabel="Pure DNN current [A]", title="Held-out trajectories")
    axes[1].scatter(test["state_soc"], test["dnn_error_a"], c=test["state_average_temperature_c"], cmap="viridis", alpha=0.65, s=18)
    axes[1].axhline(0.0, color="black", linestyle=":")
    axes[1].set(xlabel="SOC [-]", ylabel="DNN - MPC [A]", title="Error versus SOC (color: temperature)")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("Phase 6: offline explicit-control imitation")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_nominal_closed_loop(
    dnn: pd.DataFrame, teacher: pd.DataFrame, path: str | Path
) -> None:
    """比较 25 ℃ Chen2020 DFN 上裸 DNN 与在线 MPC 教师轨迹。"""
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for frame, label, color in (
        (teacher, "Online MPC teacher", "#4C78A8"),
        (dnn, "Pure paper-style DNN", "#E45756"),
    ):
        time_min = frame["time_s"] / 60.0
        axes[0, 0].plot(time_min, frame["charge_current_a"], label=label, color=color)
        axes[0, 1].plot(time_min, frame["terminal_voltage_v"], label=label, color=color)
        axes[1, 0].plot(time_min, frame["soc"], label=label, color=color)
        axes[1, 1].plot(time_min, frame["average_temperature_c"], label=label, color=color)
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
    figure.suptitle("Phase 6: nominal 25 degC Chen2020 DFN comparison")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)

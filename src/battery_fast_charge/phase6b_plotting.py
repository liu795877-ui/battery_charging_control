"""Plots for Phase 6B DNN failure diagnosis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_phase6b_error_partitions(partitions: pd.DataFrame, path: str | Path) -> None:
    """Plot the highest held-out RMSE partitions."""
    test = partitions[partitions["split"] == "test"].copy()
    test = test.sort_values("rmse_a", ascending=False).head(14)
    labels = test["partition_family"] + "\n" + test["partition_label"]
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    axis.bar(range(len(test)), test["rmse_a"], color="#4C78A8")
    axis.set_xticks(range(len(test)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_ylabel("RMSE [A]")
    axis.set_title("Phase 6B: worst held-out DNN error partitions")
    axis.grid(axis="y", alpha=0.25)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_phase6b_closed_loop_comparison(
    teacher: pd.DataFrame,
    pure: pd.DataFrame,
    projected: pd.DataFrame,
    path: str | Path,
) -> None:
    """Compare online MPC, pure DNN, and projected DNN on the nominal DFN case."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    series = (
        (teacher, "Online MPC teacher", "#4C78A8"),
        (pure, "Pure DNN", "#E45756"),
        (projected, "Projected DNN", "#54A24B"),
    )
    for frame, label, color in series:
        time_min = frame["time_s"] / 60.0
        axes[0, 0].plot(time_min, frame["charge_current_a"], label=label, color=color)
        axes[0, 1].plot(time_min, frame["terminal_voltage_v"], label=label, color=color)
        axes[1, 0].plot(time_min, frame["soc"], label=label, color=color)
        axes[1, 1].plot(time_min, frame["average_temperature_c"], label=label, color=color)
    axes[0, 0].axhline(10.0, color="black", linestyle=":")
    axes[0, 1].axhline(4.2, color="black", linestyle=":")
    axes[1, 0].axhline(0.8, color="black", linestyle=":")
    axes[1, 1].axhline(35.0, color="black", linestyle=":")
    axes[0, 0].set(xlabel="Time [min]", ylabel="Current [A]")
    axes[0, 1].set(xlabel="Time [min]", ylabel="Voltage [V]")
    axes[1, 0].set(xlabel="Time [min]", ylabel="SOC [-]")
    axes[1, 1].set(xlabel="Time [min]", ylabel="Average temperature [degC]")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    figure.suptitle("Phase 6B: pure vs projected DNN nominal 25 degC comparison")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)

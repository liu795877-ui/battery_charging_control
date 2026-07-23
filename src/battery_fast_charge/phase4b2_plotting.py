"""阶段4B-2主动数据覆盖、离线模仿和DFN闭环图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .phase3_config import PhaseThreeConfig


def plot_active_coverage(dataset: pd.DataFrame, path: str | Path) -> None:
    """比较旧状态与ANN中心主动状态的SOC-温度-电流覆盖。"""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    colors = {
        "legacy_relabelled": "#4C78A8",
        "active_ann_rollout": "#F58518",
        "dagger_round_2": "#54A24B",
        "dagger_round_3_dfn": "#B279A2",
    }
    labels = {
        "legacy_relabelled": "Legacy reachable states",
        "active_ann_rollout": "ANN-centered active states",
        "dagger_round_2": "ANN round-1 on-policy states",
        "dagger_round_3_dfn": "ANN round-2 DFN states",
    }
    for source, group in dataset.groupby("source_dataset"):
        axes[0].scatter(
            group["state_soc"],
            group["state_average_temperature_c"],
            s=18,
            alpha=0.7,
            color=colors[source],
            label=labels[source],
        )
        axes[1].scatter(
            group["state_soc"],
            group["state_previous_current_a"],
            s=18,
            alpha=0.7,
            color=colors[source],
            label=labels[source],
        )
    axes[0].set(xlabel="SOC [-]", ylabel="Average temperature [degC]")
    axes[1].set(xlabel="SOC [-]", ylabel="Previous current [A]")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    figure.suptitle("Phase 4B-2: reachable-state active data coverage")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_offline_active_imitation(
    predictions: pd.DataFrame, path: str | Path
) -> None:
    """显示第二版ANN对混合教师的预测和误差随SOC分布。"""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    split_colors = {"train": "#4C78A8", "validation": "#F58518", "test": "#54A24B"}
    for split, group in predictions.groupby("split"):
        axes[0].scatter(
            group["teacher_current_a"],
            group["ann_current_a"],
            s=18,
            alpha=0.7,
            color=split_colors[split],
            label=split,
        )
        axes[1].scatter(
            group["state_soc"],
            group["ann_error_a"],
            s=18,
            alpha=0.7,
            color=split_colors[split],
            label=split,
        )
    limits = [0.0, 10.0]
    axes[0].plot(limits, limits, "k:", linewidth=1.5)
    axes[0].set(xlim=limits, ylim=limits, xlabel="Hybrid teacher [A]", ylabel="ANN v2 [A]")
    axes[1].axhline(0.0, color="black", linestyle=":", linewidth=1.5)
    axes[1].set(xlabel="SOC [-]", ylabel="ANN v2 error [A]")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    figure.suptitle("Phase 4B-2: offline hybrid-teacher imitation")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_ann_v2_dfn_comparison(
    ann_v2: pd.DataFrame,
    ann_v1: pd.DataFrame,
    hybrid: pd.DataFrame,
    phase3: PhaseThreeConfig,
    path: str | Path,
) -> None:
    """比较新旧ANN与混合教师在相同DFN上的四条关键曲线。"""
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    series = (
        (ann_v1, "ANN v1 + safety filter", "#9E9E9E", "--"),
        (hybrid, "Hybrid teacher", "#54A24B", "-."),
        (ann_v2, "ANN v2 + safety filter", "#E45756", "-"),
    )
    for frame, label, color, style in series:
        time_min = frame["time_s"] / 60.0
        axes[0, 0].plot(time_min, frame["charge_current_a"], style, color=color, label=label)
        axes[0, 1].plot(time_min, frame["terminal_voltage_v"], style, color=color, label=label)
        axes[1, 0].plot(time_min, frame["soc"], style, color=color, label=label)
        axes[1, 1].plot(time_min, frame["average_temperature_c"], style, color=color, label=label)
    axes[0, 0].axhline(phase3.constraints.maximum_current_a, color="black", linestyle=":")
    axes[0, 1].axhline(phase3.constraints.physical_maximum_voltage_v, color="black", linestyle=":")
    axes[1, 0].axhline(phase3.battery.target_soc, color="black", linestyle=":")
    axes[1, 1].axhline(phase3.constraints.physical_maximum_temperature_c, color="black", linestyle=":")
    axes[0, 0].set(xlabel="Time [min]", ylabel="Charge current [A]")
    axes[0, 1].set(xlabel="Time [min]", ylabel="Terminal voltage [V]")
    axes[1, 0].set(xlabel="Time [min]", ylabel="SOC [-]")
    axes[1, 1].set(xlabel="Time [min]", ylabel="Average temperature [degC]")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    figure.suptitle("Phase 4B-2: Chen2020 DFN closed-loop comparison")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)

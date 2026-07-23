"""第三阶段 B 的数据覆盖图与同约束基线对比图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from .phase3_config import PhaseThreeConfig

matplotlib.use("Agg")


def plot_teacher_dataset_coverage(
    dataset: pd.DataFrame, output_path: str | Path
) -> Path:
    """展示状态空间、标签和活跃约束是否得到覆盖。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"train": "tab:blue", "validation": "tab:orange", "test": "tab:green"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for split, group in dataset.groupby("split"):
        axes[0, 0].scatter(
            100.0 * group["state_soc"],
            group["state_average_temperature_c"],
            s=18,
            alpha=0.75,
            label=split,
            color=colors.get(str(split)),
        )
        axes[0, 1].scatter(
            group["state_polarization_fast_v"],
            group["state_polarization_slow_v"],
            s=18,
            alpha=0.75,
            label=split,
            color=colors.get(str(split)),
        )
    axes[0, 0].set_xlabel("SOC [%]")
    axes[0, 0].set_ylabel("Average temperature [degC]")
    axes[0, 0].set_title("Reachable state coverage")
    axes[0, 0].legend()
    axes[0, 1].set_xlabel("Fast polarization [V]")
    axes[0, 1].set_ylabel("Slow polarization [V]")
    axes[0, 1].set_title("Electrical-state coverage")

    active_temperature = dataset["active_temperature_constraint"].astype(bool)
    axes[1, 0].scatter(
        100.0 * dataset.loc[~active_temperature, "state_soc"],
        dataset.loc[~active_temperature, "teacher_current_a"],
        s=18,
        alpha=0.65,
        label="temperature inactive",
    )
    axes[1, 0].scatter(
        100.0 * dataset.loc[active_temperature, "state_soc"],
        dataset.loc[active_temperature, "teacher_current_a"],
        s=24,
        alpha=0.85,
        label="temperature active",
    )
    axes[1, 0].set_xlabel("SOC [%]")
    axes[1, 0].set_ylabel("MPC teacher current [A]")
    axes[1, 0].set_title("Teacher labels")
    axes[1, 0].legend()

    active_columns = {
        "Voltage": "active_voltage_constraint",
        "Temperature": "active_temperature_constraint",
        "Current": "active_current_upper_constraint",
        "Current change": "active_current_change_constraint",
    }
    counts = [int(dataset[column].sum()) for column in active_columns.values()]
    axes[1, 1].bar(active_columns.keys(), counts, color="tab:purple")
    axes[1, 1].set_ylabel("Accepted samples")
    axes[1, 1].set_title("Active-constraint coverage")
    axes[1, 1].tick_params(axis="x", rotation=20)

    figure.suptitle("Phase 3B: reachable MPC teacher dataset")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_fair_baseline_comparison(
    baseline_dfn: pd.DataFrame,
    mpc_dfn: pd.DataFrame,
    config: PhaseThreeConfig,
    output_path: str | Path,
) -> Path:
    """在相同DFN与物理约束下比较非优化基线和MPC。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    baseline_time = baseline_dfn["time_s"] / 60.0
    mpc_time = mpc_dfn["time_s"] / 60.0

    series = [
        ("charge_current_a", "Charge current [A]"),
        ("terminal_voltage_v", "Terminal voltage [V]"),
        ("soc", "SOC [-]"),
        ("average_temperature_c", "Average temperature [degC]"),
    ]
    for axis, (column, label) in zip(axes.flat, series):
        axis.plot(baseline_time, baseline_dfn[column], label="Filtered 1C baseline")
        axis.plot(mpc_time, mpc_dfn[column], "--", label="Constrained MPC")
        axis.set_ylabel(label)
        axis.legend(fontsize=8)
    axes[0, 0].axhline(config.constraints.maximum_current_a, color="black", linestyle=":")
    axes[0, 1].axhline(config.constraints.physical_maximum_voltage_v, color="black", linestyle=":")
    axes[1, 0].axhline(config.battery.target_soc, color="black", linestyle=":")
    axes[1, 1].axhline(config.constraints.physical_maximum_temperature_c, color="black", linestyle=":")
    axes[1, 0].set_xlabel("Time [min]")
    axes[1, 1].set_xlabel("Time [min]")
    figure.suptitle("Same-constraint DFN comparison")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path

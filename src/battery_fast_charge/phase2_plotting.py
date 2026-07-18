"""第二阶段虚拟表征与降阶模型验证图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_characterization(
    ocv_table: pd.DataFrame, pulse_data: pd.DataFrame, output_path: str | Path
) -> Path:
    """绘制 OCV–SOC 曲线和不同初始 SOC 的脉冲电压响应。"""
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(ocv_table["soc"] * 100.0, ocv_table["ocv_v"], marker="o")
    axes[0].set(
        title="Chen2020 OCV curve",
        xlabel="SOC [%]",
        ylabel="Open-circuit voltage [V]",
    )
    for name, frame in pulse_data.groupby("profile_name"):
        initial_soc = float(frame["initial_soc"].iloc[0])
        axes[1].plot(
            frame["time_s"] / 60.0,
            frame["terminal_voltage_v"],
            label=f"SOC {initial_soc * 100:.0f}%",
        )
    axes[1].set(
        title="Charge-pulse voltage responses",
        xlabel="Time [min]",
        ylabel="Terminal voltage [V]",
    )
    axes[1].legend(ncol=2, fontsize=8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_validation(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """把独立验证集的输入、真实值、预测值和误差画在一张图中。"""
    plt.style.use("seaborn-v0_8-whitegrid")
    time_min = frame["time_s"] / 60.0
    figure, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)

    axes[0, 0].plot(time_min, frame["charge_current_a"])
    axes[0, 0].set(title="Validation current", ylabel="Current [A]")
    axes[0, 1].plot(time_min, frame["soc"] * 100.0, label="DFN")
    axes[0, 1].plot(
        time_min,
        frame["soc_predicted"] * 100.0,
        linestyle="--",
        label="2RC",
    )
    axes[0, 1].set(title="SOC", ylabel="SOC [%]")
    axes[0, 1].legend()

    axes[1, 0].plot(time_min, frame["terminal_voltage_v"], label="DFN")
    axes[1, 0].plot(
        time_min,
        frame["terminal_voltage_predicted_v"],
        linestyle="--",
        label="2RC",
    )
    axes[1, 0].set(title="Terminal voltage", ylabel="Voltage [V]")
    axes[1, 0].legend()
    axes[1, 1].plot(
        time_min,
        (frame["terminal_voltage_predicted_v"] - frame["terminal_voltage_v"]) * 1000.0,
    )
    axes[1, 1].axhline(0.0, color="black", linewidth=1)
    axes[1, 1].set(title="Voltage error", ylabel="Error [mV]")

    axes[2, 0].plot(time_min, frame["average_temperature_c"], label="DFN average")
    axes[2, 0].plot(
        time_min,
        frame["average_temperature_predicted_c"],
        linestyle="--",
        label="Two-node average",
    )
    axes[2, 0].plot(
        time_min,
        frame["core_temperature_predicted_c"],
        alpha=0.7,
        label="Latent core",
    )
    axes[2, 0].plot(
        time_min,
        frame["surface_temperature_predicted_c"],
        alpha=0.7,
        label="Latent surface",
    )
    axes[2, 0].set(
        title="Temperature (core/surface are latent)",
        xlabel="Time [min]",
        ylabel="Temperature [degC]",
    )
    axes[2, 0].legend(fontsize=8)
    axes[2, 1].plot(
        time_min,
        frame["average_temperature_predicted_c"] - frame["average_temperature_c"],
    )
    axes[2, 1].axhline(0.0, color="black", linewidth=1)
    axes[2, 1].set(
        title="Average-temperature error",
        xlabel="Time [min]",
        ylabel="Error [degC]",
    )
    for axis in axes.flat:
        if not axis.get_xlabel():
            axis.set_xlabel("Time [min]")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path

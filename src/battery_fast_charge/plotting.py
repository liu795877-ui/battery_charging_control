"""绘制不同 CC–CV 工况的 SOC、电压、电流和温度对比图。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import PhaseOneConfig


def plot_baselines(
    trajectories: Mapping[float, pd.DataFrame],
    config: PhaseOneConfig,
    output_path: str | Path,
) -> Path:
    """把所有倍率的四类关键轨迹画在同一张图中并保存。"""
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # 四个子图的位置固定为：左上 SOC、右上电压、左下电流、右下温度。
    for c_rate, frame in trajectories.items():
        time_min = frame["time_s"] / 60.0
        label = f"{c_rate:g}C"
        axes[0, 0].plot(time_min, frame["soc"] * 100.0, label=label)
        axes[0, 1].plot(time_min, frame["terminal_voltage_v"], label=label)
        axes[1, 0].plot(time_min, frame["charge_current_a"], label=label)
        axes[1, 1].plot(time_min, frame["cell_temperature_c"], label=label)

    # 黑色虚线是目标或约束，不是另一条仿真轨迹。
    axes[0, 0].axhline(
        config.battery.target_soc * 100.0, color="black", linestyle="--", linewidth=1
    )
    axes[0, 1].axhline(
        config.constraints.maximum_voltage_v,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    axes[1, 0].axhline(
        config.constraints.maximum_current_a,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    axes[1, 1].axhline(
        config.constraints.maximum_temperature_c,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    axes[0, 0].set(title="State of charge", ylabel="SOC [%]")
    axes[0, 1].set(title="Terminal voltage", ylabel="Voltage [V]")
    axes[1, 0].set(title="Charge current", xlabel="Time [min]", ylabel="Current [A]")
    axes[1, 1].set(
        title="Cell temperature", xlabel="Time [min]", ylabel="Temperature [°C]"
    )
    for axis in axes.flat:
        axis.legend()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 180 dpi 足以在 Notebook 和普通报告中清晰显示。
    figure.savefig(output_path, dpi=180)
    # 主动关闭图形可避免批量运行时 Matplotlib 持续占用内存。
    plt.close(figure)
    return output_path

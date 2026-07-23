"""第三阶段 MPC 闭环轨迹的四面板图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from .phase3_config import PhaseThreeConfig

matplotlib.use("Agg")


def plot_phase_three_closed_loop(
    reduced: pd.DataFrame,
    dfn: pd.DataFrame,
    config: PhaseThreeConfig,
    output_path: str | Path,
) -> Path:
    """并排展示电流、电压、SOC和平均温度，突出两层约束。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    time_reduced = reduced["time_s"] / 60.0
    time_dfn = dfn["time_s"] / 60.0

    axes[0, 0].plot(time_reduced, reduced["charge_current_a"], label="MPC on reduced")
    axes[0, 0].plot(time_dfn, dfn["charge_current_a"], "--", label="MPC applied to DFN")
    axes[0, 0].axhline(config.constraints.maximum_current_a, color="black", linestyle=":", label="Current limit")
    axes[0, 0].set_ylabel("Charge current [A]")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(time_reduced, reduced["terminal_voltage_v"], label="Reduced model")
    axes[0, 1].plot(time_dfn, dfn["terminal_voltage_v"], "--", label="Chen2020 DFN")
    axes[0, 1].axhline(config.constraints.mpc_maximum_voltage_v, color="tab:orange", linestyle=":", label="MPC limit")
    axes[0, 1].axhline(config.constraints.physical_maximum_voltage_v, color="black", linestyle="--", label="Physical limit")
    axes[0, 1].set_ylabel("Terminal voltage [V]")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(time_reduced, 100.0 * reduced["soc"], label="Reduced model")
    axes[1, 0].plot(time_dfn, 100.0 * dfn["soc"], "--", label="Chen2020 DFN")
    axes[1, 0].axhline(100.0 * config.battery.target_soc, color="black", linestyle=":", label="Target")
    axes[1, 0].set_ylabel("SOC [%]")
    axes[1, 0].set_xlabel("Time [min]")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(time_reduced, reduced["average_temperature_c"], label="Reduced model")
    axes[1, 1].plot(time_dfn, dfn["average_temperature_c"], "--", label="Chen2020 DFN")
    axes[1, 1].axhline(config.constraints.mpc_maximum_temperature_c, color="tab:orange", linestyle=":", label="MPC limit")
    axes[1, 1].axhline(config.constraints.physical_maximum_temperature_c, color="black", linestyle="--", label="Physical limit")
    axes[1, 1].set_ylabel("Average temperature [degC]")
    axes[1, 1].set_xlabel("Time [min]")
    axes[1, 1].legend(fontsize=8)

    figure.suptitle("Phase 3A: constrained MPC teacher closed-loop validation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path

"""用 Chen2020 DFN 生成第二阶段的虚拟表征数据。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pybamm

from .phase2_config import PhaseTwoConfig, ProfileSegment


def _model_and_parameters(
    config: PhaseTwoConfig,
) -> tuple[pybamm.BaseModel, pybamm.ParameterValues]:
    """创建与第一阶段一致的 DFN＋集总热虚拟电芯。"""
    model = pybamm.lithium_ion.DFN(options={"thermal": config.battery.thermal_model})
    parameters = pybamm.ParameterValues(config.battery.parameter_set)
    parameters.update(
        {
            "Ambient temperature [K]": config.battery.ambient_temperature_c + 273.15,
            "Initial temperature [K]": config.battery.initial_temperature_c + 273.15,
        }
    )
    return model, parameters


def _solution_frame(
    solution: pybamm.Solution,
    initial_soc: float,
    nominal_capacity_ah: float,
    profile_name: str,
) -> pd.DataFrame:
    """统一导出符号和单位；本项目仍规定充电电流为正。"""

    def values(name: str) -> np.ndarray:
        return np.asarray(solution[name].entries, dtype=float).reshape(-1)

    discharge_capacity_ah = values("Discharge capacity [A.h]")
    frame = pd.DataFrame(
        {
            "time_s": values("Time [s]"),
            "charge_current_a": -values("Current [A]"),
            "terminal_voltage_v": values("Terminal voltage [V]"),
            "battery_ocv_v": values("Battery open-circuit voltage [V]"),
            "soc": initial_soc - discharge_capacity_ah / nominal_capacity_ah,
            "average_temperature_c": values("Volume-averaged cell temperature [C]"),
            "total_heating_w": values("Total heating [W]"),
            "discharge_capacity_ah": discharge_capacity_ah,
            "profile_name": profile_name,
        }
    )
    # 实验步骤交界处可能有重复时间，只保留新步骤一侧的状态和电流。
    return frame.drop_duplicates("time_s", keep="last").reset_index(drop=True)


def _run_experiment(
    steps: Sequence[str],
    initial_soc: float,
    config: PhaseTwoConfig,
    profile_name: str,
) -> pd.DataFrame:
    """运行一组 PyBaMM 实验步骤并返回标准表格。"""
    model, parameters = _model_and_parameters(config)
    experiment = pybamm.Experiment(
        list(steps), period=f"{config.experiment.sample_period_s:g} seconds"
    )
    solution = pybamm.Simulation(
        model, parameter_values=parameters, experiment=experiment
    ).solve(initial_soc=initial_soc)
    return _solution_frame(
        solution,
        initial_soc,
        config.battery.nominal_capacity_ah,
        profile_name,
    )


def generate_ocv_curve(config: PhaseTwoConfig) -> pd.DataFrame:
    """在各 SOC 平衡初态读取 DFN 的开路电压。"""
    records: list[dict[str, float]] = []
    for soc in config.experiment.ocv_soc_points:
        frame = _run_experiment(["Rest for 10 seconds"], soc, config, f"ocv_{soc:.2f}")
        records.append(
            {
                "soc": soc,
                # 取静置末端，避免初始代数状态的极小数值扰动。
                "ocv_v": float(frame["battery_ocv_v"].iloc[-1]),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("soc").reset_index(drop=True)


def generate_pulse_data(config: PhaseTwoConfig) -> pd.DataFrame:
    """在多个 SOC 下施加不同倍率的短充电脉冲和充分静置。"""
    frames: list[pd.DataFrame] = []
    for soc in config.experiment.pulse_soc_points:
        steps = [f"Rest for {config.experiment.rest_before_s:g} seconds"]
        for c_rate in config.experiment.pulse_c_rates:
            steps.extend(
                [
                    f"Charge at {c_rate:g} C for "
                    f"{config.experiment.pulse_duration_s:g} seconds",
                    f"Rest for {config.experiment.rest_after_s:g} seconds",
                ]
            )
        name = f"pulse_soc_{soc:.2f}"
        frame = _run_experiment(steps, soc, config, name)
        frame["initial_soc"] = soc
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _segments_to_steps(segments: Sequence[ProfileSegment]) -> list[str]:
    """把易读的 YAML 工况段转换成 PyBaMM 实验语句。"""
    steps: list[str] = []
    for segment in segments:
        if segment.mode == "rest":
            steps.append(f"Rest for {segment.duration_s:g} seconds")
        else:
            steps.append(
                f"Charge at {segment.c_rate:g} C for {segment.duration_s:g} seconds"
            )
    return steps


def generate_dynamic_profile(
    segments: Sequence[ProfileSegment],
    initial_soc: float,
    config: PhaseTwoConfig,
    profile_name: str,
) -> pd.DataFrame:
    """生成热辨识或独立验证所需的多段动态电流轨迹。"""
    return _run_experiment(
        _segments_to_steps(segments), initial_soc, config, profile_name
    )

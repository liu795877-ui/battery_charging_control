"""Chen2020 高保真电芯模型与 CC–CV 基准仿真。

这里的 DFN 电化学模型相当于后续控制器面对的“虚拟电池”。第一阶段先用它
产生可靠的基准轨迹；后续 MPC 不直接使用这个复杂模型，而会使用待辨识的
简化模型，以降低在线计算量。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
import pybamm

from .checks import check_trajectory
from .config import PhaseOneConfig


def build_chen2020_model(
    config: PhaseOneConfig,
) -> tuple[pybamm.BaseModel, pybamm.ParameterValues]:
    """构建 Chen2020 DFN＋集总热模型及其参数。

    本项目统一规定“充电电流为正”。PyBaMM 则规定“放电电流为正”，因此
    从 PyBaMM 导出电流时必须反号；模型内部仍遵守 PyBaMM 自己的约定。
    """
    if config.battery.model.upper() != "DFN":
        raise ValueError("Phase one currently supports the DFN reference model only.")

    # DFN 描述固相扩散、电解液传输和电化学反应；lumped 表示用一个平均温度
    # 描述整个电芯。它比等效电路精细，适合作为当前的高保真“虚拟对象”。
    model = pybamm.lithium_ion.DFN(options={"thermal": config.battery.thermal_model})
    parameters = pybamm.ParameterValues(config.battery.parameter_set)
    # PyBaMM 使用绝对温度 K，而配置文件使用更直观的 °C，所以要加 273.15。
    parameters.update(
        {
            "Ambient temperature [K]": config.battery.ambient_temperature_c + 273.15,
            "Initial temperature [K]": config.battery.initial_temperature_c + 273.15,
        }
    )
    return model, parameters


def simulate_cccv(
    c_rate: float, config: PhaseOneConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """运行一个 C-rate 的 CC–CV 基线，并只保留到目标 SOC 的轨迹。

    先以恒定 C-rate 充电（CC），电压达到 4.2 V 后转为恒压（CV）。任一阶段
    达到目标 SOC 都立即终止；CV 阶段还保留电流截止条件作为保护。这样不会
    为比较 10%→80% 而继续求解目标之后的状态，也可避免后续数值失败导致
    已经算出的有效 CV 轨迹被整段丢弃。
    """
    pybamm.set_logging_level("ERROR")
    model, parameters = build_chen2020_model(config)
    voltage = config.constraints.maximum_voltage_v
    cutoff = config.baseline.cv_cutoff_c_rate
    period = config.control.control_interval_s
    # 这两行依次定义 CC 阶段和 CV 阶段。period 是结果采样间隔，也与第一版
    # 控制周期保持一致，但它不是数值求解器内部的固定积分步长。
    # 项目的 SOC 是由累计容量计算的。充电时 PyBaMM 的放电容量为负，因此
    # “目标容量差 + 放电容量”会从正数逐渐降到 0，正好可作为终止事件。
    target_charge_ah = (
        config.battery.target_soc - config.battery.initial_soc
    ) * config.battery.nominal_capacity_ah

    def target_soc_event(variables: dict[str, pybamm.Symbol]) -> pybamm.Symbol:
        return target_charge_ah + variables["Discharge capacity [A.h]"]

    target_termination = pybamm.step.CustomTermination(
        "Target SOC reached", target_soc_event
    )

    # 为两个阶段都添加目标 SOC 终止条件。2 小时只是防止异常情况下无限运行，
    # 正常工况会更早由电压、目标 SOC 或 CV 电流截止事件终止。
    cc_step = pybamm.step.c_rate(
        -c_rate,
        duration="2 hours",
        termination=[f"{voltage:g} V", target_termination],
        period=period,
        description=f"Charge at {c_rate:g}C until voltage or target SOC",
    )
    cv_step = pybamm.step.voltage(
        voltage,
        duration="2 hours",
        termination=[f"{cutoff:g} C", target_termination],
        period=period,
        description=f"Hold at {voltage:g}V until target SOC or current cutoff",
    )
    experiment = pybamm.Experiment([cc_step, cv_step])
    simulation = pybamm.Simulation(
        model,
        parameter_values=parameters,
        experiment=experiment,
    )
    solution = simulation.solve(initial_soc=config.battery.initial_soc)
    frame = _solution_to_frame(solution, c_rate, config)

    # 找到首次越过目标 SOC 的采样点；若不存在，则该工况没有完成充电任务。
    target_hits = np.flatnonzero(frame["soc"].to_numpy() >= config.battery.target_soc)
    reached_target = bool(target_hits.size)
    if reached_target:
        frame = frame.iloc[: int(target_hits[0]) + 1].copy()

    # 完整性检查用于发现导出或符号转换错误；约束超限则单独记入指标，
    # 因为“温度超限”是有意义的仿真结果，不等于程序运行失败。
    checks = check_trajectory(frame)
    termination = str(solution.termination)
    target_event_hit = "Target SOC reached" in termination
    cutoff_event_hit = "C-rate cut-off" in termination
    if target_event_hit:
        simulation_status = "target_soc_reached"
    elif cutoff_event_hit:
        simulation_status = "cv_cutoff_before_target_soc"
    else:
        # 例如求解器只返回上一完整步骤时，终止原因可能仍显示为 CC 电压事件。
        # 单独标记它，避免把数值中断误写成“电池物理上无法达到目标”。
        simulation_status = "unexpected_or_partial_termination"

    metrics: dict[str, Any] = {
        "c_rate": float(c_rate),
        "reached_target_soc": reached_target,
        "target_soc": config.battery.target_soc,
        "final_soc": float(frame["soc"].iloc[-1]),
        "charge_time_s": float(frame["time_s"].iloc[-1]) if reached_target else None,
        "charge_time_min": (
            float(frame["time_s"].iloc[-1] / 60.0) if reached_target else None
        ),
        "maximum_voltage_v": float(frame["terminal_voltage_v"].max()),
        "maximum_temperature_c": float(frame["cell_temperature_c"].max()),
        "maximum_charge_current_a": float(frame["charge_current_a"].max()),
        "voltage_limit_exceeded": bool(
            frame["terminal_voltage_v"].max()
            > config.constraints.maximum_voltage_v + 1.0e-3
        ),
        "temperature_limit_exceeded": bool(
            frame["cell_temperature_c"].max() > config.constraints.maximum_temperature_c
        ),
        "current_limit_exceeded": bool(
            frame["charge_current_a"].max()
            > config.constraints.maximum_current_a + 1.0e-6
        ),
        "simulation_status": simulation_status,
        "solution_termination": termination,
        "returned_cycle_count": len(solution.cycles),
        "trajectory_checks": checks,
        "configuration": {
            "battery": asdict(config.battery),
            "constraints": asdict(config.constraints),
            "control": asdict(config.control),
        },
    }
    return frame, metrics


def _solution_to_frame(
    solution: pybamm.Solution, c_rate: float, config: PhaseOneConfig
) -> pd.DataFrame:
    """把 PyBaMM 解对象整理成易读、易保存的表格。

    每一行对应一个采样时刻；列名都带有物理含义和单位。该函数也是项目
    电流符号和 SOC 定义的唯一转换入口，后续模块不再接触 PyBaMM 原始符号。
    """
    time_s = np.asarray(solution["Time [s]"].entries, dtype=float).reshape(-1)
    pybamm_current_a = np.asarray(solution["Current [A]"].entries, dtype=float).reshape(
        -1
    )
    voltage_v = np.asarray(
        solution["Terminal voltage [V]"].entries, dtype=float
    ).reshape(-1)
    discharge_capacity_ah = np.asarray(
        solution["Discharge capacity [A.h]"].entries, dtype=float
    ).reshape(-1)
    temperature_c = (
        np.asarray(
            solution["Volume-averaged cell temperature [K]"].entries,
            dtype=float,
        ).reshape(-1)
        - 273.15
    )
    # PyBaMM 的“放电容量”在充电时为负，因此下面的公式会使 SOC 随充电上升：
    # SOC = 初始 SOC - 累计放电容量 / 标称容量。
    nominal_capacity_ah = config.battery.nominal_capacity_ah
    soc = config.battery.initial_soc - discharge_capacity_ah / nominal_capacity_ah

    frame = pd.DataFrame(
        {
            "time_s": time_s,
            # PyBaMM：放电为正；本项目：充电为正。因此这里乘以 -1。
            "charge_current_a": -pybamm_current_a,
            "terminal_voltage_v": voltage_v,
            "soc": soc,
            "cell_temperature_c": temperature_c,
            "ambient_temperature_c": config.battery.ambient_temperature_c,
            "discharge_capacity_ah": discharge_capacity_ah,
            "commanded_c_rate": float(c_rate),
        }
    )
    # CC 切换到 CV 时，两个实验步骤可能共享同一时刻。只保留后一条记录，
    # 避免时间重复影响绘图、插值以及“时间严格递增”检查。
    return frame.drop_duplicates(subset="time_s", keep="last").reset_index(drop=True)

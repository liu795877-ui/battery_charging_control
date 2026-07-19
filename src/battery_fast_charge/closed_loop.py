"""把 MPC 教师分别接到降阶模型和 Chen2020 DFN 上做闭环仿真。"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
import pybamm

from .mpc import ConstrainedMPC, ReducedBatteryModel, ReducedState
from .phase3_config import PhaseThreeConfig


def initial_reduced_state(config: PhaseThreeConfig) -> ReducedState:
    """第一版从平衡极化状态和均匀 25 ℃开始。"""
    return ReducedState(
        soc=config.battery.initial_soc,
        polarization_fast_v=0.0,
        polarization_slow_v=0.0,
        core_temperature_c=config.battery.initial_temperature_c,
        surface_temperature_c=config.battery.initial_temperature_c,
        previous_current_a=0.0,
    )


def _initial_record(
    model: ReducedBatteryModel, state: ReducedState, source: str
) -> dict[str, Any]:
    """统一两类闭环轨迹在 t=0 的列结构。"""
    return {
        "time_s": 0.0,
        "charge_current_a": 0.0,
        "soc": state.soc,
        "terminal_voltage_v": model.ocv(state.soc),
        "average_temperature_c": model.average_temperature(state),
        "predicted_maximum_voltage_v": np.nan,
        "predicted_maximum_temperature_c": np.nan,
        "predicted_terminal_soc": np.nan,
        "optimizer_success": True,
        "prediction_feasible": True,
        "used_fallback": False,
        "target_current_cap_active": False,
        "control_decision_updated": False,
        "solve_time_s": 0.0,
        "minimum_constraint_margin": np.nan,
        "source": source,
    }


def _cap_current_at_target(
    requested_current_a: float, soc: float, config: PhaseThreeConfig
) -> tuple[float, bool]:
    """最后一个周期只补足剩余电量，避免离散采样导致 SOC 明显越过 80%。"""
    remaining_ah = max(config.battery.target_soc - soc, 0.0) * config.battery.nominal_capacity_ah
    final_step_current_a = remaining_ah * 3600.0 / config.control.control_interval_s
    applied = min(float(requested_current_a), final_step_current_a)
    return applied, applied < requested_current_a - 1.0e-9


def simulate_reduced_closed_loop(
    model: ReducedBatteryModel, config: PhaseThreeConfig
) -> pd.DataFrame:
    """先在 MPC 自己的模型上完成最小可运行闭环，用于诊断优化器。"""
    controller = ConstrainedMPC(model, config)
    state = initial_reduced_state(config)
    records = [_initial_record(model, state, "reduced_model")]
    result = None
    steps_until_reoptimization = 0
    number_of_steps = int(
        np.ceil(
            config.control.maximum_simulation_time_s
            / config.control.control_interval_s
        )
    )

    for step_index in range(1, number_of_steps + 1):
        decision_updated = steps_until_reoptimization <= 0 or result is None
        if decision_updated:
            previous_current = state.previous_current_a
            result = controller.solve(state)
            # 若首个动作正好受变化率约束，则 5 s 后立刻重算以继续平滑升流；
            # 否则首个分块本来就规定保持 control_block_steps 个周期。
            hits_slew_limit = (
                abs(result.current_a - previous_current)
                >= 0.95
                * config.constraints.maximum_current_change_a_per_step
            )
            steps_until_reoptimization = (
                1 if hits_slew_limit else config.control.control_block_steps
            )
        assert result is not None
        applied_current, target_cap_active = _cap_current_at_target(
            result.current_a, state.soc, config
        )
        state, output = model.step(state, applied_current)
        records.append(
            {
                "time_s": step_index * config.control.control_interval_s,
                "charge_current_a": applied_current,
                "soc": state.soc,
                "terminal_voltage_v": output.terminal_voltage_v,
                "average_temperature_c": output.average_temperature_c,
                "predicted_maximum_voltage_v": result.predicted_maximum_voltage_v,
                "predicted_maximum_temperature_c": result.predicted_maximum_temperature_c,
                "predicted_terminal_soc": result.predicted_terminal_soc,
                "optimizer_success": result.optimizer_success,
                "prediction_feasible": result.prediction_feasible,
                "used_fallback": result.used_fallback,
                "target_current_cap_active": target_cap_active,
                "control_decision_updated": decision_updated,
                "solve_time_s": result.solve_time_s if decision_updated else 0.0,
                "minimum_constraint_margin": result.minimum_constraint_margin,
                "source": "reduced_model",
            }
        )
        steps_until_reoptimization -= 1
        if state.soc >= config.battery.target_soc - config.validation.target_soc_tolerance:
            break
    return pd.DataFrame.from_records(records)


class Chen2020DFNPlant:
    """可每 5 s 接收新电流的 Chen2020 DFN＋集总热虚拟电池。"""

    def __init__(self, config: PhaseThreeConfig) -> None:
        os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
        pybamm.set_logging_level("ERROR")
        model = pybamm.lithium_ion.DFN(
            options={"thermal": config.battery.thermal_model}
        )
        parameters = pybamm.ParameterValues(config.battery.parameter_set)
        parameters.update(
            {
                "Ambient temperature [K]": config.battery.ambient_temperature_c
                + 273.15,
                "Initial temperature [K]": config.battery.initial_temperature_c
                + 273.15,
                # PyBaMM 规定放电为正，控制器规定充电为正，因此 step 时传入负值。
                "Current function [A]": pybamm.InputParameter(
                    "pybamm_applied_current_a"
                ),
            }
        )
        self.config = config
        self.simulation = pybamm.Simulation(
            model,
            parameter_values=parameters,
            solver=pybamm.CasadiSolver(mode="safe"),
        )
        self.simulation.build(initial_soc=config.battery.initial_soc)

    def step(self, charge_current_a: float) -> dict[str, float]:
        """推进一个控制周期并只导出本项目需要的测量量。"""
        solution = self.simulation.step(
            self.config.control.control_interval_s,
            inputs={"pybamm_applied_current_a": -float(charge_current_a)},
            # 轨迹已经由本模块逐行保存；PyBaMM 这里只保留当前步和终端状态。
            # 否则每一步读取变量都会反复处理越来越长的历史 Solution。
            save=False,
        )

        def last(name: str) -> float:
            return float(np.asarray(solution[name].entries).reshape(-1)[-1])

        discharge_capacity_ah = last("Discharge capacity [A.h]")
        return {
            "time_s": last("Time [s]"),
            "soc": self.config.battery.initial_soc
            - discharge_capacity_ah / self.config.battery.nominal_capacity_ah,
            "terminal_voltage_v": last("Terminal voltage [V]"),
            "average_temperature_c": last(
                "Volume-averaged cell temperature [C]"
            ),
        }


def _correct_reduced_state_from_dfn(
    predicted_state: ReducedState,
    dfn_measurement: dict[str, float],
    model: ReducedBatteryModel,
    applied_current_a: float,
) -> ReducedState:
    """用 DFN 的 SOC 和平均温度纠正 MPC 内部状态。

    DFN 集总热模型没有核心/表面温差。这里仅把两个潜在温度同时平移，使其加权
    平均值等于 DFN 测量；不会把两个节点解释成已经验证的真实空间温度。
    """
    temperature_residual = (
        dfn_measurement["average_temperature_c"]
        - model.average_temperature(predicted_state)
    )
    return ReducedState(
        soc=float(dfn_measurement["soc"]),
        polarization_fast_v=predicted_state.polarization_fast_v,
        polarization_slow_v=predicted_state.polarization_slow_v,
        core_temperature_c=predicted_state.core_temperature_c
        + temperature_residual,
        surface_temperature_c=predicted_state.surface_temperature_c
        + temperature_residual,
        previous_current_a=float(applied_current_a),
    )


def simulate_dfn_closed_loop(
    model: ReducedBatteryModel, config: PhaseThreeConfig
) -> pd.DataFrame:
    """将 MPC 第一动作逐步施加到独立 DFN 虚拟电池并反馈 SOC、平均温度。"""
    controller = ConstrainedMPC(model, config)
    plant = Chen2020DFNPlant(config)
    state = initial_reduced_state(config)
    records = [_initial_record(model, state, "chen2020_dfn")]
    result = None
    steps_until_reoptimization = 0
    number_of_steps = int(
        np.ceil(
            config.control.maximum_simulation_time_s
            / config.control.control_interval_s
        )
    )

    for _ in range(number_of_steps):
        decision_updated = steps_until_reoptimization <= 0 or result is None
        if decision_updated:
            previous_current = state.previous_current_a
            result = controller.solve(state)
            hits_slew_limit = (
                abs(result.current_a - previous_current)
                >= 0.95
                * config.constraints.maximum_current_change_a_per_step
            )
            steps_until_reoptimization = (
                1 if hits_slew_limit else config.control.control_block_steps
            )
        assert result is not None
        applied_current, target_cap_active = _cap_current_at_target(
            result.current_a, state.soc, config
        )
        predicted_state, _ = model.step(state, applied_current)
        measurement = plant.step(applied_current)
        state = _correct_reduced_state_from_dfn(
            predicted_state, measurement, model, applied_current
        )
        records.append(
            {
                **measurement,
                "charge_current_a": applied_current,
                "predicted_maximum_voltage_v": result.predicted_maximum_voltage_v,
                "predicted_maximum_temperature_c": result.predicted_maximum_temperature_c,
                "predicted_terminal_soc": result.predicted_terminal_soc,
                "optimizer_success": result.optimizer_success,
                "prediction_feasible": result.prediction_feasible,
                "used_fallback": result.used_fallback,
                "target_current_cap_active": target_cap_active,
                "control_decision_updated": decision_updated,
                "solve_time_s": result.solve_time_s if decision_updated else 0.0,
                "minimum_constraint_margin": result.minimum_constraint_margin,
                "source": "chen2020_dfn",
            }
        )
        steps_until_reoptimization -= 1
        if state.soc >= config.battery.target_soc - config.validation.target_soc_tolerance:
            break
    return pd.DataFrame.from_records(records)


def closed_loop_metrics(
    frame: pd.DataFrame, config: PhaseThreeConfig
) -> dict[str, Any]:
    """把一条闭环轨迹压缩成可与 CC–CV 对比的验收指标。"""
    tolerance = config.validation.physical_constraint_tolerance
    reached_target = bool(
        frame["soc"].iloc[-1]
        >= config.battery.target_soc - config.validation.target_soc_tolerance
    )
    control_rows = frame.iloc[1:]
    optimization_rows = control_rows[control_rows["control_decision_updated"]]
    optimizer_success_fraction = float(optimization_rows["optimizer_success"].mean())
    feasible_fraction = float(optimization_rows["prediction_feasible"].mean())
    metrics = {
        "source": str(frame["source"].iloc[-1]),
        "reached_target_soc": reached_target,
        "final_soc": float(frame["soc"].iloc[-1]),
        "charge_time_s": float(frame["time_s"].iloc[-1]) if reached_target else None,
        "charge_time_min": (
            float(frame["time_s"].iloc[-1] / 60.0) if reached_target else None
        ),
        "maximum_voltage_v": float(frame["terminal_voltage_v"].max()),
        "maximum_temperature_c": float(frame["average_temperature_c"].max()),
        "maximum_current_a": float(frame["charge_current_a"].max()),
        "voltage_limit_exceeded": bool(
            frame["terminal_voltage_v"].max()
            > config.constraints.physical_maximum_voltage_v + tolerance
        ),
        "temperature_limit_exceeded": bool(
            frame["average_temperature_c"].max()
            > config.constraints.physical_maximum_temperature_c + tolerance
        ),
        "current_limit_exceeded": bool(
            frame["charge_current_a"].max()
            > config.constraints.maximum_current_a + tolerance
        ),
        "optimizer_success_fraction": optimizer_success_fraction,
        "prediction_feasible_fraction": feasible_fraction,
        "optimization_count": int(len(optimization_rows)),
        "fallback_count": int(optimization_rows["used_fallback"].sum()),
        "mean_mpc_solve_time_ms": float(optimization_rows["solve_time_s"].mean() * 1000.0),
        "maximum_mpc_solve_time_ms": float(optimization_rows["solve_time_s"].max() * 1000.0),
        "configuration": asdict(config),
    }
    metrics["success"] = bool(
        reached_target
        and not metrics["voltage_limit_exceeded"]
        and not metrics["temperature_limit_exceeded"]
        and not metrics["current_limit_exceeded"]
        and optimizer_success_fraction
        >= config.validation.minimum_optimizer_success_fraction
        and feasible_fraction >= config.validation.minimum_optimizer_success_fraction
    )
    return metrics
